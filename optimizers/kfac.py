import math
import ipdb

import torch
import torch.optim as optim

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat


class KFACOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 stat_decay=0.95,
                 damping=0.001,
                 weight_decay=1e-2,
                 TCov=10,
                 TInv=10,
                 use_eign = False,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum, damping=damping,
                        weight_decay=weight_decay)

        # TODO (CW): KFAC optimizer now only support model as input
        super(KFACOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.use_eign = bool(use_eign) 
        if self.use_eign:
            print('kfac uses eign (unstable)')
        else:
            print('kfac uses mat inv')

        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0

        self.m_aa, self.m_gg = {}, {}
        self.Q_a, self.Q_g = {}, {}
        self.d_a, self.d_g = {}, {}
        self.stat_decay = stat_decay

        self.TCov = TCov
        self.TInv = TInv

    def _save_input(self, module, input):
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # if torch.isnan(aa).any():
                # print('NaN error')
            # Initialize buffers
            if self.steps == 0:
                self.m_aa[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
            #moving average on aa
            update_running_stat(aa, self.m_aa[module], self.stat_decay)

    def _save_grad_output(self, module, grad_input, grad_output):
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # if torch.isnan(gg).any():
                # print('NaN error')
            # Initialize buffers
            if self.steps == 0:
                self.m_gg[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
            #moving average on gg
            update_running_stat(gg, self.m_gg[module], self.stat_decay)

    def _prepare_model(self):
        count = 0
        # print(self.model)
        print("=> We keep following layers in KFAC. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers in KFAC. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                # print('(%s): %s' % (count, module))
                count += 1

    def _update_inv(self, m):
        """Do eigen decomposition for computing inverse of the ~ fisher.
        :param m: The layer
        :return: no returns.
        """
        eps = 1e-10  # for numerical stability
        self.d_a[m], self.Q_a[m] = torch.linalg.eigh(
            self.m_aa[m], UPLO='U')
        self.d_g[m], self.Q_g[m] = torch.linalg.eigh(
            self.m_gg[m], UPLO='U')

        self.d_a[m].mul_((self.d_a[m] > eps).float())
        self.d_g[m].mul_((self.d_g[m] > eps).float())

    def _inv_covs(self, m, damping):
        """Inverses the covariances."""
        # Regularizes and inverse
        eps = damping 
        diag_aat = self.m_aa[m].new(self.m_aa[m].shape[0]).fill_(eps)
        diag_ggt = self.m_gg[m].new(self.m_gg[m].shape[0]).fill_(eps)

        self.Q_a[m] = (self.m_aa[m] + torch.diag(diag_aat)).inverse()
        self.Q_g[m] = (self.m_gg[m] + torch.diag(diag_ggt)).inverse()
 

    @staticmethod
    def _get_matrix_form_grad(m, classname):
        """
        :param m: the layer
        :param classname: the class name of the layer
        :return: a matrix form of the gradient. it should be a [output_dim, input_dim] matrix.
        """
        if classname == 'Conv2d':
            p_grad_mat = m.weight.grad.data.view(m.weight.grad.data.size(0), -1)  # n_filters * (in_c * kw * kh)
        else:
            p_grad_mat = m.weight.grad.data
        if m.bias is not None:
            p_grad_mat = torch.cat([p_grad_mat, m.bias.grad.data.view(-1, 1)], 1)
        return p_grad_mat

    def _update_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # a @ b = matmul(a,b)
        # self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) == d_g @ d_a.t()
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]
        # print(self.Q_g[m].t().shape, self.Q_a[m].shape)
        # print(p_grad_mat.shape)

        if self.use_eign:
            v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
            v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
            v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        else:
            v = self.Q_g[m] @ p_grad_mat @ self.Q_a[m]

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            # v = [v[:, :-1], v[:, -1:]]
            m.weight.grad.data.copy_( v[:, :-1].view(m.weight.grad.data.size()) )
            m.bias.grad.data.copy_( v[:, -1:].view(m.bias.grad.data.size()) )
        else:
            m.weight.grad.data.copy_( v.view(m.weight.grad.data.size()) )

        return v


    def _get_natural_grad(self, m, p_grad_mat, damping):
        """
        :param m:  the layer
        :param p_grad_mat: the gradients in matrix form
        :return: a list of gradients w.r.t to the parameters in `m`
        """
        # a @ b = matmul(a,b)
        # self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) == d_g @ d_a.t()
        # p_grad_mat is of output_dim * input_dim
        # inv((ss')) p_grad_mat inv(aa') = [ Q_g (1/R_g) Q_g^T ] @ p_grad_mat @ [Q_a (1/R_a) Q_a^T]

        v1 = self.Q_g[m].t() @ p_grad_mat @ self.Q_a[m]
        v2 = v1 / (self.d_g[m].unsqueeze(1) * self.d_a[m].unsqueeze(0) + damping)
        v = self.Q_g[m] @ v2 @ self.Q_a[m].t()
        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            v = [v[:, :-1], v[:, -1:]]
            v[0] = v[0].view(m.weight.grad.data.size())
            v[1] = v[1].view(m.bias.grad.data.size())
        else:
            v = [v.view(m.weight.grad.data.size())]

        return v

    def _step(self, closure):
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data #grad
                if weight_decay != 0 and self.steps >= 20 * self.TCov:
                    d_p.add_(p.data, alpha=weight_decay) #add weight decay into grad
                if momentum != 0:#add momentum
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p,alpha=-group['lr']) #perform a SGD-like update

    def step(self, closure=None):
        group = self.param_groups[0]
        lr = group['lr']
        damping = group['damping']
        # updates = {}
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                if self.use_eign:
                    self._update_inv(m) #inverse the FIM approximation
                else:
                    self._inv_covs(m, damping)
            p_grad_mat = self._get_matrix_form_grad(m, classname)#reshape the Euclidean grad as a matrix
            self._update_natural_grad(m, p_grad_mat, damping)

        self._step(closure) #do riemannian norm and update
        self.steps += 1
