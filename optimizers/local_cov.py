import math
import numpy as np
import ipdb

import torch
import torch.optim as optim

from utils.kfac_utils import (ComputeCovA, ComputeCovG)
from utils.kfac_utils import update_running_stat



class LocalOptimizer(optim.Optimizer):
    def __init__(self,
                 model,
                 lr=0.001,
                 momentum=0.9,
                 damping=0.001,
                 beta2 = 0.5,
                 weight_decay=1e-2,
                 TCov=10,
                 TInv=10,
                 faster=True,
                 lr_cov=1e-2,
                 batch_averaged=True):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if beta2 < 0.0:
            raise ValueError("Invalid beta2: {}".format(beta2))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, momentum=momentum,
                        weight_decay=weight_decay)
        print('damping', damping)
        print('beta2', beta2)
        print('momentum', momentum)
        print('wd', weight_decay)
        print('lr_cov', lr_cov)

        self.damping = damping
        self.faster = bool(faster)
        if faster:
            assert( TCov == TInv )
            print('enable faster version')
        else:
            print('standard version')

        # TODO (CW): this optimizer now only support model as input
        super(LocalOptimizer, self).__init__(model.parameters(), defaults)
        self.CovAHandler = ComputeCovA()
        self.CovGHandler = ComputeCovG()
        self.batch_averaged = batch_averaged
        self.known_modules = {'Linear', 'Conv2d'}

        self.modules = []
        # self.grad_outputs = {}

        self.model = model
        self._prepare_model()

        self.steps = 0
        self.beta2 = beta2

        self.lr_cov = lr_cov
        self.aa, self.gg = {}, {}
        self.A, self.B = {}, {}
        self.m_A, self.m_B = {}, {}
        self.m_mu = {}

        self.TCov = TCov
        self.TInv = TInv

        group = self.param_groups[0]
        self.org_lr = group['lr']

    def _save_input(self, module, input):#ok
        if torch.is_grad_enabled() and self.steps % self.TCov == 0:
            aa = self.CovAHandler(input[0].data, module)
            # Initialize buffers
            if self.steps == 0:
                self.A[module] = torch.diag(aa.new(aa.size(0)).fill_(1))
                self.m_A[module] = torch.zeros_like(self.A[module])

            self.aa[module] = aa

    def _save_grad_output(self, module, grad_input, grad_output):#ok
        # Accumulate statistics for Fisher matrices
        if self.acc_stats and self.steps % self.TCov == 0:
            gg = self.CovGHandler(grad_output[0].data, module, self.batch_averaged)
            # Initialize buffers
            if self.steps == 0:
                self.B[module] = torch.diag(gg.new(gg.size(0)).fill_(1))
                self.m_B[module] = torch.zeros_like(self.B[module])

            self.gg[module] = gg

    def _prepare_model(self): #ok
        count = 0
        # print(self.model)
        print("=> We keep following layers in. ")
        for module in self.model.modules():
            classname = module.__class__.__name__
            # print('=> We keep following layers. <=')
            if classname in self.known_modules:
                self.modules.append(module)
                module.register_forward_pre_hook(self._save_input)
                module.register_backward_hook(self._save_grad_output)
                # print('(%s): %s' % (count, module))
                count += 1

    @staticmethod
    def _update_helper(delta, block):#for cov
        #compute block * h(delta)

        #2nd truncation
        #delta is symmetric
        #block @ h(delta)  = block @ ( I + (I+delta) @ (I+delta).t() )/2 
        # tmp = torch.eye(block.shape[0], device='cuda') + delta
        # return block @ (  (torch.eye(block.shape[0], device='cuda') + tmp @ tmp.t())/2.0 )

        #1st truncation
        tmp = block @  delta
        return block + tmp


    def _update_local(self, m):#check it !!!
        """our inverse FIM approximation
        :param m: The layer
        :return: no returns.
        """
        
        group = self.param_groups[0]
        damping = self.damping

        d = self.gg[m].size(0)
        p = self.aa[m].size(0)

        if self.faster:
            tr_B_tB = torch.sum( (self.B[m])**2 )
            tr_A_tA = torch.sum( (self.A[m])**2 )
            t1 = self.aa[m] @ self.A[m]
            t2 = self.gg[m] @ self.B[m]
            tr_v_a = torch.sum( self.A[m]*t1 )
            tr_v_b = torch.sum( self.B[m]*t2 )
            ng_a = (self.A[m].t() @ ((damping*tr_B_tB/d)*self.A[m] + (tr_v_b/d)*t1) - torch.eye(p, device='cuda') )/2.0
            ng_b = (self.B[m].t() @ ((damping*tr_A_tA/p)*self.B[m] + (tr_v_a/p)*t2) - torch.eye(d, device='cuda') )/2.0
        else:
            v_a = self.A[m].t() @  self.aa[m] @ self.A[m]
            v_b = self.B[m].t() @  self.gg[m] @ self.B[m]
            A_tA =  self.A[m].t() @  self.A[m]
            B_tB =  self.B[m].t() @  self.B[m]
            ng_a = ( (damping*torch.trace(B_tB)/d)*A_tA + (torch.trace(v_b)/d)*v_a - torch.diag(v_a.new(p).fill_(1.0)) )/2.0
            ng_b = ( (damping*torch.trace(A_tA)/p)*B_tB + (torch.trace(v_a)/p)*v_b - torch.diag(v_b.new(d).fill_(1.0)) )/2.0


        if self.steps<=100:#init phrase
            lr1 = 2e-4
        elif self.steps<500:#warm up
            lr1 = 2e-3
        else:
            lr1 = self.lr_cov
        
        beta2 = self.beta2
        #update A #p by p matrix
        self.m_A[m].mul_(beta2).add_(  ( ng_a+ng_a.t() )/2.0  ) 
        # self.A[m] = self._update_helper(-lr1*(self.m_A[m]), self.A[m])#for cov (must be negative)
        self.A[m].add_((self.A[m] @ self.m_A[m]), alpha=-lr1)

        #update B #d by d matrix
        self.m_B[m].mul_(beta2).add_( ( ng_b+ng_b.t() )/2.0 ) 
        # self.B[m] = self._update_helper(-lr1*(self.m_B[m]), self.B[m])#for cov (must be negative)
        self.B[m].add_((self.B[m] @ self.m_B[m]), alpha=-lr1)

        if self.faster:
            self.aa[m] = self.A[m] @ (self.A[m].t())
            self.gg[m] = self.B[m] @ (self.B[m].t())


    @staticmethod
    def _get_matrix_form_grad(m, classname): #ok
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



    def _update_natural_grad(self, m, p_grad_mat, damping, momentum):
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

        # (B B^T) p_grad_mat (A A^T)
        # v1 = (p_grad_mat @ self.A[m]) @ self.A[m].t()
        # v = self.B[m] @ (self.B[m].t() @ v1)

        # v1 = self.B[m].t() @ (p_grad_mat @ self.A[m]) 
        # v = self.B[m] @ (v1 @ self.A[m].t() )

        if self.faster:
            v = self.gg[m] @ p_grad_mat @ self.aa[m]
        else:
            v1 = (p_grad_mat @ self.A[m]) @ self.A[m].t()
            v = self.B[m] @ (self.B[m].t() @ v1)

        if m.bias is not None:
            # we always put gradient w.r.t weight in [0]
            # and w.r.t bias in [1]
            # v = [v[:, :-1], v[:, -1:]]
            m.weight.grad.data.copy_( v[:, :-1].view(m.weight.grad.data.size()) )
            m.bias.grad.data.copy_( v[:, -1:].view(m.bias.grad.data.size()) )
        else:
            m.weight.grad.data.copy_( v.view(m.weight.grad.data.size()) )

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
                    else:
                        buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p)
                    d_p = buf

                p.data.add_(d_p, alpha=-group['lr']) #perform a SGD-like update
    def step(self, closure=None):
        group = self.param_groups[0]
        momentum = group['momentum']
        lr = group['lr']
        damping = self.damping
        for m in self.modules:
            classname = m.__class__.__name__
            if self.steps % self.TInv == 0:
                self._update_local(m) #our inverse FIM approximation
            p_grad_mat = self._get_matrix_form_grad(m, classname)#reshape the Euclidean grad as a matrix
            self._update_natural_grad(m, p_grad_mat, damping, momentum)

        self._step(closure)
        self.steps += 1
