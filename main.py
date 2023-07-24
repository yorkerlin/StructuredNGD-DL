import argparse
import time
import ipdb
import os
from optimizers import LocalOptimizer, KFACOptimizer
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import joblib
from lion_pytorch import Lion

from tqdm import tqdm
from utils.network_utils import get_network
from utils.data_utils import get_dataloader

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# fetch args
parser = argparse.ArgumentParser()


parser.add_argument('--network', default='vgg16_bn', type=str)
parser.add_argument('--dataset', default='cifar10', type=str)

parser.add_argument('--device', default='cuda', type=str)
parser.add_argument('--resume', '-r', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--log_dir', default='runs/pretrain', type=str)

parser.add_argument('--beta2', default=0.5, type=float)
parser.add_argument('--faster', default=0, type=int)

parser.add_argument('--optimizer', default='kfac', type=str)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--epoch', default=100, type=int)
parser.add_argument('--milestone', default=None, type=str)
parser.add_argument('--learning_rate', default=0.01, type=float)
parser.add_argument('--momentum', default=0.9, type=float)
parser.add_argument('--stat_decay', default=0.95, type=float)
parser.add_argument('--damping', default=5e-3, type=float)
parser.add_argument('--weight_decay', default=3e-3, type=float)
parser.add_argument('--TCov', default=10, type=int)
parser.add_argument('--TScal', default=10, type=int)
parser.add_argument('--TInv', default=10, type=int)
parser.add_argument('--use_eign', default=0, type=int)
parser.add_argument('--run_id', default=1, type=int)
parser.add_argument('--lr_cov', default=1e-2, type=float)


parser.add_argument('--prefix', default=None, type=str)
args = parser.parse_args()

# init model
nc = {
    'tinyimagenet': 200,
    'imagenet100': 100,
    'cifar100': 100,
}
num_classes = nc[args.dataset]
net = get_network(args.network,
                  num_classes=num_classes)
net = net.to(args.device)

# init dataloader
trainloader, testloader = get_dataloader(dataset=args.dataset,
                                         train_batch_size=args.batch_size,
                                         test_batch_size=256)

# init optimizer and lr scheduler
optim_name = args.optimizer.lower()
tag = optim_name
data_name = args.dataset
model_name = args.network
print( optim_name )
print( count_parameters(net) )

if optim_name == 'sgd':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
elif optim_name == 'sgd_nol2':
    optimizer = optim.SGD(net.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          )
elif optim_name == 'kfac':
    optimizer = KFACOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              stat_decay=args.stat_decay,
                              damping=args.damping,
                              weight_decay=args.weight_decay,
                              TCov=args.TCov,
                              use_eign = args.use_eign,
                              TInv=args.TInv)
elif optim_name == 'adamw':
    optimizer = optim.AdamW(net.parameters(),
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)

elif optim_name == 'adam':
    optimizer = optim.Adam(net.parameters(),
                          lr=args.learning_rate,
                          weight_decay=args.weight_decay)

elif optim_name == 'adam_nol2':
    optimizer = optim.Adam(net.parameters(),
                          lr=args.learning_rate,
                          )
elif optim_name == 'local':
    optimizer = LocalOptimizer(net,
                              lr=args.learning_rate,
                              momentum=args.momentum,
                              damping=args.damping,
                              beta2 = args.beta2,
                              weight_decay=args.weight_decay,
                              faster = args.faster,
                              TCov=args.TCov,
                              lr_cov=args.lr_cov,
                              TInv=args.TInv)

else:
    raise NotImplementedError

if args.milestone is None:
    lr_scheduler = MultiStepLR(optimizer, milestones=[int(args.epoch*0.5), int(args.epoch*0.75)], gamma=0.1)
else:
    milestone = [int(_) for _ in args.milestone.split(',')]
    lr_scheduler = MultiStepLR(optimizer, milestones=milestone, gamma=0.1)

# init criterion
criterion = nn.CrossEntropyLoss()

start_epoch = 0
if args.resume:
    print('==> Resuming from checkpoint..')
    assert os.path.isfile(args.load_path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(args.load_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    print('==> Loaded checkpoint at epoch: %d, acc: %.2f%%' % (start_epoch, best_acc))

log_dir = os.path.join(args.log_dir, args.dataset, args.network, args.optimizer,
                       'lr%.3f_wd%.4f_damping%.4f' %
                       (args.learning_rate, args.weight_decay, args.damping))
if not os.path.isdir(log_dir):
    os.makedirs(log_dir)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    desc = ('[%s][LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (tag, lr_scheduler.get_last_lr()[0], 0, 0, correct, total))


    lr_scheduler.step()
    prog_bar = tqdm(enumerate(trainloader), total=len(trainloader), desc=desc, leave=True)

    batch_time = 0.0
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.to(args.device), targets.to(args.device)
        end = time.time()
        optimizer.zero_grad()

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        if optim_name in ['kfac', 'local'] and optimizer.steps % optimizer.TCov == 0:
            optimizer.acc_stats = True
            ################
            # compute true fisher
            # with torch.no_grad():
                # sampled_y = torch.multinomial(torch.nn.functional.softmax(outputs.cpu().data, dim=1),
                                              # 1).squeeze().cuda()
            # loss_sample = criterion(outputs, sampled_y)
            # loss_sample.backward(retain_graph=True)
            # optimizer.acc_stats = False
            # optimizer.zero_grad()  # clear the gradient for computing true-fisher.
            # loss.backward()
            ################

            # compute emprical fisher
            loss.backward()
            optimizer.acc_stats = False
            ################
        else:
            loss.backward()
        optimizer.step()
        torch.cuda.current_stream().synchronize()
        batch_time += (time.time() - end)

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[%s][%s][LR=%s][%s][%f] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (data_name, tag, lr_scheduler.get_last_lr()[0], model_name, batch_time, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)
    return batch_time


def test(epoch, info):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('[%s]Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (tag,test_loss/(0+1), 0, correct, total))

    prog_bar = tqdm(enumerate(testloader), total=len(testloader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('[%s]Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (tag, test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    acc = 100.*correct/total
    info.setdefault(epoch, acc)


def main():
    print( optim_name, args.learning_rate, args.beta2, args.momentum )
    info ={}
    time_info = {}
    cur_time = 0.0
    for epoch in range(start_epoch, args.epoch):
        batch_time = train(epoch)
        cur_time += batch_time
        time_info.setdefault(epoch, cur_time)
        test(epoch, info)



if __name__ == '__main__':
    main()


