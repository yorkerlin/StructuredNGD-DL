from models.cifar import (
                          vgg16_bn,
                          convmixer12,
                          pyramidnet65,
                          regnet_1600m,
                          )

from models.imagenet import convmixer_180_16 #for imagenet
from timm.models.regnet import regnetz_005 #for imagenet
from timm.models.byobnet import repvgg_b1g4 #for imagenet
from timm.models.rexnet import rexnetr_100 #for imagenet
import torch

def my_rexnetr_100(num_classes, **kwargs):
    model = rexnetr_100(pretrained=False, num_classes=num_classes)
    return torch.nn.DataParallel(model)

def regnetz_500m(num_classes, **kwargs):
    model = regnetz_005(pretrained=False, num_classes=num_classes)
    return torch.nn.DataParallel(model)

def my_repvgg_b1g4(num_classes, **kwargs):
    model = repvgg_b1g4(pretrained=False, num_classes=num_classes)
    return torch.nn.DataParallel(model)

def get_network(network, **kwargs):
    networks = {
        'vgg16_bn': vgg16_bn,
        'convmixer12': convmixer12, #convmixer_256_12
        'pyramidnet65': pyramidnet65, 
        'regnet_1600m': regnet_1600m,
        'regnetz_500m': regnetz_500m,
        'repvgg_b1g4': my_repvgg_b1g4,
        'rexnetr_100': my_rexnetr_100,
        'convmixer_180_16': convmixer_180_16,
    }

    return networks[network](**kwargs)

