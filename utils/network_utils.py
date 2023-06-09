from models.cifar import (
                          vgg16_bn,
                          convmixer12,
                          pyramidnet65,
                          regnet_1600m,
                          )

from timm.models.regnet import regnetz_005 #for imagenet
from timm.models.byobnet import repvgg_b1g4 #for imagenet
import torch

def regnetz_500m(num_classes, **kwargs):
    model = regnetz_005(pretrained=False, num_classes=num_classes)
    return torch.nn.DataParallel(model)

def my_repvgg_b1g4(num_classes, **kwargs):
    model = repvgg_b1g4(pretrained=False, num_classes=num_classes)
    return torch.nn.DataParallel(model)

def get_network(network, **kwargs):
    networks = {
        'vgg16_bn': vgg16_bn,
        'convmixer12': convmixer12,
        'pyramidnet65': pyramidnet65,
        'regnet_1600m': regnet_1600m,
        'regnetz_500m': regnetz_500m,
        'repvgg_b1g4': my_repvgg_b1g4,
    }

    return networks[network](**kwargs)

