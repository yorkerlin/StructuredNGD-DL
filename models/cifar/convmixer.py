'''
'''
import torch.nn as nn


__all__ = ['convmixer8', 'convmixer12'
           ]



class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


# https://stackoverflow.com/questions/62166719/padding-same-conversion-to-pytorch-padding
#same padding implies out == in
# out = (in-F+2P)/S+1 #by default S (stride = 1)
# (F-1)/2
# https://forums.fast.ai/t/solved-how-to-easily-do-same-convolution/64069/3

def ConvMixer(dim, depth, kernel_size=5, patch_size=2, n_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    # nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=int( (kernel_size-1)/2 ) ),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
        nn.Linear(dim, n_classes)
    )

class MyConvMixer(nn.Module):

    def __init__(self, depth,  num_classes=10, **kwargs):
        super(MyConvMixer, self).__init__()
        self.model = ConvMixer(
            dim = 256,
            depth = depth, #8
            patch_size = 2,
            # kernel_size = 3,
            kernel_size = 5,
            n_classes = num_classes,
            )

    def forward(self, x):
        return self.model(x)


def convmixer8(num_classes,**kwargs):
    model = MyConvMixer(num_classes=num_classes, depth=8)
    return model

def convmixer12(num_classes,**kwargs):
    model = MyConvMixer(num_classes=num_classes, depth=12)
    return model


