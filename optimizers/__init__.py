from .kfac import KFACOptimizer
from .local_cov import LocalOptimizer


def get_optimizer(name):
    if name == 'kfac':
        return KFACOptimizer
    elif name == 'local':
        return LocalOptimizer
    else:
        raise NotImplementedError
