from enum import Enum

class TqdmStrategy(Enum):
    NONE = 'none'
    EPOCH = 'epoch'
    ITER = 'iter'
    
