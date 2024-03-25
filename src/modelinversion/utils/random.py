import torch
import numpy as np
import random
import time


def set_random_seed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
_ALL_LOGITS = '0123456789qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
_ALL_LOGITS_INDICES = np.arange(len(_ALL_LOGITS), dtype=np.int32)

def get_random_string(length: int=6):
    
    seed = int(time.time() * 1000) % (2**30) ^ random.randint(0, 2**30)
    # print(seed)
    
    resindices = np.random.RandomState(seed).choice(_ALL_LOGITS_INDICES, length)
    return ''.join(map(lambda x: _ALL_LOGITS[x], resindices))