from dataclasses import dataclass
import torch
from typing import overload

class ModelResultIter:
    def __init__(self, modelresult) -> None:
        self.model_result = modelresult
        self.mod = 'result'
        
    def __next__(self):
        if self.mod == 'result':
            self.mod = 'feat'
            return self.model_result.result
        elif self.mod == 'feat':
            self.mod = 'addition'
            return self.model_result.feat
        elif self.mod == 'addition':
            self.mod = 'stop'
            return self.model_result.addition_info
        else:
            raise StopIteration()
        
        
class ModelResult:
    result: torch.Tensor
    feat: list
    addition_info: dict = None
    
    # @overload
    def __init__(self, result: torch.Tensor, feat: list = None, addition_info: dict = None) -> None:
        
        # support for DataParallel
        if isinstance(result, map):
            ls = list(result)
            return self.__init__(*ls)
        
        self.result = result
        self.feat = feat
        self.addition_info = addition_info
        
    # @overload
    # def __init__(self, m: map):
    #     self.__init__(*list(m))
    #     print('--------')
    #     print(*list(m))
    
    def __iter__(self):
        return ModelResultIter(self)