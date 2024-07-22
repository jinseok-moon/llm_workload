import numpy as np

class A100:
    def __init__(self):
        self.dram_bandwidth = 1.935e+12  # 1.94 TB/s
        self.SM = 108
        self.tensor_core = 4
        self.cuda_core = 64
        self.clock = 1410e+6
        self.memory_clock = 1512e+6
        
        
class Tensor:
    def __init__(self, shape=None, type=None) -> None:
        self.shape = shape
        self.type = type
    
    def get_size(self):
        return self.shape * self.type
    

