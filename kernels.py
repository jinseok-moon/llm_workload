import torch
        
# Class for calculate performance at hardware level
class Kernel:
    def __init__(self) -> None:
        self.cycle = 0

    def TC_GEMM(t_in: torch.Tensor, t_out: torch.Tensor):
        pass
    
    def get_cycle(self):
        pass
        
class GQA(Kernel):
    def __init__(self) -> None:
        super().__init__()
        pass
    