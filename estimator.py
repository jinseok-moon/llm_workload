import numpy as np
import torch
from ops import *
from kernels import *
from config import *

class Accelerator:
    def __init__(self) -> None:
        pass
class A100(Accelerator):
    def __init__(self):
        super.__init__()
        self.dram_bandwidth = 1.935e+12  # 1.94 TB/s
        self.SM = 108
        self.tensor_core = 4
        self.cuda_core = 64
        self.peak_performance = 312e+12  # 312 TFLOPS for fp16 tensor core
        self.compute_clock = 1.410e+9
        self.memory_clock = 1.512e+9   # 1.283 Byte per clock ?
        self.flop_clock_tc = 512  # 1 tc operate 256 FMA (512 FLOPs) per clock
        self.flop_clock_sm = self.flop_clock_tc * self.tensor_core
        self.flop_clock = self.flop_clock_sm * self.SM
        self.max_register_per_sm = 256e+3 # 256 KB
        self.max_register_total = self.max_register_per_sm * self.SM 
        
        # mma m16n8k16 fp16*fp16 -> 2(inst) * 8 (Cylcle) => 16 compute cycle
        
        # The A100 L2 read bandwidth is 5120 Bytes/clk
        # The GPU is operating at a frequency of 1065 MHz, which can be boosted up to 1410 MHz, memory is running at 1512 MHz.
        # From https://www.techpowerup.com/gpu-specs/a100-pcie-80-gb.c3821
        # https://forums.developer.nvidia.com/t/how-to-calculate-the-tensor-core-fp16-performance-of-h100/244727



if __name__ == "__main__":
    _input = torch.empty(GLOBAL_CONFIG.batch_size, GLOBAL_CONFIG.input_seq_len, dtype=GLOBAL_CONFIG.act_dtype)
    
    print("Naive Self Attention")
    llama_3_8b = Llama3_8B(use_flash_attention=False)
    output = llama_3_8b(_input, output_len=GLOBAL_CONFIG.output_seq_len)
    
    print("Flash Attention")
    llama_3_8b = Llama3_8B(use_flash_attention=True)
    output = llama_3_8b(_input, output_len=GLOBAL_CONFIG.output_seq_len)
    
    