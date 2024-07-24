import numpy as np
import torch
from ops import *
from kernels import *

LlamaConfig = {
  "_name_or_path": "meta-llama/Meta-Llama-3-8B",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "batch_size": 8,
  "attention_bias": False,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": 128001,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "llama",
  "num_attention_heads": 32,
  "head_dim": 128,
  "num_hidden_layers": 32,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": None,
  "rope_theta": 500000.0,
  "tie_word_embeddings": False,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.36.2",
  "use_cache": True,
  "vocab_size": 128256
}

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
        self.compute_clock = 1.410e+9
        self.memory_clock = 1.512e+9
    

if __name__ == "__main__":
    _batch = 8
    _input_len = 2048
    _output_len = 2048
    _dtype = torch.float16
    _input = torch.empty(_batch, _input_len, dtype=_dtype)
    llama_3_8b = Llama3_8B(LlamaConfig, _output_len)
    output, ops = llama_3_8b(_input)
    print(ops)