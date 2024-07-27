import torch
    
class A100:
    dram_bandwidth = 1.935e+12  # 1.94 TB/s
    l2_brandwith = 3.8e+12 # 3.8 TB/s from paper (measrued)
    
    SM = 108
    tensor_core = 4
    cuda_core = 64
    peak_performance = 312e+12  # 312 TFLOPS for fp16 tensor core
    peak_performance_cuda_core = 77.97e+12  # 78 TFLOPS for fp16 cuda core
    compute_clock = 1.410e+9
    memory_clock = 1.512e+9   # 1.283 Byte per clock ?
    flop_clock_tc = 512  # 1 tc operate 256 FMA (512 FLOPs) per clock
    flop_clock_sm = flop_clock_tc * tensor_core
    flop_clock = flop_clock_sm * SM
    
    l2_size = 40e+6 # 40 MB
    max_register_per_sm = 256e+3 # 256 KB
    max_register_total = max_register_per_sm * SM 
        
        
class Config:
    batch_size = 8
    input_seq_len = 2048
    output_seq_len = 2048
    act_dtype = torch.float16
    weight_dtype = torch.float16
    device = A100()
        
class LlamaConfig(Config):
    num_attention_heads = 32
    num_key_value_heads = 8
    num_hidden_layers = 32
    head_dim = 128
    hidden_size = 4096
    im_size = 14336
    vocab_size = 128256
        

GLOBAL_CONFIG = LlamaConfig()


LlamaConfig_Dict = {
  "_name_or_path": "meta-llama/Meta-Llama-3-8B",
  "architectures": [
    "LlamaForCausalLM"
  ],
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
