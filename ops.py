import torch
from kernels import *
from config import *
from collections import defaultdict
import numpy as np
class Model:
    def __init__(self, name=None) -> None:
        self.name = name
        self.result = dict({
            "name": self.name,
            "inference_time": 0,
            "ops": 0,
            "mem": 0,
            "arith_intensity": 0,
            "performance": 0,
            "mem_bandwidth": 0,
            "bound": "",
            "children": []
        })
        self.compute_cycle = 0
        self.memory_cycle = 0
        self.max_perf = GLOBAL_CONFIG.device.peak_performance
        self.max_bandwidth = GLOBAL_CONFIG.device.dram_bandwidth
        
    def __str__(self):
        return f"{self.name}"
    
    def calc_result(self):
        infer_time = 0
        ops = 0 
        mem = 0
        if self.result["children"]:
            for child in self.result["children"]:
                if isinstance(child, list):
                    for c in child:
                        c.calc_result()
                        ops += c.result["ops"]
                        mem += c.result["mem"]
                        infer_time += c.result["inference_time"]
                else:
                    child.calc_result()
                    ops += child.result["ops"]
                    mem += child.result["mem"]
                    infer_time += child.result["inference_time"]
            self.result["ops"] = ops
            self.result["mem"] = mem
            self.result["inference_time"] = infer_time
        
                    
    def calc_cycle(self):
        for layer in self.layers:
            c_cyle, m_cycle = layer.calc_cycle()
            self.compute_cycle += c_cyle
            self.memory_cycle += m_cycle
        return self.compute_cycle, self.memory_cycle

    def calc_num_ops(self, x):
        return 0

    def forward(self, x):
        return
    
    def compute_performance(self, ops, mem, max_perf=None, max_bandwidth=None):
        if max_perf is None:
            max_perf = self.max_perf
        if max_bandwidth is None:
            max_bandwidth = self.max_bandwidth
        
        arith_intensity = ops / mem  # ops/byte
        point = max_perf / max_bandwidth  # cross point
        
        if arith_intensity < point:
            bound = "memory"
            performance = arith_intensity * max_bandwidth
        else:
            bound = "compute"
            performance = max_perf
        time = ops/performance
        
        self.result["inference_time"] = time
        self.result["ops"] = ops
        self.result["mem"] = mem
        self.result["arith_intensity"] = arith_intensity
        self.result["bound"] = bound
        self.result["performance"] = performance
        self.result["mem_bandwidth"] = max_bandwidth
        return time, arith_intensity, performance, bound

    def __call__(self, *args, **kwds):
        return self.forward(*args, **kwds)
    

class KVCache:
    def __init__(self, max_batch_size, n_layers, max_seq_len, n_kv_heads, head_dim):
        self.cache_k = torch.empty(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim, dtype=GLOBAL_CONFIG.weight_dtype)
        self.cache_v = torch.empty(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim, dtype=GLOBAL_CONFIG.weight_dtype)
        
    def update(self, batch_size, n_layer, start_pos, xk, xv):
        self.cache_k[n_layer, :batch_size, start_pos :start_pos + xk.size(2)] = xk.transpose(1,2)
        self.cache_v[n_layer, :batch_size, start_pos :start_pos + xv.size(2)] = xv.transpose(1,2)

    def get(self, batch_size, n_layer, start_pos, seq_len):
        keys = self.cache_k[n_layer, :batch_size, :start_pos + seq_len]
        values = self.cache_v[n_layer, :batch_size, :start_pos + seq_len]
        return keys, values

class SoftMax(Model):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.max_perf = GLOBAL_CONFIG.device.peak_performance_cuda_core
        
    def forward(self, x: torch.Tensor):
        output = x
        if x.dim() == 2:
            x = x[None,None,:,:]
        elif x.dim() == 3:
            x = x[None,:,:,:]
            
        B, H, QS, KTS = x.shape
        ops = B * H * QS * KTS * 5
        
        _mem_load = B * H * QS * KTS * x.element_size()
        _mem_store = B * H * QS * KTS * x.element_size()
        mem = _mem_load + _mem_store
        if _mem_load <= GLOBAL_CONFIG.device.l2_size and _mem_store <= GLOBAL_CONFIG.device.l2_size:
            self.compute_performance(ops, mem, max_bandwidth=GLOBAL_CONFIG.device.l2_brandwith)
        else:
            self.compute_performance(ops, mem)
        return output, ops, mem


class LayerNormalization(Model):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.max_perf = GLOBAL_CONFIG.device.peak_performance_cuda_core
    
    def forward(self, x: torch.Tensor):
        output = x
        if x.dim() == 2:
            x = x[None,None,:,:]
        elif x.dim() == 3:
            x = x[None,:,:,:]
            
        ops = x.numel() * 7
        
        _mem_load = x.numel() * x.element_size()
        _mem_store = x.numel() * x.element_size()
        mem = _mem_load + _mem_store
        if _mem_load <= GLOBAL_CONFIG.device.l2_size and _mem_store <= GLOBAL_CONFIG.device.l2_size:
            self.compute_performance(ops, mem, max_bandwidth=GLOBAL_CONFIG.device.l2_brandwith)
        else:
            self.compute_performance(ops, mem)
        
        return output, ops, mem

class ResidualAddition(Model):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.max_perf = GLOBAL_CONFIG.device.peak_performance_cuda_core
        
    def forward(self, x:torch.Tensor, y:torch.Tensor):
        assert(x.size() == y.size())
        output = x + y
        ops = x.numel()
        _mem_load_or_store = x.numel() * x.element_size()
        mem = _mem_load_or_store*3
        if _mem_load_or_store <= GLOBAL_CONFIG.device.l2_size:
            self.compute_performance(ops, mem, max_bandwidth=GLOBAL_CONFIG.device.l2_brandwith)
        else:
            self.compute_performance(ops, mem)
        return output, ops, mem
            
class SiLU(Model):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        self.max_perf = GLOBAL_CONFIG.device.peak_performance_cuda_core
        
    def forward(self, x:torch.Tensor):
        output = x
        if x.dim() == 2:
            x = x[None,None,:,:]
        elif x.dim() == 3:
            x = x[None,:,:,:]
            
        ops = x.numel() * 5  # silu(x) = x*sigmoid(x)
        
        _mem_load = x.numel() * x.element_size()
        _mem_store = x.numel() * x.element_size()
        mem = _mem_load + _mem_store
        
        if _mem_load <= GLOBAL_CONFIG.device.l2_size and _mem_store <= GLOBAL_CONFIG.device.l2_size:
            self.compute_performance(ops, mem, max_bandwidth=GLOBAL_CONFIG.device.l2_brandwith)
        else:
            self.compute_performance(ops, mem)
            
        return output, ops, mem
    
    
class Decoder(Model):
    def __init__(self, layer_idx, name=None) -> None:
        super().__init__(name)
        self.attention_norm = LayerNormalization(name=f"attention_norm")
        self.gqa = GroupQueryAttention(hidden_size=GLOBAL_CONFIG.hidden_size, 
                                       num_attention_heads=GLOBAL_CONFIG.num_attention_heads,
                                       num_key_value_heads=GLOBAL_CONFIG.num_key_value_heads, 
                                       head_dim=GLOBAL_CONFIG.head_dim,
                                       layer_idx=layer_idx, name=f"gqa")
        self.attention_residual = ResidualAddition(name=f"attention_residual")

        self.mlp_norm = LayerNormalization(name=f"mlp_norm")
        self.mlp = MLP(GLOBAL_CONFIG.hidden_size, GLOBAL_CONFIG.im_size, layer_idx, name=f"mlp")
        self.mlp_residual = ResidualAddition(name=f"mlp_residual")
        
        self.result["children"].append(self.attention_norm)
        self.result["children"].append(self.gqa)
        self.result["children"].append(self.attention_residual)
        self.result["children"].append(self.mlp_norm)
        self.result["children"].append(self.mlp)
        self.result["children"].append(self.mlp_residual)
        
    def forward(self, x, kv_cache, cache_start_pos):
        _ops = 0
        _mem = 0
        residual = x
        y, ops, mem = self.attention_norm(x)
        _ops += ops
        y, ops, mem = self.gqa(y, kv_cache, cache_start_pos)
        _ops += ops
        y, ops, mem = self.attention_residual(residual, y)
        _ops += ops
        residual = y
        y, ops, mem = self.mlp_norm(y)
        _ops += ops
        y, ops, mem = self.mlp(y)
        _ops += ops
        y, ops, mem = self.mlp_residual(residual, y)
        _ops += ops
        
        return y, _ops, _mem

class MLP(Model):
    def __init__(self, hidden_size, im_size, layer_idx, name=None) -> None:
        super().__init__(name)
        self.hidden_size = hidden_size
        self.im_size = im_size
        self.gate_proj = GEMM(self.hidden_size, self.im_size, name=f"{name}.gate")
        self.up_proj = GEMM(self.hidden_size, self.im_size, name=f"{name}.up")
        self.down_proj = GEMM(self.im_size, self.hidden_size, name=f"{name}.down")
        self.act_fn = SiLU()
        
        self.result["children"].append(self.gate_proj)
        self.result["children"].append(self.up_proj)
        self.result["children"].append(self.act_fn)
        self.result["children"].append(self.down_proj)
        
    def forward(self, x):
        _ops = 0
        _mem = 0
        y_gate, ops, mem = self.gate_proj(x)
        _ops += ops
        _mem += mem
        y_up, ops, mem = self.up_proj(x)
        _ops += ops
        _mem += mem
        act_y_gate, ops, mem = self.act_fn(y_gate)
        _ops += ops
        _mem += mem
        y_down, ops, mem = self.down_proj(act_y_gate * y_up)
        _ops += ops
        _mem += mem
        return y_down, _ops, _mem
    
class ATTENTION_GEMM(Model):
    def __init__(self, name=None) -> None:
        super().__init__(name)
        
    def forward(self, input: torch.Tensor, weight: torch.Tensor):    
        in_dim = input.dim()
        
        if input.dim() == 2:
            input = input[None,None,:,:]
        elif input.dim() == 3:
            input = input[None,:,:,:]
            
        if weight.dim() == 2:
            weight = weight[None,None,:,:]
        elif weight.dim() == 3:
            weight = weight[None,:,:,:]
            
        S, B, M, K = input.shape
        _S, _B, _K, N = weight.shape

        if in_dim == 2:
            output = torch.empty([M, N], dtype=GLOBAL_CONFIG.act_dtype)
        elif in_dim == 3:
            output = torch.empty([B, M, N], dtype=GLOBAL_CONFIG.act_dtype)
        else:
            output = torch.empty([S, B, M, N], dtype=GLOBAL_CONFIG.act_dtype)
            
        _mem_load_weight = K*N*weight.element_size()
        _mem_load_activation = B*M*K*input.element_size()
        _mem_store_output = B*M*N*input.element_size()

        ops = 2*M*N*K*B  # mac
        mem = _mem_load_weight + _mem_load_activation + _mem_store_output
        
        if _mem_load_weight <= GLOBAL_CONFIG.device.l2_size and _mem_load_activation <= GLOBAL_CONFIG.device.l2_size and _mem_store_output <= GLOBAL_CONFIG.device.l2_size:
            self.compute_performance(ops, mem, max_bandwidth=GLOBAL_CONFIG.device.l2_brandwith)
        else:
            self.compute_performance(ops, mem)
            
        return output, ops, mem

class GEMM(Model):
    def __init__(self, in_features, out_features, name=None) -> None:
        super().__init__(name)
        self.weight = torch.empty(in_features, out_features, dtype=GLOBAL_CONFIG.weight_dtype)
        
    def forward(self, input: torch.Tensor):    
        _dim = input.dim()
        if input.dim() == 2:
            input = input[None,None,:,:]
        elif input.dim() == 3:
            input = input[None,:,:,:]
        S, B, M, K = input.shape
        N = self.weight.shape[1]
        if _dim == 2:
            output = torch.empty([M, N], dtype=GLOBAL_CONFIG.act_dtype)
        elif _dim == 3:
            output = torch.empty([B, M, N], dtype=GLOBAL_CONFIG.act_dtype)
        else:
            output = torch.empty([S, B, M, N], dtype=GLOBAL_CONFIG.act_dtype)
            
        _mem_load_weight = K*N*self.weight.element_size()
        _mem_load_activation = B*M*K*input.element_size()
        _mem_store_output = B*M*N*input.element_size()

        ops = 2*M*N*K*B  # mac
        mem = _mem_load_weight + _mem_load_activation + _mem_store_output
        if _mem_load_weight <= GLOBAL_CONFIG.device.l2_size and _mem_load_activation <= GLOBAL_CONFIG.device.l2_size and _mem_store_output <= GLOBAL_CONFIG.device.l2_size:
            self.compute_performance(ops, mem, max_bandwidth=GLOBAL_CONFIG.device.l2_brandwith)
        else:
            self.compute_performance(ops, mem)
        return output, ops, mem
    
    def calc_cycle(self):
        pass

class GroupQueryAttention(Model):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim, layer_idx, name=None):
        super().__init__(name)
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = GEMM(hidden_size, head_dim * num_attention_heads, name=f"{name}.q_proj")
        self.k_proj = GEMM(hidden_size, head_dim * num_key_value_heads, name=f"{name}.k_proj")
        self.v_proj = GEMM(hidden_size, head_dim * num_key_value_heads, name=f"{name}.v_proj")
        self.o_proj = GEMM(hidden_size, head_dim * num_attention_heads, name=f"{name}.o_proj")
        
        self.qkt = ATTENTION_GEMM(name=f"{name}.qkt_matmul")
        self.qktv = ATTENTION_GEMM(name=f"{name}.qkt_v_matmul")
        self.softmax = SoftMax(name=f"{name}.softmax")
        
        self.result["children"].append(self.q_proj)
        self.result["children"].append(self.k_proj)
        self.result["children"].append(self.v_proj)
        self.result["children"].append(self.qkt)
        self.result["children"].append(self.qktv)
        self.result["children"].append(self.softmax)
        self.result["children"].append(self.o_proj)
    
    def forward(self, x, kv_cache: KVCache, cache_start_pos):
        _ops = 0
        _mem = 0

        q_proj, ops, mem = self.q_proj(x)
        _ops += ops
        _mem += mem
        
        k_proj, ops, mem= self.k_proj(x)
        _ops += ops
        _mem += mem
        
        v_proj, ops, mem = self.v_proj(x)
        _ops += ops
        _mem += mem
        
        bsz, q_len, _ = x.size()
        q_proj = q_proj.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_proj = k_proj.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v_proj = v_proj.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        kv_cache.update(bsz, self.layer_idx, cache_start_pos, k_proj, v_proj)
        if cache_start_pos:
            k_proj, v_proj =  kv_cache.get(bsz, self.layer_idx, cache_start_pos, q_len)
            k_proj = k_proj.transpose(1,2)
            v_proj = v_proj.transpose(1,2)
        _len = q_len + cache_start_pos
        k_proj = k_proj[:,:,None,:,:].expand(bsz, self.num_key_value_heads, self.num_key_value_groups, _len, self.head_dim).reshape(bsz, self.num_heads, _len, self.head_dim)
        v_proj = v_proj[:,:,None,:,:].expand(bsz, self.num_key_value_heads, self.num_key_value_groups, _len, self.head_dim).reshape(bsz, self.num_heads, _len, self.head_dim)
        
        qkt, ops, mem = self.qkt(q_proj, k_proj.transpose(2,3))
        _ops += ops
        _mem += mem

        qkt, ops, mem = self.softmax(qkt)
        
        qktv, ops, mem = self.qktv(qkt, v_proj)
        _ops += ops
        _mem += mem
        
        qktv = qktv.reshape(bsz, q_len, self.head_dim* self.num_heads)
        
        o_proj, ops, mem = self.o_proj(qktv)
        _ops += ops
        _mem += mem
        
        return o_proj, _ops, _mem
    
    def calc_cycle(self):
        pass
    
    
class Llama3_8B(Model):
    def __init__(self) -> None:
        super().__init__()
        config = GLOBAL_CONFIG

        self.kv_cache = KVCache(config.batch_size, config.num_hidden_layers, config.hidden_size, config.num_key_value_heads, config.head_dim)
        self.cache_start_pos = 0
        self.config = config
        self.layers = []
        self.output_len = config.output_seq_len

        self.head = GEMM(config.vocab_size, config.hidden_size, name="head")
        self.tail = GEMM(config.hidden_size, config.vocab_size, name="tail")
        self.norm = LayerNormalization()
        n_layers = config.num_hidden_layers
        for i in range(n_layers):
            self.layers.append(Decoder(i, name=f"decoder.layer.{i}"))
            
        self.result["children"].append(self.head)
        self.result["children"].append(self.layers)
        self.result["children"].append(self.norm)
        self.result["children"].append(self.tail)
    
    def forward(self, x):
        y = x
        out_len = 0
        total_ops = 0
        result = []
        prefill = True
        TTFT = 0
        TPOT = []
        TPS = 0
        while out_len < self.output_len:
            bsz, q_len = y.size()
            y = y[:,:,None].expand(bsz, q_len, self.config.vocab_size)
            y, ops, mem = self.head(y)
            for layer in self.layers:
                y, ops, mem = layer(y, self.kv_cache, self.cache_start_pos)
            if prefill:
                self.cache_start_pos += 1
                out_len += 1
            else:
                self.cache_start_pos += 256
                out_len += 256
                
            y, ops, mem = self.norm(y)  # Final layer norm
            
            y, ops, mem = self.tail(y)
            y = y[:,-1,:1]  # token output
            result.append(y)
            self.calc_result()
            if prefill:
                TTFT = self.result["inference_time"]
                prefill = False
            else:
                TPOT.append(self.result["inference_time"])
        print(f"TTFT: {TTFT*1e+3} ms")
        print(f"TPOT: {np.mean(TPOT)*1e+3} ms")
        print(TPOT)
        Latency = TTFT+np.mean(TPOT)*self.output_len
        print(f"Latency: {Latency} s")
        Throughput = self.output_len/Latency
        print(f"Throughput: {Throughput} token/s")
        return result, total_ops
    
    def calc_prefill(self):
        pass
        