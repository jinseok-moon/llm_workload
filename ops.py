import torch
from kernels import *

class Model:
    def __init__(self) -> None:
        self.compute_cycle = 0
        self.memory_cycle = 0
        
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
    
    def __call__(self, *args: torch.Any, **kwds: torch.Any) -> torch.Any:
        return self.forward(*args, **kwds)
    

class KVCache:
    def __init__(self, max_batch_size, n_layers, max_seq_len, n_kv_heads, head_dim):
        self.cache_k = torch.empty(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        self.cache_v = torch.empty(n_layers, max_batch_size, max_seq_len, n_kv_heads, head_dim)
        
    def update(self, batch_size, n_layer, start_pos, xk, xv):
        self.cache_k[n_layer, :batch_size, start_pos :start_pos + xk.size(2)] = xk.transpose(1,2)
        self.cache_v[n_layer, :batch_size, start_pos :start_pos + xv.size(2)] = xv.transpose(1,2)

    def get(self, batch_size, n_layer, start_pos, seq_len):
        keys = self.cache_k[n_layer, :batch_size, :start_pos + seq_len]
        values = self.cache_v[n_layer, :batch_size, :start_pos + seq_len]
        return keys, values
    

class LayerNormalization(Model):
    def __init__(self) -> None:
        super().__init__()

    def calc_num_ops(self, x):
        return 0
    #     num, C, M, K = self.input.shape
    #     N = self.weight.shape[3]
    #     B = C
    #     self.num_ops = 2*M*N*K*B  # mac
    #     return
    
    def forward(self, x):
        output = x
        num_ops = self.calc_num_ops(x)
        return output, num_ops, 0

class SwiGLU(Model):
    def __init__(self) -> None:
        super().__init__()

    def calc_num_ops(self, x):
        return 0
    #     num, C, M, K = self.input.shape
    #     N = self.weight.shape[3]
    #     B = C
    #     self.num_ops = 2*M*N*K*B  # mac
    #     return
    
    def forward(self, x):
        output = x
        num_ops = self.calc_num_ops(x)
        return output, num_ops, 0
    
    
    
class Decoder(Model):
    def __init__(self, config, layer_idx) -> None:
        super().__init__()
        self.config = config
        self.layernorm = LayerNormalization()
        self.gqa = GroupQueryAttention(hidden_size=self.config["hidden_size"], 
                                       num_attention_heads=self.config["num_attention_heads"],
                                       num_key_value_heads=self.config["num_key_value_heads"], 
                                       head_dim=self.config["head_dim"],
                                       layer_idx=layer_idx)
        self.mlp = MLP(self.config["hidden_size"], self.config["intermediate_size"])
        self.residual = ResidualAddition()
        
    def forward(self, x, kv_cache, cache_start_pos):
        _ops = 0
        _mem = 0
        residual = x
        y, ops, mem = self.layernorm(x)
        _ops += ops
        y, ops, mem = self.gqa(y, kv_cache, cache_start_pos)
        _ops += ops
        y, ops, mem = self.residual(residual, y)
        _ops += ops
        residual = y
        y, ops, mem = self.layernorm(y)
        _ops += ops
        y, ops, mem = self.mlp(y)
        _ops += ops
        y, ops, mem = self.residual(residual, y)
        _ops += ops
        return y, _ops, _mem

class MLP(Model):
    def __init__(self, hidden_size, im_size) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.im_size = im_size
        self.gate_proj = GEMM(torch.empty(self.hidden_size, self.im_size))
        self.up_proj = GEMM(torch.empty(self.hidden_size, self.im_size))
        self.down_proj = GEMM(torch.empty(self.im_size, self.hidden_size))
        self.act_fn = SwiGLU()
    
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
        return y_down.squeeze(0), _ops, _mem
    
    
class GEMM(Model):
    def __init__(self, t_weight) -> None:
        super().__init__()
        if t_weight.dim() == 2:
            t_weight = t_weight[None,None,:,:]
        elif t_weight.dim() == 3:
            t_weight = t_weight[None,:,:,:]
        
        self.weight: torch.Tensor = t_weight
    
    def forward(self, input: torch.Tensor):    
        if input.dim() == 2:
            input = input[None,None,:,:]
        elif input.dim() == 3:
            input = input[None,:,:,:]
        
        S, B, M, K = input.shape
        N = self.weight.shape[3]
        output = torch.empty([S, B, M, N])
        
        _ops = 2*M*N*K*B  # mac
        _mem = B*(M*N + N*K + K*N)
        return output, _ops, _mem
    
    def calc_cycle(self):
        pass

class ResidualAddition(Model):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        return x+y, 0, 0
            
class GroupQueryAttention(Model):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads, head_dim, layer_idx):
        super().__init__()
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
        self.num_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

        self.q_proj = GEMM(torch.empty(hidden_size, head_dim * num_attention_heads))
        self.k_proj = GEMM(torch.empty(hidden_size, head_dim * num_key_value_heads))
        self.v_proj = GEMM(torch.empty(hidden_size, head_dim * num_key_value_heads))
        self.o_proj = GEMM(torch.empty(hidden_size, head_dim * num_attention_heads))
        
        self.qkt = GEMM(torch.empty(head_dim, num_attention_heads))
        self.qktv = GEMM(torch.empty(num_attention_heads, head_dim))
        
    def calc_num_ops(self, x):
        _kv_repeat_size = self.hidden_size // self.num_key_value_groups
        return 0
    
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
        
        qkt, ops, mem = self.qkt(q_proj.transpose(1,2))
        _ops += ops
        _mem += mem
        qktv, ops, mem = self.qktv(qkt)
        _ops += ops
        _mem += mem
        
        qktv = qktv.reshape(bsz, q_len, self.head_dim* self.num_heads)
        o_proj, ops, mem = self.o_proj(qktv)
        _ops += ops
        _mem += mem
        return o_proj.squeeze(0), _ops, _mem
    
    def calc_cycle(self):
        pass
    
    
class Llama3_8B(Model):
    """
    Input Tensor: (batch, Input token) == (8, 2048)
    Embedded Token : (B, I, Hidden state) -> (8, 2048, 4096)
    Attention-
        Q Projection: (8, 2048, 4096) * (4096, 4096) -> (8, 2048, 4096)
        V Projection: (8, 2048, 4096) * (4096, 1024) -> (8, 2048, 1024)
        K Projection: (8, 2048, 4096) * (4096, 1024) -> (8, 2048, 1024)
        ROPE 
    """
    def __init__(self, config, output_len) -> None:
        super().__init__()
        self.kv_cache = KVCache(config["batch_size"], config["num_hidden_layers"], config["hidden_size"], config["num_key_value_heads"], config["head_dim"])
        self.cache_start_pos = 0
        self.config = config
        self.layers = []
        self.output_len = output_len
        self.tail = GEMM(torch.empty(config["hidden_size"], config["vocab_size"]))
        n_layers = config["num_hidden_layers"]
        for i in range(n_layers):
            self.layers.append(Decoder(config, i))

    def forward(self, x):
        y = x
        out_len = 0
        total_ops = 0
        result = []
        while out_len < self.output_len:
            _token_ops = 0
            bsz, q_len = y.size()
            y = y[:,:,None].expand(bsz, q_len, self.config["hidden_size"])  # Head
            for layer in self.layers:
                y, ops, mem = layer(y, self.kv_cache, self.cache_start_pos)
                _token_ops += ops
            self.cache_start_pos += q_len
            y, ops, mem = self.tail(y)
            _token_ops += ops
            y = y.squeeze(0)[:,-1,:1]  # token output
            result.append(y)
            out_len += 1
            print("TFLOPs", _token_ops / 1e+12)
            total_ops += _token_ops
        return result, total_ops
    
    def calc_prefill(self):
        pass
        