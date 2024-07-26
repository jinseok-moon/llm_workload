
B = 8
M = 2048
N = 4096
K = 4096

_M = 16
_N = 8
_K = 16

def roofline_analyze(bandwidth, max_OPS, OPs, memory_access):
    # bandwidth is bytes/s
    # memory_access in byte
    # x axis is OPS/byte
    # y axis is OPS/s
    y_max = max_OPS
    memory_access_bytes = memory_access
    turning_point = y_max / bandwidth
    arithmetic_intensity = OPs / memory_access_bytes
    if arithmetic_intensity < turning_point:
        bound = "memory"
        performance = arithmetic_intensity * bandwidth
    else:
        bound = "compute"
        performance = y_max
    if performance==0:
        1==1
        pass
    return arithmetic_intensity, performance, bound


tc = (B*M*N*K) / (_M*_N*_K)  # hardware instruction count
_tc_total = tc/4/108  # TOTAL Instruction at parallel
compute_clock = 1.410e+9

memory_bandwidth = 1.935e+12  # 1.94 TB/s
max_OPS = 312e+12   # 312 TFLOPS
ops = (2*B*M*N*K)
memory_access = B*(M*N + N*K + M*N)

ai, perf, bound = roofline_analyze(memory_bandwidth, max_OPS, ops, memory_access)
__time = ops / perf
print(__time)
_time = _tc_total*8 / compute_clock
print(f"Time :{_time} s")
