import matplotlib.pyplot as plt
import numpy as np


def visualize(c_band, m_band):
    x = np.arange(0, 1000, 20)
    
    y = np.array([min(m_band*_x, c_band) for _x in x])
    
    plt.plot(x, y)
    plt.show()


dram_bandwidth = 1.935e+12  # 1.94 TB/s
SM = 108
tensor_core = 4
cuda_core = 64
compute_clock = 1.410e+9
memory_clock = 1.512e+9
compute_bandwidth = 312e+12 # 312 TFLOPS

if __name__ == "__main__":
    visualize(compute_bandwidth, dram_bandwidth)