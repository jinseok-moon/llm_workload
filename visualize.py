import matplotlib.pyplot as plt
import numpy as np
from config import *
import json


def visualize_ops(data, plt_list):
    if data["children"]:
        for k, v in data["children"].items():
            if v["children"]:
                plt_list = visualize_ops(v, plt_list)
            else:
                plt_list.append(v)
    return plt_list


with open("decode.json", "r") as data: 
    _data = json.load(data)
    decode_plt_list = visualize_ops(_data, [])
    
    
with open("prefill.json", "r") as data: 
    _data = json.load(data)
    prefill_plt_list = visualize_ops(_data, [])
    
figure = plt.Figure(figsize=(12.8, 12.8), dpi=200)

_x_list_pr = []
_y_list_pr = []
_x_list_dc = []
_y_list_dc = []

for op in prefill_plt_list:
    if op["max_perf"] == GLOBAL_CONFIG.device.peak_performance:
        _x_list_pr.append(op["arith_intensity"])
        _y_list_pr.append(op["performance"])
        
for op in decode_plt_list:
    if op["max_perf"] == GLOBAL_CONFIG.device.peak_performance:
        _x_list_dc.append(op["arith_intensity"])
        _y_list_dc.append(op["performance"])
        
x = np.linspace(0, 1000)
y = np.array([min(GLOBAL_CONFIG.device.dram_bandwidth*_x, GLOBAL_CONFIG.device.peak_performance) for _x in x])

plt.plot(x, y)

plt.plot(_x_list_pr, _y_list_pr, 'ro', label="Prefill")
plt.plot(_x_list_dc, _y_list_dc, 'bo', label="Decode")
plt.xlabel("Arithmetic Intensity")
plt.ylabel("Performance (TFLOPS)")
plt.legend()
plt.suptitle("Roofline Analysis")
plt.savefig("performance.png")