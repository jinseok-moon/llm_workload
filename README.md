# LLM WORKLOAD

## Condition
- Input token 2048
- Output token 2048
- Batch size 8
- fp16 weight/activation
- Llama3-8B Model 
- NVIDIA A100 80GB PCIe

## Hardware spec
|                      |            |     |     |     |
| -------------------- | ---------- | --- | --- | --- |
| FP16 Tensor Core     | 312 TFLOPS |     |     |     |
| BF16 Tensor Core     | 312 TFLOPS |     |     |     |
| GPU Memory Bandwidth | 1,935 GB/s |     |     |     |
| Memory Interface     | 5120-bit   |     |     |     |

1024 FMA / clock

base clock : 1065 MHz
boost clock : 1410 MHz

- 7 GPCs, 7 or 8 TPCs/GPC, 2 SMs/TPC, up to 16 SMs/GPC, 108 SMs
- 64 FP32 CUDA Cores/SM, 6912 FP32 CUDA Cores per GPU
- 4 Third-generation Tensor Cores/SM, 432 Third-generation Tensor Cores per GPU
- 5 HBM2 stacks, 10 512-bit Memory Controllers

- GEMM tensor core operation 만 계산
  - Load/Store 는 async copy 를 통해 computation 에 hiding 됨
- layer norm, softmax, activation 은 CUDA core 활용

## Reference
- https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9893362


## TODO
- flash attention 구현하기