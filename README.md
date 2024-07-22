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

Each SM in the A100 GPU includes four of the new redesigned Tensor Cores and
therefore each SM in A100 delivers 1024 FP16 FMA operations per clock (or 2048 individual
FP16 floating point operations per clock).
Comparing total GPU performance, not just SM-level performance, the NVIDIA A100 Tensor
Core GPU with its 108 SMs includes a total of 432 Tensor Cores that deliver up to 312 TFLOPS
of dense mixed-precision FP16/FP32 performance. That equates to 2.5x the mixed-precision
Tensor Core performance of the entire Tesla V100 GPU, and 20x V100’s standard FP32 (FMA
operations running on traditional FP32 CUDA cores) throughput.
