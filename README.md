# LLM WORKLOAD

## Condition
- Llama3-8B Model (based on Huggingface)

- Input token 2048
- Output token 2048
- Batch size 8
- fp16 weight/activation
- NVIDIA A100 80GB PCIe
- 3rd gen. tensor core
- base clock : 1065 MHz
- boost clock : 1410 MHz
- 7 GPCs, 7 or 8 TPCs/GPC, 2 SMs/TPC, up to 16 SMs/GPC, 108 SMs
- 64 FP32 CUDA Cores/SM, 6912 FP32 CUDA Cores per GPU
- 4 Third-generation Tensor Cores/SM, 432 Third-generation Tensor Cores per GPU
- 5 HBM2 stacks, 10 512-bit Memory Controllers
- GEMM tensor core operation 만 계산
  - Load/Store 는 async copy 를 통해 computation 에 hiding 됨
- layer norm, softmax, activation 은 CUDA core 활용

## About Model Layer
- Decoder Layer
![Model Decoder layer](llama_decoder.png)

- KV Cache with Group Query Attention
![kvcache](kvcache_and_gqa.png)



## Environment
You need to install `PyTorch` because of `torch.Tensor` usages.
```
conda install pytorch::pytorch torchvision torchaudio -c pytorch
or
pip3 install torch torchvision torchaudio
```

```
$ python estimator.py

# visualize.py would make the roofline analysis image, but it's not easily readable..
$ python visualize.py
```

## Estimation
```bash
$ python estimator.py

---Flash Attention---
TTFT: 956.6947879034522 ms
TPOT: 10.290764998449612 ms
Latency: 22.032181504728257 s
Throughput: 92.95493501451435 token/s
```

```bash
# peak performance 614 TFLOPS
$ python estimator.py

---Flash Attention---
TTFT: 541.8848160945525 ms
TPOT: 10.290764998449612 ms
Latency: 21.617371532919357 s
Throughput: 94.73862244913843 token/s
```

## For more accurate estimation
- DRAM to L2 or L1 bandwidth
- compute/memory cycle calculation
- async copy with tiling
- L2 persistent cache


## Reference
- [Future Scaling of Memory Hierarchy for Tensor Cores and Eliminating Redundant Shared Memory Traffic Using Inter-Warp Multicasting](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9893362)
- [Flash Attention v2](https://arxiv.org/abs/2307.08691)
- [NVIDIA A100 Tensor Core GPU Architecture](https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf)
- [DEVELOPING CUDA KERNELS TO PUSH TENSOR CORES TO THE ABSOLUTE LIMIT ON NVIDIA A100](https://developer.download.nvidia.com/video/gputechconf/gtc/2020/presentations/s21745-developing-cuda-kernels-to-push-tensor-cores-to-the-absolute-limit-on-nvidia-a100.pdf)
