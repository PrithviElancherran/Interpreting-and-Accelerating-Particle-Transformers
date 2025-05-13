# Interpreting and Accelerating Particle Transformers

## Overview

This research project builds upon the **Particle Transformer (ParT)** architecture introduced in the paper ["Particle Transformer for Jet Tagging"](https://arxiv.org/abs/2202.03772). The original ParT serves as a state-of-the-art backbone for jet tagging tasks in high-energy particle physics, leveraging a Transformer-based design enhanced with pairwise particle interaction features injected as attention biases before the softmax operation. ParT significantly outperforms previous approaches like ParticleNet on standard jet tagging benchmarks.

**This project extends the original work by focusing on two key research directions:**

1. **Interpretability**: Analyzing and visualizing attention matrices to understand how the model learns particle interactions and dependencies.

2. **Acceleration**: Implementing and benchmarking multiple optimization techniques to make ParT faster and more efficient for real-world deployment, including:
   - Efficient attention mechanisms (Linformer, Reformer, Mamba)
   - Int-8 quantization for reduced model size and faster inference
   - Flash Attention for memory-efficient training
   - Comprehensive model profiling to identify bottlenecks

![arch](figures/arch.png)

---

## Research Contributions

### Attention Matrix Analysis

Understanding how a Transformer model attends to different inputs is crucial for interpretability. The attention matrix reveals which particles the model considers most relevant when making predictions. In well-trained models, attention patterns tend to be sparse and focused rather than uniformly distributed, reflecting meaningful physical relationships between particles. By visualizing these patterns, we gain insights into the model's decision-making process and can identify opportunities for computational optimization by focusing only on significant interactions.

![image](https://github.com/user-attachments/assets/17ec9a88-755f-44ad-92d6-50cf5c5e44e2)

### Linformer ParT (LinParT)

Standard self-attention scales quadratically with sequence length O(n²), which becomes computationally expensive for inputs with many particles. We integrate Linformer, which applies low-rank projections to keys and values, reducing the attention complexity to O(n). This modification enables the Particle Transformer to handle larger inputs with significantly lower memory consumption and compute requirements—essential for real-time inference and deployment on resource-constrained hardware.

![image](https://github.com/user-attachments/assets/eaa4d26c-3a36-4973-9e37-77245f70da4c)

### Int-8 Quantization

Quantization compresses model weights from 32-bit floating point to 8-bit integers, reducing model size by approximately 4x while accelerating inference. This technique is particularly valuable for deploying Transformer models on edge devices, FPGAs, or in latency-sensitive applications. Using calibration-based quantization, we demonstrate that the Particle Transformer retains most of its classification accuracy while achieving substantial speedups.

![image](https://github.com/user-attachments/assets/3b156033-42f6-4a4d-a7e9-c08f6028419d)

### Flash Attention

Flash Attention optimizes the standard attention computation by fusing multiple operations (matrix multiplication, softmax, dropout) into a single GPU kernel, minimizing memory transfers between GPU global memory and on-chip SRAM. This allows for significantly larger batch sizes and longer sequences without running into memory limitations, improving both training throughput and inference efficiency.

<img width="621" alt="image" src="https://github.com/user-attachments/assets/348801e0-f489-4246-b4d4-01faaec04596" />

---

## Model Profiling and Results

We conduct detailed profiling to identify performance bottlenecks across different layers of the Particle Transformer. Profiling helps determine whether the model is memory-bound (limited by data transfer speeds) or compute-bound (limited by arithmetic throughput), guiding targeted optimizations such as kernel fusion, quantization, or memory layout adjustments.

Base:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/f14e7d8c-6b8a-476f-bad7-f0d5772e5b99" />


Quantized:

<img width="800" alt="image" src="https://github.com/user-attachments/assets/db28fe5a-6a95-4f87-b99d-a1778df99754" />


Base:

<img src="https://github.com/user-attachments/assets/972bae55-d1fb-485a-afe5-1b8fa5c43451" width="800"/>


Flash Attention:

<img src="https://github.com/user-attachments/assets/883bde3a-49e7-4f6f-91f8-88a7af383fe2" width="800"/>


Base:

<img src="https://github.com/user-attachments/assets/0e86e45e-d6e0-41ff-9317-2ac03bdbf1a0" width="800"/>


Optimized:

<img src="https://github.com/user-attachments/assets/eb0c83b4-97ec-425a-92c8-3d92a21db2a7" width="800"/>


Base:

<img src="https://github.com/user-attachments/assets/8c4b3c90-8c6e-442c-b76e-7c498dc72fa9" width="800"/>


Optimized:

<img src="https://github.com/user-attachments/assets/6119b1eb-7669-47c7-9269-5ac1cf0c123d" width="800"/>


Base:

<img src="https://github.com/user-attachments/assets/18c84a3d-8347-449c-b824-dc89b57a4115" width="800"/>


Optimized:

<img src="https://github.com/user-attachments/assets/707e7c7c-e0b2-45c5-9781-5bff2fc80d3c" width="800"/>

---

## Getting Started

### Prerequisites

Install the [weaver](https://github.com/hqucms/weaver-core) framework for dataset handling and training:

```bash
pip install 'weaver-core>=0.4'
```

### Download Datasets

Download the JetClass, QuarkGluon, or TopLandscape datasets:

```bash
./get_datasets.py [JetClass|QuarkGluon|TopLandscape] [-d DATA_DIR]
```

Dataset paths are automatically updated in `env.sh` after download.

### Training

**JetClass Dataset:**

```bash
./train_JetClass.sh [MODEL] [FEATURES] [OPTIONS]
```

**Available Models:**
| Model | Description | Paper |
|-------|-------------|-------|
| `ParT` | Particle Transformer | [arXiv:2202.03772](https://arxiv.org/abs/2202.03772) |
| `LinformerParT` | ParT with Linformer attention | This work |
| `ReformerParT` | ParT with Reformer attention | This work |
| `MambaParT` | ParT with Mamba state-space model | This work |
| `PN` | ParticleNet | [arXiv:1902.08570](https://arxiv.org/abs/1902.08570) |
| `PFN` | Particle Flow Network | [arXiv:1810.05165](https://arxiv.org/abs/1810.05165) |
| `PCNN` | P-CNN | [arXiv:1902.09914](https://arxiv.org/abs/1902.09914) |

**Feature Sets:**
| Option | Description |
|--------|-------------|
| `kin` | Kinematic features only (pt, eta, phi, energy) |
| `kinpid` | Kinematic + particle identification |
| `full` | All features including trajectory displacement (default) |

**Multi-GPU Training:**

Using DataParallel:
```bash
./train_JetClass.sh ParT full --gpus 0,1,2,3 --batch-size [total_batch_size]
```

Using DistributedDataParallel:
```bash
DDP_NGPUS=4 ./train_JetClass.sh ParT full --batch-size [batch_size_per_gpu]
```

**Other Datasets:**

```bash
# QuarkGluon dataset
./train_QuarkGluon.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin|kinpid|kinpidplus]

# TopLandscape dataset
./train_TopLandscape.sh [ParT|ParT-FineTune|PN|PN-FineTune|PFN|PCNN] [kin]
```

Use `ParT-FineTune` or `PN-FineTune` to fine-tune from [pre-trained JetClass models](models/).

---

## Citation

If you use this work, please cite the original Particle Transformer paper:

```bibtex
@article{qu2022particle,
  title={Particle Transformer for Jet Tagging},
  author={Qu, Huilin and Li, Congqiao and Qian, Sitian},
  journal={arXiv preprint arXiv:2202.03772},
  year={2022}
}
```
