# VQ-VAE Implementation

Implementation of "Neural Discrete Representation Learning" with Vector Quantization for CIFAR10.

A comparison study of VQ, FSQ and VQ+Rotation.

![model](model_all.jpg)

## Setup
```bash
pip install -r requirements.txt
```

## Training
```bash
python main.py --batch_size 128 --epochs 100
```

## Parameters
- batch_size: default 128
- epochs: default 100
- lr: default 3e-4
- hidden_dims: default 256
- codebook_size: default 512
- decay: default 0.8
- commitment_weight: default 1.0