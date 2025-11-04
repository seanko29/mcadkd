# MCADKD - Multi-scale Contrastive and Adversarial Knowledge Distillation for SISR

### ICCV Workshop 2025

## Overview

MCADKD (Multi-scale Contrastive and Adversarial Knowledge Distillation) is an advanced knowledge distillation framework for Single Image Super-Resolution (SISR). This method combines contrastive learning at multiple scales with adversarial training to effectively transfer knowledge from large teacher networks to compact student networks.

## Key Features

- **Hybrid Contrastive Learning**: Combines both image-level and patch-level contrastive learning for comprehensive feature alignment
- **Adaptive Temperature Learning**: Learnable temperature parameter in InfoNCE loss for better optimization
- **Adversarial Training**: Incorporates GAN-based adversarial loss for enhanced perceptual quality
- **Multi-scale Knowledge Transfer**: Transfers knowledge at different spatial scales for better detail preservation
- **Flexible Architecture Support**: Works with various SR architectures (SwinIR, EDSR, etc.)

## Method

MCADKD employs a multi-faceted distillation approach:

1. **Image-level Contrastive Loss**: Aligns global representations between student and teacher outputs
2. **Patch-level Contrastive Loss**: Captures fine-grained local details through patch-based learning
3. **Response-based Knowledge Distillation**: Direct supervision from teacher predictions
4. **Adversarial Loss**: Enhances perceptual quality through discriminator feedback
5. **Reconstruction Loss**: Ensures fidelity to ground truth images

The combined loss function enables effective knowledge transfer while maintaining computational efficiency in the student network.

## Installation

### Create a Conda Environment

```bash
ENV_NAME="mcadkd"
conda create -n $ENV_NAME python=3.10
conda activate $ENV_NAME
```

### Install Dependencies

Run the following script to install PyTorch and other dependencies:

```bash
bash install.sh
```

Or manually install:

```bash
# Install PyTorch (adjust CUDA version as needed)
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# Install basicsr in development mode
python setup.py develop
```

## Usage

### Training

Pre-trained teacher checkpoints should be placed in appropriate directories and specified in the config files.

For single-GPU training:
```bash
torchrun --nproc_per_node=1 --master_port=4321 basicsr/train.py -opt options/train/mcadkd_swinir_x2.yml --launcher pytorch
```

For multi-GPU training:
```bash
torchrun --nproc_per_node=4 --master_port=4321 basicsr/train.py -opt options/train/mcadkd_swinir_x2.yml --launcher pytorch
```

### Testing

```bash
python basicsr/test.py -opt [TEST_CONFIG_PATH]
```

## Configuration Files

The `options/train/` directory contains training configurations:

- `mcadkd_swinir_x2.yml`: SwinIR-based model for 2x super-resolution
- `mcadkd_swinir_x3.yml`: SwinIR-based model for 3x super-resolution
- `mcadkd_edsr_x2.yml`: EDSR-based model for 2x super-resolution

### Key Configuration Parameters

**Contrastive Learning Options** (`contrastive_opt`):
- `type`: Contrastive learning type (`image`, `patch`, or `hybrid`)
- `patch_size`: Size of patches for patch-level contrastive learning (default: 48)
- `num_patches`: Number of patches to sample per image (default: 8)
- `image_weight`: Weight for image-level contrastive loss (default: 0.3)
- `patch_weight`: Weight for patch-level contrastive loss (default: 0.7)
- `loss_weight`: Overall weight for contrastive loss (default: 0.05)
- `tau`: Initial temperature for InfoNCE loss (default: 0.07)
- `mining`: Mining strategy for hard negatives (default: 'hard')

**Loss Weights**:
- `pixel_opt.loss_weight`: Weight for reconstruction loss (default: 3.0)
- `resp_kd_opt.loss_weight`: Weight for response distillation (default: 0.5)
- `adversarial_opt.loss_weight`: Weight for adversarial loss (default: 0.01)

## Dataset Preparation

Update the dataset paths in the YAML configuration files:

```yaml
datasets:
  train:
    dataroot_gt: /path/to/your/DIV2K_train_HR
    dataroot_lq: /path/to/your/DIV2K_train_LR_bicubic/X2
  val:
    dataroot_gt: /path/to/your/benchmark/Set5/HR
    dataroot_lq: /path/to/your/benchmark/Set5/LR_bicubic/X2
```

## Model Architecture

MCADKD supports various SR architectures as student and teacher networks:

- **SwinIR**: Transformer-based architecture with shifted window attention
- **EDSR**: Enhanced Deep Residual Networks for Image Super-Resolution
- Custom architectures can be easily integrated through BasicSR's modular design

## Results

The model achieves competitive performance with significantly reduced parameters compared to teacher networks, while maintaining high PSNR and SSIM scores on standard benchmarks (Set5, Set14, BSD100, Urban100, Manga109).

## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{mcadkd2025,
  title={MCADKD: Multi-scale Contrastive and Adversarial Knowledge Distillation for Image Super-Resolution},
  author={Your Name},
  booktitle={ICCV Workshop},
  year={2025}
}
```

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
