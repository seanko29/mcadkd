# MCADKD - ICCV Workshop 2025

### Multi-scale Contrastive and Adversarial Knowledge Distillation for SISR

#### [Donggeun Ko<sup>1,2</sup>](your-website-url), [Youngsang Kwak<sup>2</sup>](co-author-url), [San Kim<sup>2</sup>](co-author-url), [Jaekwang Kim<sup>2</sup>](co-author-url)

#### **<sup>1</sup> Sungkyunkwan University - <sup>2</sup> AiM Future Inc.**

[![paper](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://openaccess.thecvf.com/content/ICCV2025W/AIGENS/papers/Ko_Multi-Scale_Contrastive-Adversarial_Distillation_for_Super-Resolution_ICCVW_2025_paper.pdf)

## Latest
- `10/2024`: Code & checkpoints release.
- `09/2024`: MCADKD has been accepted at ICCV Workshop 2025 at [AIGENS2025](https://ai4streaming-workshop.github.io/)! 🎉 

## Method:
<br>
<details>
  <summary>
  <font size="+1">Abstract</font>
  </summary>
Knowledge distillation (KD) is a powerful technique for model compression, enabling the creation of compact and efficient "student" models by transferring knowledge from large-scale, pre-trained "teacher" models. However, the application of traditional KD methods in this domain is considerably more challenging than in high-level tasks like classification, as the SISR task is to reconstruct image pixels a regression problem. Hence, to effectively distill the knowledge of a teacher model in SR, we propose MCAD-KD, Multi-Scale Contrastive–Adversarial Distillation for Super-Resolution. We utilize a novel hybrid contrastive learning framework that operates on both global (image-level) and local (patch-level) scales. Furthermore, we integrate adversarial guidance, which pushes the student's output towards the manifold of realistic images, allowing it to potentially surpass the perceptual quality of the teacher by learning directly from the ground-truth data distribution. Our comprehensive framework synergistically combines these components to train a lightweight student model that achieves a superior trade-off between perceptual quality and computational efficiency.
</details>

<p align="center">
<img src="https://github.com/user-attachments/assets/f7fd13a8-fc06-459a-92d0-1a76cae3b90a" width="800"/>
</p>

**Framework Overview:**

<p align="center">
<a href="https://github.com/user-attachments/files/23337530/SR-San-2.pdf">
<img src="https://img.shields.io/badge/View-Detailed_Architecture-blue?style=for-the-badge" alt="Detailed Architecture"/>
</a>
</p>

## Results:

<details>
  <summary>
  <font size="+1">Main Results</font>
  </summary>

The model achieves competitive performance with significantly reduced parameters compared to teacher networks, while maintaining high PSNR and SSIM scores on standard benchmarks (Set5, Set14, BSD100, Urban100, Manga109).

<p align="center">
<img src="path/to/your/results_table.png" width="900"/>
</p>

</details>

<details>
  <summary>
  <font size="+1">Visual Comparison</font>
  </summary>

<p align="center">
<img src="path/to/your/visual_comparison.png" width="900"/>
</p>

</details>


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
<br>
<details>
  <summary>
  <font size="+1">**Contrastive Learning Options**</font>
  </summary>
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

</details>



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


## Results

The model achieves competitive performance with significantly reduced parameters compared to teacher networks, while maintaining high PSNR and SSIM scores on standard benchmarks (Set5, Set14, BSD100, Urban100, Manga109).

## Citation

If you find this work helpful, please consider citing:

```bibtex
@inproceedings{ko2025multi,
  title={Multi-Scale Contrastive-Adversarial Distillation for Super-Resolution},
  author={Ko, Donggeun and Kwak, Youngsang and Kim, San and Kwak, Jaehwa and Kim, Jaekwang},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={5069--5078},
  year={2025}
}
```

## Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
