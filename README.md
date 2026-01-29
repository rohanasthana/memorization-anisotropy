<h2 align="center">[ICLR '26] Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability</h2>
<p align="center">
Rohan Asthana, Vasileios Belagiannis
</p>

<p align="center">
  <a href="https://rohanasthana.github.io/memorization_anisotropy/">
    <img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page">
  </a>
  <a href="https://arxiv.org/abs/2601.20642">
    <img src="https://img.shields.io/badge/arXiv-2601.20642-b31b1b.svg?style=for-the-badge" alt="arXiv">
  </a>
</p>

<p align="center">
  <img src="alignment_full.jpg" width="500">
</p>


This repository contains the official code for the ICLR '26 paper "Detecting and Mitigating Memorization in Diffusion Models through Anisotropy of the Log-Probability".

## Abstract
Diffusion-based image generative models produce high-fidelity images through iterative denoising but remain vulnerable to memorization, where they unintentionally reproduce exact copies or parts of training images. Recent memorization detection methods are primarily based on the norm of score difference as indicators of memorization. We prove that such norm-based metrics are mainly effective under the assumption of isotropic log-probability distributions, which generally holds at high or medium noise levels. In contrast, analyzing the anisotropic regime reveals that memorized samples exhibit strong angular alignment between the guidance vector and unconditional scores in the low-noise setting. Through these insights, we develop a memorization detection metric by integrating isotropic norm and anisotropic alignment. Our detection metric can be computed directly on pure noise inputs via two conditional and unconditional forward passes, eliminating the need for costly denoising steps. Detection experiments on Stable Diffusion v1.4 and v2 show that our metric outperforms existing denoising-free detection methods while being at least approximately 5x faster than the previous best approach. Finally, we demonstrate the effectiveness of our approach by utilizing a mitigation strategy that adapts memorized prompts based on our developed metric.

## Getting Started

To get started, follow these steps:

### 1. Create the conda environment

```bash
conda env create -f environment.yml
```

### 2. Activate the environment

```bash
conda activate sail
```
### 3. Run detection
```bash
bash run_detection.sh
```
### 4. Evaluate detection results
```bash
python detect_eval.py
```

## Acknowledgments
This code is based upon the repositories [Understanding and Mitigating Memorization in Generative Models via Sharpness of Probability Landscapes](https://github.com/Dongjae0324/sharpness_memorization_diffusion) and [Detecting, Explaining, and Mitigating Memorization in Diffusion Models](https://github.com/YuxinWenRick/diffusion_memorization). The memorized and non memorized prompt data is taken from [here](https://github.com/Dongjae0324/sharpness_memorization_diffusion).

## License
This project is licensed under the [MIT License](LICENSE).
