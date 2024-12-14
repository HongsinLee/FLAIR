# FLAIR: Official Code for Adversarially Robust Class Incremental Learning (AAAI 2025)

This is the official repository for **"Enhancing Robustness in Incremental Learning with Adversarial Training" (AAAI 2025)**.  
We extend the codebase from [Mammoth](https://github.com/aimagelab/mammoth) to implement FLAIR, a novel method for enhancing adversarial robustness in Class Incremental Learning.

---

## Features

- **FLAIR Integration**: Implements our method to enhance robustness against adversarial attacks in incremental learning.
- **Dataset Support**: Out-of-the-box support for CIFAR-10, CIFAR-100, and other custom datasets.
- **Customizability**: Easily extend with new models and datasets.

---

## Installation

- Clone the repository:

   `git clone https://github.com/HongsinLee/FLAIR.git`
   
   `cd FLAIR`

- Set up a conda environment:

    `conda create -n flair_env python=3.8 `

    `conda activate flair_env`

- Install the required dependencies:

   `pip install -r requirements.txt`

---

## Running Experiments

### Quick Start

- For CIFAR-10 experiments:  
  `bash cifar10.sh`

- For CIFAR-100 experiments:  
  `bash cifar100.sh`

- For SVHN experiments:  
  `bash svhn.sh`

- For Tiny-ImageNet experiments:  
  `bash tinyimg.sh`

### Custom Method and Datasets

- Add your custom method files to the `models/` directory.  
- Add your custom dataset files to the `datasets/` directory.  



