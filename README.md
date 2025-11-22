# Noise2Average
Keras implementation for Noise2Average. Here we provide the startcode of Noise2Average and simulation data for running the Noise2Average. As shown in our paper in reference, Noise2Average can be also applied in denoising diffusion data and has superior performance in denoising empirical data where <font color=red>**the misalignment cannot be ignored.**</font> 

In this repository, we provide demonstration on simulation data. To figure out how Noise2Average works in empirical data, please use your own dataset or refer to the data availability.

## 1. Network architecture
![](https://github.com/birthlab/Noise2Average/blob/main/Images/cnn.jpg)


**Fig1. Learning strategies.** Learning strategies of supervised (a), Noise2Noise (b) and Noise2Average (c) denoising are illustrated for the case of two repetitions of noisy images. Supervised denoising trains a CNN to map the average of the two noisy images to its residual image compared to the ground-truth image with high signal-to-noise ratio (SNR) (a). Noise2Noise trains a CNN to map one noisy image to the other noisy image (b). Noise2Average performs supervised residual learning iteratively, with the training target as the average of the two noisy images with slightly higher SNR for iteration 1, and the average of denoised images from iteration _i_-1 for iteration _i_ (_i_ = 2, 3, 4, …) (c).

![](https://github.com/birthlab/Noise2Average/blob/main/Images/MUnets.jpg)

**Fig2. Convolutional neural networks.** 3D modified U-Nets (MU-Nets) excluding pooling and up-sampling layers are adopted in this study. For denoising T1-weighted images at higher spatial resolution, a deeper MU-Net (18 layers) with a larger receptive field is employed (a), with 64 channels at intermediate layers. For denoising diffusion images at lower resolution but with much more image channels, a slightly shallower MU-Net (10 layers) is employed, with 192 channels at intermediate layers and batch normalization before every 3D convolution layer.

## 2. Noise2Average denoising efficacy

## 3. Tutorial
- **./PretrainModel**: contains and saves the pretrain model
- superv_lr1e4_ep20.h5: the model pretrained on WU-Minn-Oxford HCP T1-weighted data (for more details please refer to reference 2)
- superv_lr1e4_ep20.mat: the training process of pretrain model

- **./Data/sub_001**: contains simulated T1-weighted images for denoising which were slightly misaligned after registration
- T1_mask.nii.gz: the brain mask
- T1_noise1.nii.nii.gz: the 1st repetition of simulated T1-weighted image
- T1_noise2.nii.nii.gz: the 2nd repetition of simulated T1-weighted image

- **./Code**: the code for denoising
- munet_res.py: contains CNN of Fig2.a
- s_N2A_cnnTrainN2A.py: contains the training/finetuning process of Noise2Average in Fig1.c


**Run in local:**
- 1. install the package in requirements.txt where we found that tensorflow 2.15 also works
- 2. cd ./Code
- 3. python s_N2A_cnnTrainN2A.py

## 4. Data availability

T1w and diffusion MRI data from the Human Connectome Project, WU-Minn-Ox Consortium and MGH-USC Consortium,  are publicly available ([https://www.humanconnectome.org](https://www.humanconnectome.org)). Diffusion MRI data from the Lifespan Human Connectome Project in Aging are publicly available ([https://www.humanconnectome.org/study/hcp-lifespan-aging](https://www.humanconnectome.org/study/hcp-lifespan-aging)). OVGU 7T T1w data are publicly available ([http://open-science.ub.ovgu.de/xmlui/handle/684882692/61](http://open-science.ub.ovgu.de/xmlui/handle/684882692/61)). MGH gSlider-SMS diffusion MRI data are publicly available ([https://datadryad.org/stash/dataset/doi:10.5061/dryad.nzs7h44q2](https://datadryad.org/stash/dataset/doi:10.5061/dryad.nzs7h44q2) and [https://datadryad.org/stash/dataset/doi:10.5061/dryad.rjdfn2z8g](https://datadryad.org/stash/dataset/doi:10.5061/dryad.rjdfn2z8g)). MGH T1w Wave-MPRAGE and ME-MPRAGE data are available from the corresponding author upon reasonable request.
