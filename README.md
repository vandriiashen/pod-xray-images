# Collection of scripts to train MSD and compute POD curves

**make_dataset.py** - pre-processing of Monte-Carlo images and generating datasets

**train.py** and **apply.py** - training and test scripts for MSD based on pytorch implementation

**smp.py** - alternative script to train and test other network architectures. It is based on Lightning for training, Segmentation models for architectures and Albumentations for data augmentations.

**univariate_pod.py** - the first script for POD computation. It uses the output format of testing scripts.

**multivariate_pod.py** - more advanced script that computes POD not only as a function of defect size but also scattering-to-primary ratio.
