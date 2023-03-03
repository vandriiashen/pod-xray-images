import msd_pytorch as mp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from skimage import morphology
from scipy import ndimage
from tqdm import tqdm
import imageio
import torch
import shutil
from skimage import filters, feature
        
def train(dataset_folder, nn_name):
    '''Here
    '''
    c_in = 1
    depth = 30
    width = 1
    dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [0, 1]
    #loss = "L2"
    c_out = 1
    task = "segmentation"
    batch_size = 3
    epochs = 500

    train_input_glob = dataset_folder / 'train' / 'inp' / '*.tiff'
    train_target_glob = dataset_folder / 'train' / 'tg' / '*.tiff'
    val_input_glob = dataset_folder / 'val' / 'inp' / '*.tiff'
    val_target_glob = dataset_folder / 'val' / 'tg' / '*.tiff'

    print("Load training dataset")
    train_ds = mp.ImageDataset(train_input_glob, train_target_glob, labels=labels)
    train_dl = DataLoader(train_ds, batch_size, shuffle=True)
    val_ds = mp.ImageDataset(val_input_glob, val_target_glob, labels=labels)
    val_dl = DataLoader(val_ds, batch_size, shuffle=False)

    model = mp.MSDSegmentationModel(c_in, train_ds.num_labels, depth, width, dilations=dilations)

    print("Start estimating normalization parameters")
    model.set_normalization(train_dl)
    print("Done estimating normalization parameters")

    print("Starting training...")
    best_validation_error = np.inf
    best_epoch = -1
    validation_error = 0.0
    save_folder = Path('./network_state/')
    (save_folder / nn_name).mkdir(exist_ok = True)
    log_path = Path('./log/')
    (log_path / nn_name).mkdir(exist_ok = True)
    logger = SummaryWriter(log_path / nn_name)

    for epoch in tqdm(range(epochs)):
        # Train
        model.train(train_dl, 1)
        # Compute training error
        train_error = model.validate(train_dl)
        print(f"{epoch:05} Training error: {train_error: 0.6f}")
        logger.add_scalar('Loss/train', train_error, epoch)
        # Compute validation error
        if val_dl is not None:
            validation_error = model.validate(val_dl)
            print(f"{epoch:05} Validation error: {validation_error: 0.6f}")
            logger.add_scalar('Loss/validation', train_error, epoch)
        # Save network if worthwile
        if validation_error < best_validation_error or val_dl is None:
            best_validation_error = validation_error
            best_epoch = epoch
            model.save(save_folder / nn_name / '{:04d}.torch'.format(epoch), epoch)

    # Save final network parameters
    model.save(save_folder / nn_name / '{:04d}.torch'.format(epoch), epoch)
    return best_epoch
    
def train_repeat(train_folder, base_name, iter_num):
    '''To train
    '''
    for i in range(0, 0+iter_num):
        nn_name = '{}{}'.format(base_name, i+1)
        train(train_folder, nn_name)
                
if __name__ == "__main__":
    root_folder = Path('/export/scratch2/vladysla/Data/Simulated/MC/Server_tmp/')

    train_folder = root_folder / 'fe450_train/mc/'
    nn_name = 'fe450_mc_'
    train_repeat(train_folder, nn_name, 4)
    
    train_folder = root_folder / 'fe450_train/radon/'
    nn_name = 'fe450_r_'
    train_repeat(train_folder, nn_name, 4)