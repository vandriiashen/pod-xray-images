import numpy as np
import imageio
from pathlib import Path
import argparse
import shutil

#remove
import matplotlib.pyplot as plt


def ff_cor_base(im, ff, im_total, ff_total):
    '''Performs flatfield correction for a projection.
    
    :param im: Array with a projection.
    :type im: :class:`np.ndarray`
    :param ff: Flatfield image.
    :type ff: :class:`np.ndarray`
    :param im_total: Number of photons per acquisition in the projection.
    :type im_total: :class:`int`
    :param ff_total: Number of photons per acquisition in the flatfield image.
    :type ff_total: :class:`int`
    
    :return: Flatfield corrected image
    :rtype: :class:`np.ndarray`
    '''
    ff = ff.astype(np.float32)
    im = im.astype(np.float32)
    
    ff *= float(im_total) / float(ff_total)
    # avoid /0
    ff[ff < 1] = 1
    # with high absorption it is possible that all photons are scattered, and radon projection is 0
    im[im < 1] = 1
    
    im /= ff
    im = -np.log(im)
    # We need to remove noisy regions with not enough flatfield
    im[ff < 5000] = 0.
    
    return im
    
def normalize_im(im):
    '''Normalize a corrected image accounting for mean value and standard deviation.
    
    :param im: Array with a corrected projection.
    :type im: :class:`np.ndarray`
    
    :return: Normalized image
    :rtype: :class:`np.ndarray`
    '''
    im -= im.mean()
    im /= im.std()
    return im

def log_cor(data_folder, ff_cor):
    '''Process projections in the data folder.
    Processing includes flatfield- and logarithm-correction
    Outputs to sets of images: Radon - with substracted scattering, MC - with scattering.
    
    :param data_folder: Folder with raw data
    :type data_folder: :class:`pathlib.PosixPath`
    :param ff_cor: Function to perform flatfield-correction. Based on ff_cor_base with fixed flatfield image and total number of photons
    :type ff_cor: :class:`function`

    '''
    paths = sorted((data_folder / 'proj').glob('*.tiff'))
    (data_folder / 'log_mc').mkdir(exist_ok = True)
    (data_folder / 'log_radon').mkdir(exist_ok = True)
    fnames = [path.name for path in paths]
    for fname in fnames:
        proj = imageio.imread(data_folder / 'proj' / fname)
        scat = imageio.imread(data_folder / 'scat' / fname)
        radon = proj - scat
        log_mc = ff_cor(proj)
        log_radon = ff_cor(radon)
        imageio.imwrite(data_folder / 'log_mc' / fname, log_mc.astype(np.float32))
        imageio.imwrite(data_folder / 'log_radon' / fname, log_radon.astype(np.float32))
        
def create_datasets(data_folder):
    '''Split corrected images between training and validation. 
    Normalize images with respect to mean and std to make data with and without scattering similar for NNs (otherwise, scattering introduces an offset).
    
    :param data_folder: Folder with raw data
    :type data_folder: :class:`pathlib.PosixPath`
    '''
    (data_folder / 'mc').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'train').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'train' / 'inp').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'train' / 'tg').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'val').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'val' / 'inp').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'val' / 'tg').mkdir(exist_ok = True)
    (data_folder / 'radon').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'train').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'train' / 'inp').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'train' / 'tg').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'val').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'val' / 'inp').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'val' / 'tg').mkdir(exist_ok = True)
    
    paths = sorted((data_folder / 'proj').glob('*.tiff'))
    fnames = [path.name for path in paths]
    total_num = len(fnames)
    # 20% of images is for validation
    val_num = total_num // 5
    train_num = total_num - val_num
    print('Images for training - {}, for validation - {}'.format(train_num, val_num))
    
    for i in range(train_num):
        log_mc = imageio.imread(data_folder / 'log_mc' / fnames[i])
        imageio.imwrite(data_folder / 'mc' / 'train' / 'inp' / fnames[i], (normalize_im(log_mc)).astype(np.float32))
        shutil.copy(data_folder / 'segm' / fnames[i], data_folder / 'mc' / 'train' / 'tg' / fnames[i])
        log_radon = imageio.imread(data_folder / 'log_radon' / fnames[i])
        imageio.imwrite(data_folder / 'radon' / 'train' / 'inp' / fnames[i], (normalize_im(log_radon)).astype(np.float32))
        shutil.copy(data_folder / 'segm' / fnames[i], data_folder / 'radon' / 'train' / 'tg' / fnames[i])
    for i in range(train_num, train_num + val_num):
        log_mc = imageio.imread(data_folder / 'log_mc' / fnames[i])
        imageio.imwrite(data_folder / 'mc' / 'val' / 'inp' / fnames[i], (normalize_im(log_mc)).astype(np.float32))
        shutil.copy(data_folder / 'segm' / fnames[i], data_folder / 'mc' / 'val' / 'tg' / fnames[i])
        log_radon = imageio.imread(data_folder / 'log_radon' / fnames[i])
        imageio.imwrite(data_folder / 'radon' / 'val' / 'inp' / fnames[i], (normalize_im(log_radon)).astype(np.float32))
        shutil.copy(data_folder / 'segm' / fnames[i], data_folder / 'radon' / 'val' / 'tg' / fnames[i])
        
def create_test_datasets(data_folder):
    '''Make test datasets
    
    :param data_folder: Folder with raw data
    :type data_folder: :class:`pathlib.PosixPath`
    '''
    (data_folder / 'mc').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'test').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'test' / 'inp').mkdir(exist_ok = True)
    (data_folder / 'mc' / 'test' / 'tg').mkdir(exist_ok = True)
    (data_folder / 'radon').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'test').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'test' / 'inp').mkdir(exist_ok = True)
    (data_folder / 'radon' / 'test' / 'tg').mkdir(exist_ok = True)
    
    paths = sorted((data_folder / 'proj').glob('*.tiff'))
    fnames = [path.name for path in paths]
    total_num = len(fnames)
    for i in range(total_num):
        log_mc = imageio.imread(data_folder / 'log_mc' / fnames[i])
        imageio.imwrite(data_folder / 'mc' / 'test' / 'inp' / fnames[i], (normalize_im(log_mc)).astype(np.float32))
        shutil.copy(data_folder / 'segm' / fnames[i], data_folder / 'mc' / 'test' / 'tg' / fnames[i])
        log_radon = imageio.imread(data_folder / 'log_radon' / fnames[i])
        imageio.imwrite(data_folder / 'radon' / 'test' / 'inp' / fnames[i], (normalize_im(log_radon)).astype(np.float32))
        shutil.copy(data_folder / 'segm' / fnames[i], data_folder / 'radon' / 'test' / 'tg' / fnames[i])

if __name__ == "__main__":
    data_root = Path('/export/scratch2/vladysla/Data/Simulated/MC/')
    ff_fname = '/export/scratch2/vladysla/Data/Simulated/MC/Server_tmp/scan_ff/proj/0000.tiff'
    im_total = 10**9
    ff_total = 10**10
    ff = imageio.imread(ff_fname)
    ff_cor = lambda proj: ff_cor_base(proj, ff, im_total, ff_total)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=str, required=False, help='Folder with the training set')
    parser.add_argument('--test', type=str, required=False, help='Folder with the test set')
    args = parser.parse_args()
    
    if args.train is not None:
        log_cor(data_root / args.train, ff_cor)
        create_datasets(data_root / args.train)
    elif args.test is not None:
        log_cor(data_root / args.test, ff_cor)
        create_test_datasets(data_root / args.test)
    else:
        print('No data folder')
