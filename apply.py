import numpy as np
import imageio
import torch
from torch.utils.data import DataLoader
import msd_pytorch as mp
from pathlib import Path
from scipy import ndimage
import argparse
#Result visualization
import matplotlib.pyplot as plt
from skimage import morphology

def compute_f1(pred, tg):
    '''Computes F1 score to evaluate accuracy of segmentation
    
    :param pred: Network prediction
    :type pred: :class:`np.ndarray`
    :param tg: Ground-truth segmentatin
    :type tg: :class:`np.ndarray`
    
    :return: F1 score
    :rtype: :class:`float`
    '''
    tp = np.count_nonzero(np.logical_and(pred == 1, tg == 1))
    fp = np.count_nonzero(np.logical_and(pred == 1, tg == 0))
    fn = np.count_nonzero(np.logical_and(pred == 0, tg == 1))
    
    if tp == 0:
        f1 = 0
    else:
        f1 = 2*float(tp) / (2*tp + fp + fn)
    
    return f1

def extract_fo_properties(mat):
    '''Computes different properties of the defect based on the distribution of material thickness
    
    :param mat: 2D material thickness distribution (line integrals of segmentation). Consists of 2 channels: cylinder material and void (defect)
    :type mat: :class:`list`
    
    :return: List of properties: location of the defect (array mask), the largest thickness and area
    :rtype: :class:`tuple`
    '''
    mat = np.array(mat)
    fo_map = mat[1,:]
    fo = fo_map > 0
    
    area = np.count_nonzero(fo)
    if area > 0:
        fo_th = fo_map.max()
    else:
        fo_th = 0
    
    return fo, fo_th, area

def extract_im_properties(fo, log, proj, scat):
    '''Computes different image properties of the projection and scattered signal
    
    :param fo: Defect mask
    :type fo: :class:`np.ndarray`
    :param log: Logarithm-corrected projection to extract attenuation intensity
    :type log: :class:`np.ndarray`
    :param proj: Raw projection before corrections
    :type proj: :class:`np.ndarray`
    :param scat: Distribution of scattered X-ray
    :type scat: :class:`np.ndarray`

    
    :return: List of properties of the defect region: average attenuation and scattering to primary ratio
    :rtype: :class:`tuple`
    '''
    if np.count_nonzero(fo) > 0.:
        mean_att = log[fo].mean()
        
        #scat = ndimage.gaussian_filter(scat, sigma=5)
        #print('Area: ', np.count_nonzero(fo))
        #print('Att: ', mean_att)
        
        prim = proj - scat        
        bg = morphology.binary_dilation(morphology.binary_dilation(fo)).astype(np.uint8) - fo.astype(np.uint8)
        ext_fo = morphology.binary_dilation(morphology.binary_dilation(fo))
                
        prim_bg = prim[bg == 1]
        prim_fo = prim[fo]
        scat_fo = scat[fo]
        div = scat[ext_fo] / prim[ext_fo]
        
        #print('Prim: {:.2f} +- {:.2f}'.format(prim_fo.mean(), prim_fo.std()))
        #print('Prim bg: {:.2f} +- {:.2f}'.format(prim_bg.mean(), prim_bg.std()))
        #print('Scat = {:.2f} +- {:.2f}'.format(scat_fo.mean(), scat_fo.std()))
        #print('Div = {:.2f} +- {:.2f}'.format(div.mean(), div.std()))
                
        scat_fract = div.mean()
        scat_std = div.std()
    else:
        mean_att = 0.
        scat_fract = 0.
        scat_std = 0.
    
    return mean_att, scat_fract, scat_std

def make_comparison(inp, tg, pred, fname):
    '''Visualizes the network prediction and compares it with ground-truth
    
    :param inp: Input image
    :type inp: :class:`np.ndarray`
    :param pred: Network prediction
    :type pred: :class:`np.ndarray`
    :param tg: Ground-truth segmentatin
    :type tg: :class:`np.ndarray`
    :param fname: File name to save the image
    :type fname: :class:`str`
    '''
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    segmented_im = np.zeros((*inp.shape, 3))
    display_im = (inp - inp.min()) / (inp.max() - inp.min()) * 1.
    
    ax[0].imshow(display_im, cmap='gray')
    ax[0].set_title('Input image', fontsize=24)
    
    for i in range(3):
        segmented_im[:,:,i] = display_im
    # Draw ground-truth boundary in red channel
    tg_map = morphology.binary_dilation(tg) - tg
    segmented_im[tg_map == 1, :] = 0.
    segmented_im[tg_map == 1, 0] = 1.
    # Draw prediction boundary in green channel
    pred_map = morphology.binary_dilation(pred) - pred
    segmented_im[pred_map == 1, :] = 0.
    segmented_im[pred_map == 1, 1] = 1.
    
    ax[1].imshow(segmented_im)
    ax[1].set_title('Segmentation comparison', fontsize=24)
    ax[1].text(40, 230, 'R = GT, G = Prediction', color='r', fontsize=16)
    f1 = compute_f1(pred, tg)
    ax[1].text(40, 250, "F1 score = {:.0%}".format(f1), color='r', fontsize=16)
    plt.savefig(fname)
    plt.clf()

def test(test_folder, nn_name, best_epoch, save_comparison=False):
    '''Tests the network on a test dataset. F1 score is used as accuracy metric.
    
    :param test_folder: Test dataset folder
    :type test_folder: :class:`pathlib.PosixPath`
    :param nn_name: Network name
    :type nn_name: :class:`str`
    :param best_epoch: Epoch to load
    :type best_epoch: :class:`int`
    :param save_comparison: Flag to save images comparing network prediction and gt. Slows the testing process.
    :type save_comparison: :class:`bool`
    '''
    c_in = 1
    depth = 30
    width = 1
    dilations = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    labels = [0, 1]
    batch_size = 3
    
    save_folder = Path('./network_state/')
    res_folder = Path('./res/')
    if save_comparison:
        (res_folder / nn_name).mkdir(exist_ok=True)
    
    model = mp.MSDSegmentationModel(c_in, len(labels), depth, width, dilations=dilations)
    epoch = model.load(save_folder / nn_name / '{:04d}.torch'.format(best_epoch))
    test_input_glob = test_folder / "inp" / "*.tiff"
    test_target_glob = test_folder / "tg" / "*.tiff"
    ds = mp.ImageDataset(test_input_glob, test_target_glob)
    dl = DataLoader(ds, batch_size, shuffle=False)
    res = []
    
    for i, data in enumerate(dl):
        inp, tg = data
        output = model.net(inp.cuda())
        prediction = torch.max(output.data, 1).indices
        prediction_numpy = prediction.detach().cpu().numpy()
        inp_numpy = inp.detach().cpu().numpy().squeeze(1)
        tg_numpy = tg.detach().cpu().numpy().squeeze(1)
        for b in range(batch_size):
            output_index = i * batch_size + b
            if output_index < len(ds):
                acc = compute_f1(prediction_numpy[b], tg_numpy[b])
                res.append(acc)
                if save_comparison:
                    make_comparison(inp_numpy[b], tg_numpy[b], prediction_numpy[b], res_folder / nn_name / 'output_{:04}.png'.format(output_index))
                
    return np.array(res)

def test_model(test_folder, base_name):
    '''Tests multiple instances of the same model and returns the results file containing model accuracy and properties of the individual test images
    
    :param test_folder: Test dataset folder
    :type test_folder: :class:`pathlib.PosixPath`
    :param base_name: Base name used by all instances of the model
    :type base_name: :class:`str`
    
    :return: Array with test results and image properties
    :rtype: :class:`np.ndarray`
    '''
    save_folder = Path('./network_state/')
    inp_fnames = sorted((test_folder / 'inp').glob('*.tiff'))
    stats_file = np.genfromtxt(test_folder / '../../stats.csv', delimiter=',', names=True)
    
    print(test_folder.parts[-2])
    
    if test_folder.parts[-2] == 'radon':
        print(base_name, ' Radon')
        log_folder = 'log_radon'
    elif test_folder.parts[-2] == 'mc':
        print(base_name, ' MC')
        log_folder = 'log_mc'
    else:
        print('Unknown model type, default to Radon')
        log_folder = 'log_radon'
    
    networks = [x for x in save_folder.iterdir() if x.is_dir()]
    networks = sorted(filter(lambda x: x.name.startswith(base_name), networks))
    print(networks)
    num_networks = len(networks)
    
    # First 3 fields are image properties
    image_fields = 8
    res_arr = np.zeros((len(inp_fnames), image_fields+num_networks+1))
    
    for i in range(len(inp_fnames)):
        im_num = int(inp_fnames[i].stem)
        cyl_r = stats_file['cyl_r'][im_num]
        cav_r = stats_file['cav_r'][im_num]
        mat = imageio.mimread(test_folder / '../../mat' / inp_fnames[i].name)
        log = imageio.imread(test_folder / '../..' / log_folder / inp_fnames[i].name)
        proj = imageio.imread(test_folder / '../../proj' / inp_fnames[i].name)
        scat = imageio.imread(test_folder / '../../scat' / inp_fnames[i].name)
        fo_mask, fo_th, area = extract_fo_properties(mat)
        fo_att, scat_fract, scat_std = extract_im_properties(fo_mask, log, proj, scat)
        res_arr[i,:image_fields] = i, fo_th, area, cyl_r, cav_r/cyl_r, fo_att, scat_fract, scat_std
    
    for i in range(num_networks):
        nn_name = networks[i]
        print(nn_name)
        epochs = sorted(nn_name.glob('*.torch'))
        # epochs[-1] is the last training epoch, not the best one 
        best_epoch = int(epochs[-2].stem)
        print('Best epoch - ', best_epoch)
        acc = test(test_folder, nn_name.parts[-1], best_epoch)
        res_arr[:,image_fields+1+i] = acc
    
    res_arr[:,image_fields] = res_arr[:,image_fields+1:].mean(axis=1)
    
    return res_arr

def batch_mode(test_root, res_folder, mat_name = 'pl90'):
    res_arr = test_model(test_root / '{}_test/radon/test'.format(mat_name), '{}_r'.format(mat_name))
    np.savetxt(res_folder / '{}_r_r.csv'.format(mat_name), res_arr, delimiter=',',
                header='ID,FO_th,Area,Cyl_R,Cav_Pos,FO_att,Scat_fract,Scat_std,Mean,' + ','.join(['Iter{}'.format(i+1) for i in range(res_arr.shape[1]-9)]))
    res_arr = test_model(test_root / '{}_test/mc/test'.format(mat_name), '{}_r'.format(mat_name))
    np.savetxt(res_folder / '{}_r_mc.csv'.format(mat_name), res_arr, delimiter=',',
                header='ID,FO_th,Area,Cyl_R,Cav_Pos,FO_att,Scat_fract,Scat_std,Mean,' + ','.join(['Iter{}'.format(i+1) for i in range(res_arr.shape[1]-9)]))
    res_arr = test_model(test_root / '{}_test/radon/test'.format(mat_name), '{}_mc'.format(mat_name))
    np.savetxt(res_folder / '{}_mc_r.csv'.format(mat_name), res_arr, delimiter=',',
                header='ID,FO_th,Area,Cyl_R,Cav_Pos,FO_att,Scat_fract,Scat_std,Mean,' + ','.join(['Iter{}'.format(i+1) for i in range(res_arr.shape[1]-9)]))
    res_arr = test_model(test_root / '{}_test/mc/test'.format(mat_name), '{}_mc'.format(mat_name))
    np.savetxt(res_folder / '{}_mc_mc.csv'.format(mat_name), res_arr, delimiter=',',
                header='ID,FO_th,Area,Cyl_R,Cav_Pos,FO_att,Scat_fract,Scat_std,Mean,' + ','.join(['Iter{}'.format(i+1) for i in range(res_arr.shape[1]-9)]))
    
if __name__ == "__main__":
    test_root = Path('/export/scratch2/vladysla/Data/Simulated/MC/Server_tmp')
    res_folder = Path('./test_res')
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=str, help='Batch mode: execute the function instead')
    parser.add_argument('--name', type=str, required=False, help='Folder with network weights')
    parser.add_argument('--test', type=str, required=False, help='Folder with the test set')
    parser.add_argument('--out', type=str, required=False, default='res', help='Name for the file with results')
    args = parser.parse_args()
    
    if args.batch:
        batch_mode(test_root, res_folder, args.batch)
    else:
        test_folder = test_root / args.test
        res_arr = test_model(test_folder, args.name)
        np.savetxt(res_folder / '{}.csv'.format(args.out), res_arr, delimiter=',',
                header='ID,FO_th,Area,Cyl_R,Cav_Pos,FO_att,Scat_fract,Scat_std,Mean,' + ','.join(['Iter{}'.format(i+1) for i in range(res_arr.shape[1]-9)]))
