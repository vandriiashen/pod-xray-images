# The code for augmenations, dataset and Lightning module is adapted from https://github.com/qubvel/segmentation_models.pytorch/tree/master/examples
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import albumentations as albu
import segmentation_models_pytorch as smp

from pathlib import Path
import numpy as np
import imageio.v3 as imageio
import matplotlib.pyplot as plt
from skimage import morphology

def get_training_augmentation():
    # Only use flip and translations, make dimensions divisible by 32 ()
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=1),
        albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0.2, shift_limit=0.1, p=1, border_mode=1),
        albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=1),
        albu.RandomCrop(height=320, width=320, always_apply=True),
    ]
    return albu.Compose(train_transform)

def get_validation_augmentation():
    test_transform = [
        albu.PadIfNeeded(320, 320, always_apply=True, border_mode=1)
    ]
    return albu.Compose(test_transform)

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

# don't use this because the networks were pre-trained on RGB images, not grayscale X-ray projections
def get_preprocessing(preprocessing_fn):
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

class Dataset(BaseDataset):
    def __init__(self, images_dir, masks_dir, augmentation=None, preprocessing=None):
        self.ids = os.listdir(images_dir)
        self.images_fps = sorted([os.path.join(images_dir, image_id) for image_id in self.ids])
        self.masks_fps = sorted([os.path.join(masks_dir, image_id) for image_id in self.ids])
        self.augmentation = augmentation
        self.preprocessing = preprocessing
    
    def __getitem__(self, i):
        image = imageio.imread(self.images_fps[i])
        mask = imageio.imread(self.masks_fps[i])
        # convert image with N labels into N channels corresponding to each lable
        masks = [(mask == v) for v in [1]]
        mask = np.stack(masks, axis=-1).astype('float')
        
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']
            
        image = np.expand_dims(image, 0)
        mask = np.swapaxes(mask, 0, 2)
            
        return {
            'name': self.images_fps[i],
            'image' : image,
            'mask'  : mask
        }
        
    def __len__(self):
        return len(self.ids)

class DefectModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.model = smp.create_model(arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs)

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        # Image are already normalized
        self.register_buffer("std", torch.tensor(1.))
        self.register_buffer("mean", torch.tensor(0.))

        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def forward(self, image):
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        image = batch["image"]
        assert image.ndim == 4

        # Common architectures have 5 stages of downsampling by factor 2, so image dimensions should be divisible by 32
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0
        
        mask = batch["mask"]
        assert mask.ndim == 4
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        loss = self.loss_fn(logits_mask, mask)
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        step_out = self.shared_step(batch, "train")
        self.training_step_outputs.append(step_out)
        self.log('train_loss', step_out['loss'])
        return step_out
    
    def on_train_epoch_end(self):
        self.shared_epoch_end(self.training_step_outputs, "train")
        # remove saved step outputs to prevent memory leak
        self.training_step_outputs = []

    def validation_step(self, batch, batch_idx):
        step_out = self.shared_step(batch, "valid")
        self.validation_step_outputs.append(step_out)
        self.log('val_loss', step_out['loss'])
        return step_out
    
    def on_validation_epoch_end(self):
        self.shared_epoch_end(self.validation_step_outputs, "valid")
        self.validation_step_outputs = []

    def test_step(self, batch, batch_idx):
        step_out = self.shared_step(batch, "test")
        self.test_step_outputs.append(step_out)
        return step_out 
    
    def on_test_epoch_end(self):
        # Test epoch end uses F1 as a metric
        outputs = self.test_step_outputs
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="macro-imagewise")
        metrics = {
            "test_f1_score": f1,
        }
        self.log_dict(metrics)
        self.test_step_outputs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)
    
def train(out_folder, log_folder, arch, encoder, train_folder):
    train_dataset = Dataset(
        train_folder / 'train/inp', 
        train_folder / 'train/tg', 
        augmentation=get_training_augmentation()
    )
    val_dataset = Dataset(
        train_folder / 'val/inp', 
        train_folder / 'val/tg', 
        augmentation=get_validation_augmentation()
    )
    
    model = DefectModel(arch, encoder, encoder_weights='imagenet', in_channels=1, out_classes=1)
    
    train_dataloader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8)
    val_dataloader = DataLoader(val_dataset, batch_size=10, shuffle=False, num_workers=8)
    
    tb_logger = TensorBoardLogger(save_dir=log_folder, name='{}_{}'.format(arch, encoder))
    checkpoint_callback = ModelCheckpoint(dirpath=out_folder, save_top_k=1, monitor="val_loss")
    trainer = pl.Trainer(max_epochs=300, callbacks=[checkpoint_callback], logger=[tb_logger])
    trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    
def train_multiple(mc_train_folder, mat_name, forward_type):
    arch_list = ['UnetPlusPlus', 'FPN', 'DeepLabV3Plus']
    encoder_list = ['tu-resnet18', 'tu-efficientnet_b0', 'tu-mobilenetv3_rw', 'tu-efficientnet_b4']
    for arch in arch_list:
        for encoder in encoder_list:
            log_folder = Path('./logs') / '{}_{}'.format(mat_name, forward_type)
            out_folder = Path('./out') / '{}_{}'.format(mat_name, forward_type) / '{}_{}_imagenet'.format(arch, encoder)
            log_folder.mkdir(exist_ok=True)
            out_folder.mkdir(exist_ok=True)
            train(out_folder, log_folder, arch, encoder, mc_train_folder)
    
# The next functions are copied from apply.py to provide the same results csv file
def compute_f1(pred, tg):
    tp = np.count_nonzero(np.logical_and(pred == 1, tg == 1))
    fp = np.count_nonzero(np.logical_and(pred == 1, tg == 0))
    fn = np.count_nonzero(np.logical_and(pred == 0, tg == 1))
    if tp == 0:
        f1 = 0
    else:
        f1 = 2*float(tp) / (2*tp + fp + fn)
    return f1
    
def make_comparison(inp, tg, pred, fname):
    fig, ax = plt.subplots(1, 2, figsize=(18,9))
    
    tg = np.swapaxes(tg, 0, 1)
    pred = np.swapaxes(pred, 0, 1)
    
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
    print(pred.shape)
    segmented_im[pred_map == 1, :] = 0.
    segmented_im[pred_map == 1, 1] = 1.
    
    ax[1].imshow(segmented_im)
    ax[1].set_title('Segmentation comparison', fontsize=24)
    ax[1].text(40, 280, 'R = GT, G = Prediction', color='r', fontsize=16)
    f1 = compute_f1(pred, tg)
    ax[1].text(40, 300, "F1 score = {:.0%}".format(f1), color='r', fontsize=16)
    plt.savefig(fname)
    plt.clf()

def extract_fo_properties(mat):
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
    if np.count_nonzero(fo) > 0.:
        mean_att = log[fo].mean()
        prim = proj - scat        
        bg = morphology.binary_dilation(morphology.binary_dilation(fo)).astype(np.uint8) - fo.astype(np.uint8)
        ext_fo = morphology.binary_dilation(morphology.binary_dilation(fo))
        prim_bg = prim[bg == 1]
        prim_fo = prim[fo]
        scat_fo = scat[fo]
        div = scat[ext_fo] / prim[ext_fo]
        scat_fract = div.mean()
        scat_std = div.std()
    else:
        mean_att = 0.
        scat_fract = 0.
        scat_std = 0.
    
    return mean_att, scat_fract, scat_std

def apply_model(out_folder, test_folder, res_folder, save_comparison=False):
    ckpt_list = sorted(out_folder.glob('*.ckpt'))
    print(ckpt_list[-1])
    model = DefectModel.load_from_checkpoint(ckpt_list[-1])
    
    test_dataset = Dataset(
        test_folder / 'inp', 
        test_folder / 'tg', 
        augmentation=get_validation_augmentation()
    )
    batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    res = []
    for i, data in enumerate(test_dataloader):
        with torch.no_grad():
            model.eval()
            logits = model(data['image'])
        prediction = torch.where(logits.sigmoid() > 0.5, 1, 0)
        prediction_numpy = prediction.detach().cpu().numpy().squeeze(1)
        inp_numpy = data['image'].detach().cpu().numpy().squeeze(1)
        tg_numpy = data['mask'].detach().cpu().numpy().squeeze(1)
        for b in range(batch_size):
            output_index = i * batch_size + b
            if output_index < len(test_dataset):
                acc = compute_f1(prediction_numpy[b], tg_numpy[b])
                res.append(acc)
                if save_comparison:
                    make_comparison(inp_numpy[b], tg_numpy[b], prediction_numpy[b], res_folder / 'output_{:04}.png'.format(output_index))
    return np.array(res)

def test_metrics(test_folder):
    test_dataset = Dataset(
        test_folder / 'inp', 
        test_folder / 'tg', 
        augmentation=get_validation_augmentation()
    )
    batch_size = 1
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    arch_list = ['UnetPlusPlus', 'FPN', 'DeepLabV3Plus']
    encoder_list = ['tu-efficientnet_b0', 'tu-mobilenetv3_rw', 'tu-densenet121']
    for arch in arch_list:
        if arch == 'DeepLabV3Plus':
            encoder_list = ['tu-efficientnet_b0', 'tu-mobilenetv3_rw']
        for encoder in encoder_list:
            print('{}_{}'.format(arch, encoder))
            out_folder = Path('./out') / mat_name / '{}_{}_imagenet'.format(arch, encoder)
            ckpt_list = sorted(out_folder.glob('*.ckpt'))
            print(ckpt_list[-1])
            model = DefectModel.load_from_checkpoint(ckpt_list[-1])
            
            trainer = pl.Trainer()
            trainer.test(model, dataloaders=test_dataloader)
    
def test_multiple(test_folder, mat_name, forward_type):
    log_folder = 'log_mc'
    inp_fnames = sorted((test_folder / 'inp').glob('*.tiff'))
    stats_file = np.genfromtxt(test_folder / '../../stats.csv', delimiter=',', names=True)
    
    image_fields = 8
    num_networks = 9
    res_arr = np.zeros((len(inp_fnames), image_fields+num_networks+1))
    
    for i in range(len(inp_fnames)):
        im_num = int(inp_fnames[i].stem)
        cyl_r = stats_file['cyl_r'][im_num]
        cav_r = stats_file['cav_r'][im_num]
        mat = []
        for img in imageio.imiter(test_folder / '../../mat' / inp_fnames[i].name):
            mat.append(img)
        log = imageio.imread(test_folder / '../..' / log_folder / inp_fnames[i].name)
        proj = imageio.imread(test_folder / '../../proj' / inp_fnames[i].name)
        scat = imageio.imread(test_folder / '../../scat' / inp_fnames[i].name)
        fo_mask, fo_th, area = extract_fo_properties(mat)
        fo_att, scat_fract, scat_std = extract_im_properties(fo_mask, log, proj, scat)
        res_arr[i,:image_fields] = i, fo_th, area, cyl_r, cav_r/cyl_r, fo_att, scat_fract, scat_std
    
    arch_list = ['UnetPlusPlus', 'FPN', 'DeepLabV3Plus']
    encoder_list = ['tu-efficientnet_b0', 'tu-mobilenetv3_rw', 'tu-efficientnet_b4']
    j = 0
    model_names = []
    for arch in arch_list:
        for encoder in encoder_list:
            model_names.append('{}_{}'.format(arch, encoder[3:]))
            print('{}_{}'.format(arch, encoder))
            res_folder = Path('./res') / '{}_{}'.format(mat_name, forward_type) / '{}_{}_imagenet'.format(arch, encoder)
            res_folder.mkdir(exist_ok = True)
            print(res_folder)
            out_folder = Path('./network_state') / '{}_{}'.format(mat_name, forward_type) / '{}_{}_imagenet'.format(arch, encoder)
            acc = apply_model(out_folder, test_folder, res_folder, save_comparison=False)
            res_arr[:,image_fields+1+j] = acc
            j += 1
    
    res_arr[:,image_fields] = res_arr[:,image_fields+1:].mean(axis=1)
    header = 'ID,FO_th,Area,Cyl_R,Cav_Pos,FO_att,Scat_fract,Scat_std,Mean,' + ','.join(model_names)
    
    return res_arr, header

if __name__ == '__main__':
    root_folder = Path('/export/scratch2/vladysla/Data/Simulated/MC/Server_tmp')
    mat_name = 'fe450'
    forward_type = 'r'
    mc_train_folder = root_folder / '{}_train/mc/'.format(mat_name)
    radon_train_folder = root_folder / '{}_train/radon/'.format(mat_name)
    mc_test_folder = root_folder / '{}_test/mc/test'.format(mat_name)
    
    (Path('./network_state') / '{}_{}'.format(mat_name, forward_type)).mkdir(exist_ok=True)
    (Path('./res') / '{}_{}'.format(mat_name, forward_type)).mkdir(exist_ok=True)
    (Path('./log') / '{}_{}'.format(mat_name, forward_type)).mkdir(exist_ok=True)

    #train_multiple(radon_train_folder, mat_name, forward_type)
    res_arr, header = test_multiple(mc_test_folder, mat_name, forward_type)
    np.savetxt('./test_res/smp_{}_{}_mc.csv'.format(mat_name, forward_type), res_arr, delimiter=',', header=header)
