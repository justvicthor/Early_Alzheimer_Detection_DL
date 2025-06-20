import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from nilearn import datasets, image
from skimage.filters import threshold_otsu
from scipy.ndimage import label
from model import ClassifierCNN
import yaml

def load_cfg(cfg_path):
    with open(cfg_path, 'r') as f:
        return yaml.safe_load(f)


def center_crop(vol, crop_size):
    d,h,w = vol.shape
    z0 = (d - crop_size)//2; y0 = (h - crop_size)//2; x0 = (w - crop_size)//2
    return vol[z0:z0+crop_size, y0:y0+crop_size, x0:x0+crop_size]


def crop_by_mask(vol, mask, crop_size, margin=5):
    coords = np.array(np.where(mask > 0))
    mins = coords.min(axis=1) - margin
    maxs = coords.max(axis=1) + margin
    z0,y0,x0 = np.maximum(mins, 0).astype(int)
    z1,y1,x1 = np.minimum(maxs, vol.shape).astype(int)
    cropped = vol[z0:z1, y0:y1, x0:x1]
    cd = crop_size
    dz,dy,dx = cropped.shape
    if dz >= cd and dy >= cd and dx >= cd:
        return center_crop(cropped, cd)
    pad_z = max(0, cd - dz); pad_y = max(0, cd - dy); pad_x = max(0, cd - dx)
    pad_width = [(pad_z//2, pad_z-pad_z//2), (pad_y//2, pad_y-pad_y//2), (pad_x//2, pad_x-pad_x//2)]
    padded = np.pad(cropped, pad_width, mode='constant')
    return padded[:cd, :cd, :cd]


def load_volume_and_crop(path, crop_size, mask_img=None):
    img = nib.load(path)
    vol = img.get_fdata()
    vol = np.nan_to_num(vol)
    vol = (vol - vol.min()) / (vol.max() - vol.min() + 1e-6)
    if mask_img is not None:
        mask_data = mask_img.get_fdata() > 0
        if not np.any(mask_data):
            print("Warning: hippocampal mask did not overlap the volume, falling back to center crop.")
            vol = center_crop(vol, crop_size)
        else:
            vol = crop_by_mask(vol, mask_data, crop_size)
    else:
        vol = center_crop(vol, crop_size)
    return vol, img.affine


def overlay_heatmap(img_norm, cam_norm, color, alpha=0.5, mode='pct', pct=90, k=1.0):
    # Determine threshold based on mode
    if mode == 'otsu':
        thr = threshold_otsu(cam_norm.ravel())
    elif mode == 'std':
        thr = cam_norm.mean() + k * cam_norm.std()
    else:
        thr = np.percentile(cam_norm, pct)
    mask_cam = cam_norm > thr
    # keep largest connected component
    lab, n = label(mask_cam)
    if n > 1:
        sizes = [(lab == i).sum() for i in range(1, n+1)]
        keep = np.argmax(sizes) + 1
        mask_cam = lab == keep
    brain_mask = img_norm > 0.1
    final_mask = mask_cam & brain_mask
    overlay = np.stack([img_norm]*3, axis=-1)
    colored = np.zeros_like(overlay)
    for i in range(3): colored[..., i] = color[i]
    overlay[final_mask] = (1-alpha)*overlay[final_mask] + alpha*colored[final_mask]
    return overlay


def save_montage(vol, cam, axis, out_dir, suffix, color, alpha, mode, pct, k):
    if axis == 'axial': scores = cam.sum((1,2)); get=lambda i:(vol[i],cam[i])
    elif axis == 'coronal': scores = cam.sum((0,2)); get=lambda i:(vol[:,i],cam[:,i])
    else: scores = cam.sum((0,1)); get=lambda i:(vol[:,:,i],cam[:,:,i])
    top = np.argsort(scores)[-3:]; top.sort()
    fig, axs = plt.subplots(1,3,figsize=(12,4), frameon=False)
    for ax,i in zip(axs, top):
        img_sl, cam_sl = get(i)
        img_n = (img_sl - img_sl.min())/(img_sl.max() - img_sl.min() + 1e-6)
        cam_n = (cam_sl - cam_sl.min())/(cam_sl.max() - cam_sl.min() + 1e-6)
        ov = overlay_heatmap(img_n, cam_n, color, alpha=alpha, mode=mode, pct=pct, k=k)
        ax.imshow(ov); ax.set_title(f"{axis.capitalize()} slice {i}"); ax.axis('off')
    patches = [mpatches.Patch(color='green', label='CN'), mpatches.Patch(color='yellow', label='MCI'), mpatches.Patch(color='red', label='AD')]
    fig.legend(handles=patches, loc='lower center', ncol=3, framealpha=0.7)
    plt.tight_layout()
    fn = os.path.join(out_dir, f"{suffix}_{axis}_montage.png")
    fig.savefig(fn, bbox_inches='tight', pad_inches=0); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='3D Grad-CAM visualization with multiple threshold modes')
    parser.add_argument('--config', default='../config.yaml')
    parser.add_argument('--scan', required=True, help='Path to NIfTI scan (.nii or .nii.gz)')
    parser.add_argument('--mask', default=None, help='Optional hippocampus mask .nii.gz')
    parser.add_argument('--output_dir', default='./gradcam_out')
    parser.add_argument('--target_class', type=int, default=None, help='Class index: 0=CN,1=MCI,2=AD')
    parser.add_argument('--alpha', type=float, default=0.5, help='Overlay transparency [0-1]')
    parser.add_argument('--mode', choices=['pct','otsu','std'], default='pct', help='Threshold mode')
    parser.add_argument('--pct', type=float, default=90, help='Percentile for pct mode')
    parser.add_argument('--k', type=float, default=1.0, help='Multiplier k for mean+std mode')
    args = parser.parse_args()

    cfg = load_cfg(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    idx2label = {0:'CN',1:'MCI',2:'AD'}
    idx2color = {0:np.array([0,1,0]),1:np.array([1,1,0]),2:np.array([1,0,0])}

    # Load or fetch hippocampus mask
    mask_img = None
    if args.mask is None:
        try:
            aal = datasets.fetch_atlas_aal()
            labels = aal.labels; maps = aal.maps
            iL = labels.index('Hippocampus_L')+1; iR = labels.index('Hippocampus_R')+1
            mask_L = image.math_img(f"img == {iL}", img=maps)
            mask_R = image.math_img(f"img == {iR}", img=maps)
            mask = image.math_img("(mask_L + mask_R) > 0", mask_L=mask_L, mask_R=mask_R)
            mask_img = image.resample_to_img(mask, nib.load(args.scan), interpolation='nearest', force_resample=True, copy_header=True)
        except Exception as e:
            print("Warning: could not fetch AAL atlas (Nilearn), using center crop.", e)
    else:
        mask_img = nib.load(args.mask)

    # Load and crop
    vol, affine = load_volume_and_crop(args.scan, cfg['data']['crop_size'], mask_img)
    inp = torch.tensor(vol).unsqueeze(0).unsqueeze(0).float().to(device)

    # Load model
    model = ClassifierCNN(
        in_channels=cfg['model']['in_channels'], num_classes=cfg['model']['num_classes'],
        expansion=cfg['model']['expansion'], feature_dim=cfg['model']['feature_dim'],
        nhid=cfg['model']['nhid'], norm_type=cfg['model']['norm_type'],
        crop_size=cfg['data']['crop_size']
    ).to(device)
    state = torch.load(cfg['file_name']+'.pth', map_location=device)
    model.load_state_dict(state, strict=False); model.eval()

    # Compute GradCAM
    layer = model.conv[12]
    cam = GradCAM(model=model, target_layers=[layer])

    out = model(inp)
    pred = int(torch.softmax(out, dim=1).argmax(1).item())
    targ = args.target_class if args.target_class is not None else pred
    print(f"Predicted {idx2label[pred]}, target {idx2label[targ]}")

    grayscale_cam = cam(input_tensor=inp, targets=[ClassifierOutputTarget(targ)])[0]
    cam_tensor = torch.tensor(grayscale_cam).unsqueeze(0).unsqueeze(0).to(device)
    cam_up = F.interpolate(cam_tensor, size=vol.shape, mode='trilinear', align_corners=False)
    cam_vol = cam_up.squeeze().detach().cpu().numpy()

    # Save raw CAM
    os.makedirs(args.output_dir, exist_ok=True)
    label = idx2label[targ]
    nib.save(nib.Nifti1Image(cam_vol.astype(np.float32), affine), os.path.join(args.output_dir, f"cam_{label}.nii.gz"))

    # Save montages
    for ax in ['axial','coronal','sagittal']:
        save_montage(vol, cam_vol, ax, args.output_dir, label, idx2color[targ], alpha=args.alpha, mode=args.mode, pct=args.pct, k=args.k)

    # Combined GIF
    try:
        import imageio, io
        frames = []
        max_slices = max(vol.shape)
        for i in range(max_slices):
            fig, axes = plt.subplots(1, 3, figsize=(12, 4), frameon=True)
            for ax, (name, dim) in zip(axes, [('Axial',0),('Coronal',1),('Sagittal',2)]):
                if i < vol.shape[dim]:
                    sl = np.take(vol, i, axis=dim)
                    cm = np.take(cam_vol, i, axis=dim)
                else:
                    sl = np.zeros((vol.shape[(dim+1)%3],)); cm = np.zeros_like(sl)
                img_n = (sl - sl.min())/(sl.max() - sl.min() + 1e-6)
                cam_n = (cm - cm.min())/(cm.max() - cm.min() + 1e-6)
                ov = overlay_heatmap(img_n, cam_n, idx2color[targ], alpha=args.alpha, mode=args.mode, pct=args.pct, k=args.k)
                axes[['Axial','Coronal','Sagittal'].index(name)].imshow(ov)
                axes[['Axial','Coronal','Sagittal'].index(name)].set_title(name)
                axes[['Axial','Coronal','Sagittal'].index(name)].axis('off')
            fig.text(0.5, 0.05, f"Diagnosis: {label}", ha='center', fontsize='large', color='white')
            legend_patches = [mpatches.Patch(color='green', label='CN'), mpatches.Patch(color='yellow', label='MCI'), mpatches.Patch(color='red', label='AD')]
            fig.legend(handles=legend_patches, loc='lower center', ncol=3, framealpha=0.7)
            buf = io.BytesIO(); fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0); plt.close(fig)
            buf.seek(0); frames.append(imageio.v2.imread(buf))
        gif_path = os.path.join(args.output_dir, f"{label}_combined.gif")
        imageio.mimsave(gif_path, frames, duration=0.1)
        print(f"Saved combined GIF to {gif_path}")
    except ImportError:
        print("Install imageio to enable GIF generation")

if __name__ == '__main__':
    main()
