import argparse
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import model.lanenet as lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, get_color

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='path to input image for prediction')
    parser.add_argument('--ckpt_path', type=str, required=True, help='path to model checkpoint (.pth)')
    parser.add_argument('--arch', type=str, default='fcn', help='network architecture type (default: FCN)')
    parser.add_argument('--dual_decoder', action='store_true', help='use separate decoders for two branches')
    parser.add_argument('--show', action='store_true', help='show the prediction visualization')
    parser.add_argument('--save_img', type=str, help='path to save the output visualization')
    return parser.parse_args()

def load_model(ckpt_path, arch, dual_decoder, device):
    if 'fcn' in arch.lower():
        arch = 'lanenet.LaneNet_FCN_Res'
    elif 'enet' in arch.lower():
        arch = 'lanenet.LaneNet_ENet'
    elif 'icnet' in arch.lower():
        arch = 'lanenet.LaneNet_ICNet'
    
    arch = arch + '_1E2D' if dual_decoder else arch + '_1E1D'
    print('Architecture:', arch)
    
    net = eval(arch)()
    net = nn.DataParallel(net)
    net.to(device)
    
    checkpoint = torch.load(ckpt_path)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)
    net.eval()
    return net

def preprocess_image(image_path, VGG_MEAN):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (512, 256))
    image_rgb = image_rgb.astype(np.float32) - VGG_MEAN
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0) / 255.0
    return image_tensor

def predict_single_image(image_path, model, device, VGG_MEAN, show=False, save_path=None):
    input_tensor = preprocess_image(image_path, VGG_MEAN).to(device)
    
    with torch.no_grad():
        embeddings, logit = model(input_tensor)
        pred_bin = torch.argmax(logit, dim=1, keepdim=True).cpu().numpy().squeeze(0)
        pred_inst = cluster_embed(embeddings, pred_bin, band_width=0.5)[0]

        h, w = input_tensor.shape[2:]
        rgb = cv2.imread(image_path)
        rgb = cv2.resize(rgb, (w, h))
        pred_bin_rgb = (pred_bin * 255).astype(np.uint8)
        pred_inst_rgb = np.zeros_like(rgb)

        for i in np.unique(pred_inst):
            if i == 0:
                continue
            index = np.where(pred_inst == i)
            pred_inst_rgb[index] = get_color(i)

        overlay = cv2.addWeighted(rgb, 0.5, pred_inst_rgb, 0.5, 0)
        if save_path:
            cv2.imwrite(save_path, overlay)
            print(f"Saved visualization to {save_path}")
        
        if show:
            cv2.imshow("Lane Prediction", overlay)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    args = init_args()
    VGG_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(args.ckpt_path, args.arch, args.dual_decoder, device)
    predict_single_image(args.image_path, model, device, VGG_MEAN, show=args.show, save_path=args.save_img)