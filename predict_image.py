import os
import argparse
import torch
import cv2
import numpy as np
import ujson as json
from model import lanenet
from model.utils import cluster_embed, fit_lanes, sample_from_curve, generate_json_entry, get_color
from torchvision import transforms

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, help='path to the input image')
    parser.add_argument('--ckpt_path', type=str, help='path to model checkpoint (.pth)')
    parser.add_argument('--save_dir', type=str, help='directory to save results')
    parser.add_argument('--arch', type=str, default='fcn', help='network architecture type(default: FCN)')
    parser.add_argument('--dual_decoder', action='store_true', help='use separate decoders for two branches')
    parser.add_argument('--show', action='store_true', help='display the results')
    parser.add_argument('--save_img', action='store_true', help='save visualization images')
    return parser.parse_args()

def preprocess_image(image_path, size=(512, 288), mean=np.array([103.939, 116.779, 123.68])):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"No image found at {image_path}")
    image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image -= mean
    image = np.transpose(image, (2, 0, 1))
    return torch.from_numpy(image).float() / 255

def main():
    args = init_args()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model initialization
    arch = 'lanenet.LaneNet_FCN_Res'
    if 'fcn' in args.arch.lower():
        arch = 'lanenet.LaneNet_FCN_Res'
    elif 'enet' in args.arch.lower():
        arch = 'lanenet.LaneNet_ENet'
    elif 'icnet' in args.arch.lower():
        arch = 'lanenet.LaneNet_ICNet'

    arch = arch + '_1E2D' if args.dual_decoder else arch + '_1E1D'
    net = eval(arch)()
    net = torch.nn.DataParallel(net)
    net.to(device)
    net.eval()

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    net.load_state_dict(checkpoint['model_state_dict'], strict=True)

    input_tensor = preprocess_image(args.image_path)
    input_batch = input_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        embeddings, logit = net(input_batch)
        pred_bin_batch = torch.argmax(logit, dim=1, keepdim=True)

    if args.show or args.save_img:
        input_rgb = ((input_tensor.cpu().numpy().transpose(1, 2, 0) * 255) + np.array([103.939, 116.779, 123.68])).astype(np.uint8)
        pred_bin_rgb = (pred_bin_batch[0].repeat(3, 1, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        combined_img = cv2.addWeighted(input_rgb, 1, pred_bin_rgb, 0.5, 0)

        if args.show:
            cv2.imshow('Predicted Binary Image', pred_bin_rgb)
            cv2.imshow('Overlay Input with Prediction', combined_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if args.save_img:
            cv2.imwrite(os.path.join(args.save_dir, 'input.jpg'), input_rgb)
            cv2.imwrite(os.path.join(args.save_dir, 'binary_prediction.jpg'), pred_bin_rgb)
            cv2.imwrite(os.path.join(args.save_dir, 'overlay.jpg'), combined_img)

if __name__ == '__main__':
    main()

