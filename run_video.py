import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from tqdm import tqdm
from depth_anything_v2.dpt import DepthAnythingV2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2')
    
    parser.add_argument('--video-path', type=str)
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--outdir', type=str, default='./vis_video_depth')
    
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', help='do not apply colorful palette')
    parser.add_argument("--precision", type=str, default='fp32', choices= ['fp32', 'fp16'])
    
    args = parser.parse_args()
    
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
    depth_anything = depth_anything.to(DEVICE).eval()

    if args.precision == 'fp16' and DEVICE == 'cuda':
        depth_anything = depth_anything.half()
    else:
        print('FP16 precision is only available on CUDA devices. Using FP32 instead.')
        args.precision = 'fp32'
        depth_anything = depth_anything.float()

    
    if os.path.isfile(args.video_path):
        if args.video_path.endswith('txt'):
            with open(args.video_path, 'r') as f:
                lines = f.read().splitlines()
        else:
            filenames = [args.video_path]
    else:
        filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
    
    os.makedirs(args.outdir, exist_ok=True)
    
    margin_width = 50
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')
    
    for k, filename in enumerate(filenames):
        print(f'Progress {k+1}/{len(filenames)}: {filename}')
        
        
        raw_video = cv2.VideoCapture(filename)
        frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

        aspectRatio = frame_width / frame_height
        # Fix height at 518 and adjust width
        newHeight = 518
        newWidth = round(newHeight * aspectRatio / 14) * 14
        # Ensure newWidth is a multiple of 14
        newWidth = (newWidth // 14) * 14

        if args.pred_only: 
            output_width = frame_width
        else: 
            output_width = frame_width * 2 + margin_width
        
        output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '.mp4')
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (output_width, frame_height))
        
        totalFrameCount = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        for _ in tqdm(range(totalFrameCount)):
            ret, raw_frame = raw_video.read()
            if not ret:
                break
            
            # Comes back with a Torch float 16 / 32 based on precision desired precision
            depth = depth_anything.infer_image(raw_frame, args.input_size, precision=args.precision, newHeight=newHeight, newWidth=newWidth)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0

            # Preferably don't convert to uint8 here but only on the final output, to do.
            depth = depth.cpu().numpy().astype(np.uint8)
            
            if args.grayscale:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
            else:
                depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
            if args.pred_only:
                out.write(depth)
            else:
                split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                combined_frame = cv2.hconcat([raw_frame, split_region, depth])
                
                out.write(combined_frame)
        
        raw_video.release()
        out.release()
