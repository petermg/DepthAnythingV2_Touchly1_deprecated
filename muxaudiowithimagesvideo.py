import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
from tqdm import tqdm
import subprocess

from depth_anything_v2.dpt import DepthAnythingV2

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Depth Anything V2 2D to Volumetric 3D video encoder for use in VR with the Touchly Volumetric Video Player')
    parser.add_argument('--video-path', type=str, default='inputvideo', help='default is "inputvideo"')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--img-path', type=str, default='inputpics', help='default is "inputpics"')
    parser.add_argument('--imgoutdir', type=str, default='outputpics', help='default is "outputpics"')
    parser.add_argument('--outdir', type=str, default='outputvideo', help='default is "outputvideo"')
    parser.add_argument('--height', type=int, default=518, help='default is 518. Must be used in conjunction with "--custom-height"')
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'], help='default is vitl')
    parser.add_argument('--video-bitrate', type=str, default='0k', help='Set the video bitrate when using --ffmpeg. Default value is "0k".')
    parser.add_argument('--audio-bitrate', type=str, default='0k', help='Set the audio bitrate when using --ffmpeg. Default value is "0k".')
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', help='Only output the depthmap prediction')
    parser.add_argument('--ffmpeg', dest='ffmpeg', action='store_true', help='Encode using FFMPEG instead of cv2.VideoWriter_fourcc. This first generates temp image files.')
    parser.add_argument('--color', dest='color', action='store_true', help='Apply colorful palette')
    parser.add_argument('--dualencode', dest='dualencode', action='store_true', help='Encodes first with the VideoWriter then with ffmpeg. You can use the ffmpeg flags for this without needing to use "--ffmpeg".')
    parser.add_argument('--bit16', dest='bit16', action='store_true', help='Used with --ffmpeg, creates 16bit grayscale png files as intermediates before encoding video file. Does not work with --color option.')
    parser.add_argument("--precision", type=str, default='fp64', choices= ['fp64', 'fp32', 'fp16'], help='Does this even really do anything?')
    parser.add_argument('--codec', type=str, default='HFYU', help='Sets the video codec to be used by the default cv2.VideoWriter_fourcc video encoder. Default is "HFYU". Use --showcodecs to see what codecs are available. Specify --extension to specify which container if other than mkv')
    parser.add_argument('--pix-fmt', type=str, default='yuv420p', help='Sets the video input pixel format. Default is "yuv420p". To be used with the --ffmpeg option.')
    parser.add_argument('--ffmpeg-codec', type=str, default='huffyuv', help='Sets the ffmpeg video codec for the ffmpeg output. To be used in conjunction with the --ffmpeg option.')
    parser.add_argument('--extension', type=str, default='mkv', help='Sets the file extension/container. Default is "mkv". Note, different containers support different codecs.')
    parser.add_argument('--ffmpeg-extension', type=str, default='mkv', help='Sets the file extension/container for the final output by ffmpeg. Default is "mkv". Note, different containers support different codecs.')
    parser.add_argument('--showcodecs', action='store_true', help='Shows available video codecs for the cv2.VideoWriter_fourcc encoder to use. Best used in conjunction with "--extension" to specify a format like mp4, avi, mkv, etc. to see supported codecs for respective file formats.')
    parser.add_argument('--images', action='store_true', help='Create depthmaps from image files stored in the images input folder')
    parser.add_argument('--imagetovideo', action='store_true', help='Creates 30 second clips in the Touchly1 format from input images. MUST be used in conjunction with --images option.')
    parser.add_argument('--custom-height', action='store_true', help='Specify input height, generally used to DECREASE hight in case of Out Of Memory errors.')
    parser.add_argument('--audio-codec', type=str, default='copy', help='Specify the audio codec to be used when ffmpeg adds the audio. By default this is set to "copy" which just remuxes the audio from the original video file without any re-encoding. The --ffmpeg option is NOT required for this.')
    #parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'mps', 'cpu'], help='default is cuda')
    args = parser.parse_args()
    #DEVICE = args.device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    if args.images:
        depth_anything = DepthAnythingV2(**model_configs[args.encoder])
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()
        
        if os.path.isfile(args.img_path):
            if args.img_path.endswith('txt'):
                with open(args.img_path, 'r') as f:
                    filenames = f.read().splitlines()
            else:
                filenames = [args.img_path]
        else:
            filenames = glob.glob(os.path.join(args.img_path, '**/*'), recursive=True)
        
        os.makedirs(args.imgoutdir, exist_ok=True)
        os.makedirs(args.outdir, exist_ok=True)
        
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
            raw_image = cv2.imread(filename)
            raw_image16 = (raw_image.astype(np.uint16) * 255)
            
            depth = depth_anything.infer_image(raw_image, args.input_size)
            
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
            depth = depth.cpu().numpy().astype(np.uint16)
            #depth = depth.astype(np.uint16)
            
            if args.color:
                depth = (cmap(depth)[:, :, :3] * 65536)[:, :, ::-1].astype(np.uint16)
            else:
                depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)

                
            topimage = raw_image16
            bottomimage = depth
            
            output_img_path = os.path.join(args.imgoutdir, os.path.splitext(os.path.basename(filename))[0] + '.png')
            #output_img_path = os.path.join(args.imgoutdir, os.path.splitext(os.path.basename(filename))[0])
            
            if args.pred_only:
                cv2.imwrite(output_img_path, depth)
            else:
                #split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint16) * 65536
                combined_result = cv2.vconcat([topimage, bottomimage])
                
                cv2.imwrite(output_img_path, combined_result)
            if args.imagetovideo:
                # Create a 30-second video from the saved PNG file
                output_video_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_pic_Touchly1.' + args.ffmpeg_extension)
                cmd = [
                    'ffmpeg',
                    '-loop', '1',
                    '-framerate', '1',
                    '-i', output_img_path,
                    '-c:v', args.ffmpeg_codec,
                    '-t', '30',
                    '-pix_fmt', args.pix_fmt,
                    '-b:v', args.video_bitrate,
                   # '-vf', f'scale={raw_image.shape[1]}:{raw_image.shape[0]}',
                    '-y', output_video_path
                ]
                subprocess.run(cmd)

    
    else:
        depth_anything = DepthAnythingV2(**model_configs[args.encoder])
        depth_anything.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{args.encoder}.pth', map_location='cpu'))
        depth_anything = depth_anything.to(DEVICE).eval()

        if os.path.isfile(args.video_path):
            if args.video_path.endswith('txt'):
                with open(args.video_path, 'r') as f:
                    lines = f.read().splitlines()
            else:
                filenames = [args.video_path]
        else:
            filenames = glob.glob(os.path.join(args.video_path, '**/*'), recursive=True)
        
        os.makedirs(args.outdir, exist_ok=True)
        
        margin_width = 0
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        for k, filename in enumerate(filenames):
            print(f'Progress {k+1}/{len(filenames)}: {filename}')
                    
            raw_video = cv2.VideoCapture(filename)
            frame_width, frame_height = int(raw_video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(raw_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frame_rate = int(raw_video.get(cv2.CAP_PROP_FPS))

            if args.custom_height:
                newHeight = round(args.height / 14) * 14
                
            else:
                newHeight = round(frame_height / 14) * 14
                
            aspectRatio = frame_width / frame_height
            newWidth = round(newHeight * aspectRatio / 14) * 14
            newWidth = (newWidth // 14) * 14

            if args.pred_only: 
                output_width = frame_width
                output_height = frame_height
            else: 
                output_height = frame_height * 2
                output_width = frame_width
            

            output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_Touchly1.' + args.extension)
            
            #finaloutput_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_Touchly1.' + args.ffmpeg_extension)
            
            if args.showcodecs:
                print(cv2.VideoWriter(args.outdir + '/dummy.' + args.extension, -1, frame_rate, (output_width, output_height)))
                break        
            
           
            if args.ffmpeg:
                output_basename = os.path.splitext(os.path.basename(filename))[0] + '_Touchly1'
                output_path = os.path.join(args.outdir, output_basename + '.' + args.ffmpeg_extension)
                frames_dir = os.path.join(args.outdir, output_basename + '_frames')
                os.makedirs(frames_dir, exist_ok=True)
                
                totalFrameCount = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
               
                for frame_idx in tqdm(range(totalFrameCount)):
                    ret, raw_frame = raw_video.read()
                    if not ret:
                        break
                    
                    if args.bit16:
                        temppics = 'png'
                        raw_frame16 = (raw_frame.astype(np.uint16) * 255)
                        depth = depth_anything.infer_image(raw_frame, precision=args.precision, newHeight=newHeight, newWidth=newWidth)
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 65536.0
                        depth = depth.cpu().numpy().astype(np.uint16)
                    
                    else:
                        temppics = 'jpg'
                        raw_frame16 = raw_frame
                        depth = depth_anything.infer_image(raw_frame, precision=args.precision, newHeight=newHeight, newWidth=newWidth)
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        depth = depth.cpu().numpy().astype(np.uint8)
                    
                    if args.color:
                        raw_frame16 = raw_frame
                        depth = depth_anything.infer_image(raw_frame, precision=args.precision, newHeight=newHeight, newWidth=newWidth)
                        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                        depth = depth.cpu().numpy().astype(np.uint8)
                        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)


                    else:
                        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                        
                    if args.pred_only:
                        frame_to_save = depth
                    else:
                        #split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                        frame_to_save = cv2.vconcat([raw_frame16, depth])
                    
                    frame_filename = os.path.join(frames_dir, f'frame_{frame_idx:06d}.' + temppics)
                    cv2.imwrite(frame_filename, frame_to_save)
                
                raw_video.release()
                
                # Encode video using ffmpeg
                ffmpeg_cmd = [
                    'ffmpeg', '-framerate', str(frame_rate), '-i',
                    os.path.join(frames_dir, 'frame_%06d.' + temppics),
                    '-c:v', args.ffmpeg_codec, '-pix_fmt', args.pix_fmt, '-b:v', args.video_bitrate, output_path
                ]
                subprocess.run(ffmpeg_cmd)
                
                # Mux audio into the video using ffmpeg
                temp_output_path = os.path.join(args.outdir, output_basename + '_temp.' + args.ffmpeg_extension)
                mux_command = [
                     'ffmpeg', '-i', output_path, '-i', filename, '-c:v', 'copy', '-c:a', args.audio_codec, '-b:a:', args.audio_bitrate, '-map', '0:v:0', '-map', '1:a:0', temp_output_path
                ]
                subprocess.run(mux_command)
                os.replace(temp_output_path, output_path)
                
                # Clean up frames directory
                for file in os.listdir(frames_dir):
                    os.remove(os.path.join(frames_dir, file))
                os.rmdir(frames_dir)
            
            else:
                out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*args.codec), frame_rate, (output_width, output_height))
                
                totalFrameCount = int(raw_video.get(cv2.CAP_PROP_FRAME_COUNT))
               
                for _ in tqdm(range(totalFrameCount)):
                    ret, raw_frame = raw_video.read()
                    if not ret:
                        break
                    
                    depth = depth_anything.infer_image(raw_frame, precision=args.precision, newHeight=newHeight, newWidth=newWidth)
                    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                    depth = depth.cpu().numpy().astype(np.uint8)
                    
                    if args.color:
                        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
                    else:
                        depth = np.repeat(depth[..., np.newaxis], 3, axis=-1)
                        
                    if args.pred_only:
                        out.write(depth)
                    else:
                        split_region = np.ones((frame_height, margin_width, 3), dtype=np.uint8) * 255
                        combined_frame = cv2.vconcat([raw_frame, depth])
                        
                        out.write(combined_frame)
                
                raw_video.release()
                out.release()
                
                # Muxing audio into the video using ffmpeg
                temp_output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + '_Touchly1_temp.' + args.extension)
                ffmpeg_output_path = os.path.join(args.outdir, os.path.splitext(os.path.basename(filename))[0] + 'ffmpeg_Touchly1.' + args.extension)
                mux_command = [
                    'ffmpeg', '-i', output_path, '-i', filename, '-c:v', 'copy', '-c:a', args.audio_codec, '-map', '0:v:0', '-map', '1:a:0', temp_output_path
                ]
                ffmpegencode_command = [
                    'ffmpeg', '-i', output_path, '-i', filename, '-c:v', args.ffmpeg_codec, '-b:v', args.video_bitrate, '-c:a', args.audio_codec, '-map', '0:v:0', '-map', '1:a:0', ffmpeg_output_path
                ]                
                subprocess.run(mux_command)
                if args.dualencode:
                    subprocess.run(ffmpegencode_command)
                os.replace(temp_output_path, output_path)
                #os.replace(temp_output_path, finaloutput_path)
