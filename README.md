This is a fork of https://github.com/NevermindNilas/Faster-Depth-Anything-V2 to which I have made the following changes:

### This fork has been modified to output 16bit precision depth map png files instead of 8bit. This allows for better 3D model creation without a "stair-stepping" look to the model.

### This fork also outputs video files in the "Touchly1" format, which has the depth map on the bottom and the original video on the top, vertiacally stacking the output. Just append "_Touchly1" to the name of the video file to play it back in the Touchly player in 6DoF 3D!

You can process video files by running ["run_video.py"](#running-script-on-videos)

You can download the Touchly Volumetric media player for the standalone Quest (Free) here: https://www.meta.com/experiences/5564815066942737/

You can download the Touchly Volumetric media player for PCVR on Steam ($6.99) here: https://store.steampowered.com/app/2480680/Touchly_Volumetric_VR_Video_Player/

The main Touchly website is here: https://touchly.app/

You can also download the official Touchly renderer / encoder app here: https://touchly.app/renderer/ There is a free version and a Pro version ($10). The Touchly Renderer both free and paid version can create volumetric videos from 2D videos and also from 3D SBS videos and 3D VR180 videos.

Only [run.py (image processing)](#running-script-on-images) has been modified to output 16 bit precision images.

<div align="center">
<h1>Depth Anything V2</h1>

[**Lihe Yang**](https://liheyoung.github.io/)<sup>1</sup> · [**Bingyi Kang**](https://bingykang.github.io/)<sup>2&dagger;</sup> · [**Zilong Huang**](http://speedinghzl.github.io/)<sup>2</sup>
<br>
[**Zhen Zhao**](http://zhaozhen.me/) · [**Xiaogang Xu**](https://xiaogang00.github.io/) · [**Jiashi Feng**](https://sites.google.com/site/jshfeng/)<sup>2</sup> · [**Hengshuang Zhao**](https://hszhao.github.io/)<sup>1*</sup>

<sup>1</sup>HKU&emsp;&emsp;&emsp;<sup>2</sup>TikTok
<br>
&dagger;project lead&emsp;*corresponding author

<a href="https://arxiv.org/abs/2406.09414"><img src='https://img.shields.io/badge/arXiv-Depth Anything V2-red' alt='Paper PDF'></a>
<a href='https://depth-anything-v2.github.io'><img src='https://img.shields.io/badge/Project_Page-Depth Anything V2-green' alt='Project Page'></a>
<a href='https://huggingface.co/spaces/depth-anything/Depth-Anything-V2'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>
<a href='https://huggingface.co/datasets/depth-anything/DA-2K'><img src='https://img.shields.io/badge/Benchmark-DA--2K-yellow' alt='Benchmark'></a>
</div>

This work presents Depth Anything V2. It significantly outperforms [V1](https://github.com/LiheYoung/Depth-Anything) in fine-grained details and robustness. Compared with SD-based models, it enjoys faster inference speed, fewer parameters, and higher depth accuracy.

![teaser](assets/teaser.png)


## News

- **2024-07-06:** Depth Anything V2 is supported in [Transformers](https://github.com/huggingface/transformers/). See the [instructions](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for convenient usage.
- **2024-06-25:** Depth Anything is integrated into [Apple Core ML Models](https://developer.apple.com/machine-learning/models/). See the instructions ([V1](https://huggingface.co/apple/coreml-depth-anything-small), [V2](https://huggingface.co/apple/coreml-depth-anything-v2-small)) for usage.
- **2024-06-22:** We release [smaller metric depth models](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth#pre-trained-models) based on Depth-Anything-V2-Small and Base.
- **2024-06-20:** Our repository and project page are flagged by GitHub and removed from the public for 6 days. Sorry for the inconvenience.
- **2024-06-14:** Paper, project page, code, models, demo, and benchmark are all released.


## Pre-trained Models

We provide **four models** of varying scales for robust relative depth estimation:

| Model | Params | Checkpoint |
|:-|-:|:-:|
| Depth-Anything-V2-Small | 24.8M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth?download=true) |
| Depth-Anything-V2-Base | 97.5M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth?download=true) |
| Depth-Anything-V2-Large | 335.3M | [Download](https://huggingface.co/depth-anything/Depth-Anything-V2-Large/resolve/main/depth_anything_v2_vitl.pth?download=true) |
| Depth-Anything-V2-Giant | 1.3B | Coming soon |


## Usage

### Prepraration

```bash
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2
pip install -r requirements.txt
```

Download the checkpoints listed [here](#pre-trained-models) and put them under the `checkpoints` directory.

### Use our models
```python
import cv2
import torch

from depth_anything_v2.dpt import DepthAnythingV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('your/image/path')
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
```

If you do not want to clone this repository, you can also load our models through [Transformers](https://github.com/huggingface/transformers/). Below is a simple code snippet. Please refer to the [official page](https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2) for more details.

- Note 1: Make sure you can connect to Hugging Face and have installed the latest Transformers.
- Note 2: Due to the [upsampling difference](https://github.com/huggingface/transformers/pull/31522#issuecomment-2184123463) between OpenCV (we used) and Pillow (HF used), predictions may differ slightly. So you are more recommended to use our models through the way introduced above.
```python
from transformers import pipeline
from PIL import Image

pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")
image = Image.open('your/image/path')
depth = pipe(image)["depth"]
```

### Running script on *images*

```bash
python run.py \
  --encoder <vits | vitb | vitl | vitg> \
  --img-path <path> --outdir <outdir> \
  [--input-size <size>] [--pred-only] [--color]
```
Options:
- `--img-path`: You can either 1) point it to an image directory storing all interested images, 2) point it to a single image, or 3) point it to a text file storing all image paths.
- `--input-size` (optional): By default, we use input size `518` for model inference. ***You can increase the size for even more fine-grained results.***
- `--pred-only` (optional): Only save the predicted depth map, without raw image.
- `--color` (optional): Save the color depth map, instead of gray.

For example:
```bash
python run.py --encoder vitl --img-path assets/examples --outdir depth_vis
```

### Running script on *videos*

```bash
python run_video.py \
  --encoder <vits | vitb | vitl | vitg> \
  [--video-path <path to input video file(s)> (default is inputvideo)]  [--outdir <path to output video files> (default is outputvideo)]
  [--custom-height --height <size> (default is 518)] [--pred-only (optional as this only produces a depthmap video NOT a Touchly1 formatted video.)] [--color (DO NOT USE THIS OPTION if you want to create a Touchly1 formatted video)] [--codec <fourcc codec> (default is HFYU)] [--extension <video file container extension> (default is mkv)]
```
By default the encoder uses vitl. By default using "--custom-height" sets the new height to 518. If you add "--height" you can specify whatever height you want your input video to be resized to. The width will automatically be resized to maintain the same aspect ratio. This is generally used for the purpose of LOWERING the input size of the video in cases of OUT OF MEMORY errors when processing.
Some command line examples for processing videos to the Touchly1 format:
```bash
python run_video.py --custom-height --height 256
```
The above command creates a video with the original video on top, the depthmap video on the bottom and a reduced input height set to 256. Since no directories are specified it looks for the input files in the default location of 'assets/inputvideo' and saves the output to the default location of 'outputvideo'.
Another example:
```bash
python run_video.py
```
The above example processes the input video(s) from the default folder of 'inputvideo', does not resize the video for processing but uses the same resolution to create the depthmap and saves the video output to the default folder of 'outputvideo'. This creates a vertically stacked output of the original videon on top and the depthmap on the bottom, which is the Touchly1 volumetric video format.
```bash
python run_video.py --custom-height --height 256 --extension avi --codec mjpg
```
The above command first resizes the input video to 256 heigh with a width maintaining the original aspect ratio, it saves it as an AVI file, and encodes it using the MJPG codec.

You can also determine what codecs are supported for what containers / extensions by using the following command:
```bash
python run_video.py --showcodecs --extension avi
```
The above command will show you which codecs are available to be used with the avi format/extension/container. By default it will show you the codecs avialable for the mkv format/extension/container if a format/extension/container is not specified by use of the "--extension" option.


***Our larger model has better temporal consistency on videos.***

### Gradio demo

To use our gradio demo locally:

```bash
python app.py
```

You can also try our [online demo](https://huggingface.co/spaces/Depth-Anything/Depth-Anything-V2).

***Note: Compared to V1, we have made a minor modification to the DINOv2-DPT architecture (originating from this [issue](https://github.com/LiheYoung/Depth-Anything/issues/81)).*** In V1, we *unintentionally* used features from the last four layers of DINOv2 for decoding. In V2, we use [intermediate features](https://github.com/DepthAnything/Depth-Anything-V2/blob/2cbc36a8ce2cec41d38ee51153f112e87c8e42d8/depth_anything_v2/dpt.py#L164-L169) instead. Although this modification did not improve details or accuracy, we decided to follow this common practice.


## Fine-tuned to Metric Depth Estimation

Please refer to [metric depth estimation](./metric_depth).


## DA-2K Evaluation Benchmark

Please refer to [DA-2K benchmark](./DA-2K.md).


## Community Support

**We sincerely appreciate all the community support for our Depth Anything series. Thank you a lot!**

- Apple Core ML:
    - https://developer.apple.com/machine-learning/models
    - https://huggingface.co/apple/coreml-depth-anything-v2-small
    - https://huggingface.co/apple/coreml-depth-anything-small
- Transformers:
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything_v2
    - https://huggingface.co/docs/transformers/main/en/model_doc/depth_anything
- TensorRT:
    - https://github.com/spacewalk01/depth-anything-tensorrt
    - https://github.com/zhujiajian98/Depth-Anythingv2-TensorRT-python
- ONNX: https://github.com/fabio-sim/Depth-Anything-ONNX
- ComfyUI: https://github.com/kijai/ComfyUI-DepthAnythingV2
- Transformers.js (real-time depth in web): https://huggingface.co/spaces/Xenova/webgpu-realtime-depth-estimation
- Android:
  - https://github.com/shubham0204/Depth-Anything-Android
  - https://github.com/FeiGeChuanShu/ncnn-android-depth_anything


## Acknowledgement

We are sincerely grateful to the awesome Hugging Face team ([@Pedro Cuenca](https://huggingface.co/pcuenq), [@Niels Rogge](https://huggingface.co/nielsr), [@Merve Noyan](https://huggingface.co/merve), [@Amy Roberts](https://huggingface.co/amyeroberts), et al.) for their huge efforts in supporting our models in Transformers and Apple Core ML.

We also thank the [DINOv2](https://github.com/facebookresearch/dinov2) team for contributing such impressive models to our community.


## LICENSE

Depth-Anything-V2-Small model is under the Apache-2.0 license. Depth-Anything-V2-Base/Large/Giant models are under the CC-BY-NC-4.0 license.


## Citation

If you find this project useful, please consider citing:

```bibtex
@article{depth_anything_v2,
  title={Depth Anything V2},
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Zhao, Zhen and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  journal={arXiv:2406.09414},
  year={2024}
}

@inproceedings{depth_anything_v1,
  title={Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data}, 
  author={Yang, Lihe and Kang, Bingyi and Huang, Zilong and Xu, Xiaogang and Feng, Jiashi and Zhao, Hengshuang},
  booktitle={CVPR},
  year={2024}
}
```
