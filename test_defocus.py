import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import os
import random
import numpy as np
from Utils import GaussPSF
from PIL import Image
from time import time

from skimage import metrics


import util.pytorch_ssim as pytorch_ssim

generator = GaussPSF(51, 0, 200, pixel_size=0.0044, scale=3)
# generator = GaussPSF(601, 0, 200)

generator = torch.nn.DataParallel(generator)
generator = generator.cuda()

focal_depth = torch.Tensor([100] * 1).float().cuda()


aperture = torch.Tensor([2.5] * 1).float().cuda()
# aperture = torch.Tensor([2.5e-3] * 1).float().cuda()
focal_length = torch.Tensor([1] * 1).float().cuda()


image =np.array(Image.open(r"C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\datasets\Simcol3D_480_mirrored\aif\val\SyntheticColon_II_Frames_B11_FrameBuffer_0257.png"))
gt = Image.open(r"C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\datasets\Simcol3D_480_mirrored\depth\val\SyntheticColon_II_Frames_B11_Depth_0257.png")
gt = np.array(gt)

# Convert to int16
image_int = torch.from_numpy(image).float()
gt_int = torch.from_numpy(gt / 256 / 255 * 200).float()

image_int = image_int.permute(2, 0, 1)
image_int = image_int.reshape(1, image_int.shape[0], image_int.shape[1], image_int.shape[2])
gt_int = gt_int.reshape(1,1,gt_int.shape[0], gt_int.shape[1])

start = time()

refocused = generator( image_int,  gt_int, focal_depth, aperture, focal_length)
print("refocalizarea a durat: ", time()-start, "s")


start = time()
refocused = refocused[0].permute(1,2,0).cpu()
print(start - time())
refocused = refocused.numpy()
# Create PIL Image object
pil_image = Image.fromarray(np.uint8(refocused))

# Save image as PNG
pil_image.save('test_blur/SyntheticColon_II_Frames_B11_FrameBuffer_0257.png')
ssim_value = metrics.structural_similarity(np.uint8(refocused), image, data_range=255, channel_axis=2)
print(ssim_value)