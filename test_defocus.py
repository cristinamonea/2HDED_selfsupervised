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

generator = GaussPSF(51, 1e-3, 10000, pixel_size=0.006, scale=1)
# generator = GaussPSF(601, 0, 200)

generator = torch.nn.DataParallel(generator)
generator = generator.cuda()

focal_depth = torch.Tensor([1600] * 1).float().cuda()


N = torch.Tensor([2.8] * 1).float().cuda()
# aperture = torch.Tensor([2.5e-3] * 1).float().cuda()
focal_length = torch.Tensor([14.0] * 1).float().cuda()


image =np.array(Image.open(r"C:\Users\ubuntu\Desktop\Cristina\DFD\iDFD-main\Data\Raw\inpainting\All_In_Focus\Apartment1_0.JPG"))
gt = Image.open(r"C:\Users\ubuntu\Desktop\Cristina\DFD\iDFD-main\Data\Raw\inpainting\Depth\Apartment1_0.png")
gt = np.array(gt, dtype=float)

# Convert to int16
image_int = torch.from_numpy(image).float()
gt_int = torch.from_numpy(gt).float()

image_int = image_int.permute(2, 0, 1)
image_int = image_int.reshape(1, image_int.shape[0], image_int.shape[1], image_int.shape[2])
gt_int = gt_int.reshape(1,1,gt_int.shape[0], gt_int.shape[1])

start = time()

refocused = generator( image_int,  gt_int, focal_depth, N, focal_length)
print("refocalizarea a durat: ", time()-start, "s")


start = time()
refocused = refocused[0].permute(1,2,0).cpu()
print(start - time())
refocused = refocused.numpy()
# Create PIL Image object
pil_image = Image.fromarray(np.uint8(refocused))

# Save image as PNG
pil_image.save('test_blur/ap.png')
ssim_value = metrics.structural_similarity(np.uint8(refocused), image, data_range=255, channel_axis=2)
print(ssim_value)