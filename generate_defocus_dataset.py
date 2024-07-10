import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tqdm import tqdm
import os
import numpy as np
from Utils import GaussPSF
from PIL import Image
from time import time

generator = GaussPSF(51, 0, 200, pixel_size=0.0044, scale=3)

generator = torch.nn.DataParallel(generator)
generator = generator.cuda()

focal_depth = torch.Tensor([10] * 1).float().cuda()


N = torch.Tensor([2.5] * 1).float().cuda()
focal_length = torch.Tensor([1] * 1).float().cuda()

test_folder = r"C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\datasets\Simcol3D_480_mirrored\aif\test"
depth_folder = r"C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\datasets\Simcol3D_480_mirrored\depth\test"
dest = r"C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\datasets\Simcol3D_480_mirrored\rgb_10mm\test"

test_files = sorted(os.listdir(test_folder), reverse=True)
depth_files = sorted(os.listdir(depth_folder), reverse=True)

for file_path, depth_path in tqdm(zip(test_files, depth_files), total=min(len(test_files), len(depth_files))):
    # Load the image and depth using PIL and convert to numpy arrays
    image = np.array(Image.open(os.path.join(test_folder, file_path)))
    gt = np.array(Image.open(os.path.join(depth_folder, depth_path)))

    # Normalize the ground truth and scale
    gt_scaled = gt / 256 / 255 * 200

    # Permute the axes in numpy (change channel from last to first) and add batch dimension
    image = np.transpose(image, (2, 0, 1))[np.newaxis, ...]  # Add batch dimension as first dimension
    gt_scaled = gt_scaled[np.newaxis, np.newaxis, ...]  # Add batch and channel dimensions to gt

    # Convert numpy arrays to torch tensors
    image_int = torch.from_numpy(image).float()
    gt_int = torch.from_numpy(gt_scaled).float()

    # Process with the model
    refocused = generator(image_int, gt_int, focal_depth, N, focal_length)
    # Convert back to numpy for image saving
    start = time()
    refocused = refocused[0]
    refocused = refocused.permute(1, 2, 0)
    refocused = refocused.to('cpu')
    refocused = refocused.numpy()
    pil_image = Image.fromarray(np.uint8(refocused))
    pil_image.save(os.path.join(dest, file_path))
