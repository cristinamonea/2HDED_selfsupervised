# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 13:43:50 2024

@author: ubuntu
"""
import numpy as np
# import sobel
import os
from PIL import Image
from tqdm import tqdm
    
from skimage import metrics

def test(aif_path, pred_path):
    # PSNR ERROR CALCULATION
     # We load the gt and pred images
    avg_psnr = 0.0
    avg_ssim = 0.0
    

    gt_list = [file for file in os.listdir(aif_path) if file.endswith('.npy')]
    pred_list = [file for file in os.listdir(pred_path) if file.endswith('.npy')]
    
    if (len(gt_list) != len(pred_list)):
        print("not good")
        return

    prediction = []
    for i in pred_list:
        img_temp = np.load(os.path.join(pred_path,i))[0]
        # img_temp = np.moveaxis(img_temp, source=2, destination=4)
        #if pred
        img_temp = img_temp[0].transpose(1, 2, 0)
        #if input
        # img_temp = img_temp.transpose(1, 2, 0)
        # img_temp = (img_temp + 1)/2
        prediction.append(img_temp)
        ####Save as PNG
        # prediction_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # plt.imsave('D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/Results/Images/aif_pred/%s.png' %i, prediction_temp)
       
    target = []

    for i in gt_list:
        img_temp = np.load(os.path.join(aif_path, i))[0,0]
        # img_temp = np.moveaxis(img_temp, source=2, destination=4)
        img_temp =  img_temp.transpose(1, 2, 0)
        # img_temp = (img_temp + 1)/2
        target.append(img_temp)
        ####Save as PNG
        # target_temp = np.squeeze((img_temp - np.min(img_temp))/(np.max(img_temp) - np.min(img_temp)))
        # plt.imsave('D:/Saqib/Defocus/d3net_depth_estimation-master/d3net_depth_estimation-master/pytorch/Results/Images/aif_data/%s.png' %i, target_temp)
   
    print(img_temp.shape)
        
    for i in tqdm(range(len(target))):
       
       
        img_psnr = metrics.peak_signal_noise_ratio(np.squeeze(target[i]), np.squeeze(prediction[i]))
        # img_ssim = metrics.structural_similarity(np.squeeze(target[i]), np.squeeze(prediction[i]), data_range=2, channel_axis=2)
        # print('[PSNR]: ', img_psnr)
        avg_psnr = avg_psnr + img_psnr
        # avg_ssim = avg_ssim + img_ssim
       
    
    
    print()
    avg_PSNR_error = avg_psnr/len(target)
    print("PSNR_Avg:", avg_PSNR_error)
    # avg_SSIM_error = avg_ssim/len(target)
    # print("SSIM_Avg:", avg_SSIM_error)



#######Loading and testing depths
folder = "2HDED-selfsupervised"
experiment = "simcol3d_480_mirrored_30-100_l1_loss_2_100mm_30mm"
epoch = "best"

aif_path = fr'C:\Users\ubuntu\Desktop\Cristina\disertation-code\2HDED\results\depth\simcol3d_480_mirrored_2hded_supervised_L1+L1\best\aif_target'
pred_path = fr'C:\Users\ubuntu\Desktop\Cristina\disertation-code\{folder}\results\depth\{experiment}\{epoch}\pred_aif'
# input_path = r'C:\Users\ubuntu\Desktop\Cristina\disertation-code\2HDED\results\depth\simcol3d_480_mirrored_2hded_supervised_(L1+0.001grad)+10(Charb+2SSIM)\best\input_array'
# path_100mm = r'C:\Users\ubuntu\Desktop\Cristina\disertation-code\2HDED-selfsupervised\results\depth\simcol3d_480_mirrored_100-30\best\input_array'
# path_10mm =  r'C:\Users\ubuntu\Desktop\Cristina\disertation-code\2HDED-selfsupervised\results\depth\simcol3d_480_mirrored_100-30_loss_100mm_2_30mm\best\input_array_20mm'
# path_30mm =  r'C:\Users\ubuntu\Desktop\Cristina\disertation-code\2HDED-selfsupervised\results\depth\simcol3d_480_mirrored_100-30_loss_100mm_2_30mm\best\input_array'

test(aif_path, pred_path)

