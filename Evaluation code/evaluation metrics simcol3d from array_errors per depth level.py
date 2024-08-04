# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 13:55:08 2024

@author: ubuntu
"""

import os
import numpy as np
from PIL import Image


depth_v_err = True

MIN_DEPTH = 0
MAX_DEPTH = 300
N_BINS = 100


def compute_depth_v_error(gt: np.Array, pred: np.Array, bin_ed: np.Array):
    count = np.zeros(bin_ed.size - 1, dtype=int)
    total_err = np.zeros(bin_ed.size - 1, dtype="float32")
    for i, (l,r) in enumerate(zip(bin_ed[:-1], bin_ed[1:])):
        selection = (l<gt) & (gt<r)
        gti = gt[selection]
        predi = pred[selection]
        count[i] = gti.size
        if count[i] == 0:
            continue
        abs_rel_c = np.sum(np.abs(gti - predi))  # Abs. err.
        abs_rel[i] = abs_rel_c

    return count, abs_rel


folder = "2HDED-selfsupervised"
experiment = "simcol3d_480_mirrored_charb_100-30_loss_100mm_2_30mm"
epoch = "best"

subset = "SyntheticColon_III"

# Paths to the folders
gt_folder = fr'C:\Users\ubuntu\Desktop\Cristina\disertation-code\depth_estimation_endoscopy\results\depth\simcol3d_480_mirrored_d3net_with_defocus\best\target'
output_folder = fr'C:\Users\ubuntu\Desktop\Cristina\disertation-code\{folder}\results\depth\{experiment}\{epoch}\pred_depth'

# List of file names in the ground truth folder
gt_files = os.listdir(gt_folder)
output_files = os.listdir(output_folder)

match subset:
    case "SyntheticColon_III":
        gt_subset_files = gt_files[:1803]
        output_subset_files = output_files[:1803]
    case "SyntheticColon_II":
        gt_subset_files = gt_files[1803:5406]
        output_subset_files = output_files[1803:5406]
    case "SyntheticColon_I":
        gt_subset_files = gt_files[5406:]
        output_subset_files = output_files[5406:]
# Initialize lists to store errors
L1 = []
rel = []
rmse = []

if depth_v_err:
        hist_ed = np.linspace(MIN_DEPTH, MAX_DEPTH, N_BINS+1)
        all_err = []
        all_hist = []

for i in range(len(gt_subset_files)):
    # Construct full file paths
    gt_path = os.path.join(gt_folder, gt_subset_files[i])
    output_path = os.path.join(output_folder, output_subset_files[i])

    # Read the images
    true_depth = np.load(gt_path)
    predicted_depth = np.load(output_path)
    #mask = (true_depth > 0) & (predicted_depth > 0)
    
    L1_error = np.mean(np.abs(predicted_depth - true_depth))
    #rel_error = np.median(np.abs((predicted_depth - true_depth)/(true_depth + 10e-5))) * 100
    RMSE_error = np.sqrt(np.mean((predicted_depth - true_depth)**2))
    
    L1.append(L1_error)
   # rel.append(rel_error)
    rmse.append(RMSE_error)

    if depth_v_err:
            add_hist, sum_err = compute_depth_v_error(gt_depth, pred_depth, hist_ed)
            all_err.append(sum_err)
            all_hist.append(add_hist)


    # mean_abs.append(np.mean(np.abs(true_depth[mask] - predicted_depth[mask])))

    # # Calculate errors for the current image and add to lists
    # abs_rel_error = np.mean(np.abs(true_depth[mask] - predicted_depth[mask]) / true_depth[mask])
    # abs_rel_errors.append(abs_rel_error)

    # sq_rel_error = np.mean(((true_depth[mask] - predicted_depth[mask]) ** 2) / true_depth[mask])
    # sq_rel_errors.append(sq_rel_error)

    # rmse = np.sqrt(np.mean((true_depth[mask] - predicted_depth[mask]) ** 2))
    # rmse_errors.append(rmse)

    # rmse_log = np.sqrt(np.mean((np.log(true_depth[mask]) - np.log(predicted_depth[mask])) ** 2))
    # rmse_log_errors.append(rmse_log)

# Calculate the average errors across all images

if depth_v_err:
        all_hist = np.array(all_hist)
        all_err = np.array(all_err)
        total_hist = np.sum(all_hist,0,keepdims=True)
        total_err = np.sum(all_hist/total_hist*all_err,0)
        total_hist = total_hist.squeeze()
        # write to folder
        folder =  '.'
        np.savez(os.path.join(folder,'depth_v_err.nbz'),
                 bin_ed=hist_ed,
                 hist=total_hist,
                 abs_err=total_err)



print(subset)
print(f"L1: {np.mean(L1)}")
#print(f"rel: {np.mean(rel)}")
print(f"RMSE: {np.mean(rmse)}")


