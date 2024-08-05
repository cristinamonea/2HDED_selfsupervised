import os
import shutil
from tqdm import tqdm
import time
from collections import OrderedDict
from ipdb import set_trace as st
import random
import numpy as np
from PIL import Image
# import matplotlib.image
#import matplotlib.pyplot as plt

import re

import torch
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn

from util.visualizer import Visualizer
import networks.networks as networks


from dataloader.data_loader import CreateDataLoader

class MTL_Test():
    def name(self):
        return 'Test Model for MTL'

    def initialize(self, opt):
        self.opt = opt
        self.opt.imageSize = self.opt.imageSize if len(self.opt.imageSize) == 2 else self.opt.imageSize * 2
        self.gpu_ids = ''
        self.batchSize = self.opt.batchSize
        self.checkpoints_path = os.path.join(self.opt.checkpoints, self.opt.name)
        self.create_save_folders()

        self.netG = self.load_network()

        self.data_loader, _ = CreateDataLoader(opt)
        
        # visualizer
        self.visualizer = Visualizer(self.opt)
        if 'semantics' in self.opt.tasks:
            from util.util import get_color_palette
            self.opt.color_palette = np.array(get_color_palette(self.opt.dataset_name))
            self.opt.color_palette = list(self.opt.color_palette.reshape(-1))

    def load_network(self):
        if self.opt.epoch != 'latest' or self.opt.epoch != 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        print(checkpoint_file)
        if os.path.isfile(checkpoint_file):
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            checkpoint = torch.load(checkpoint_file)
            self.start_epoch = checkpoint['epoch']
            self.opt.net_architecture = checkpoint['arch_netG']
            netG = self.create_G_network()
            try:
                self.opt.n_classes = checkpoint['n_classes']
                self.opt.mtl_method = checkpoint['mtl_method']
                self.opt.tasks = checkpoint['tasks']
            except:
                pass
            pretrained_dict = checkpoint['state_dictG']
            pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
            for key in list(pretrained_dict.keys()):
                res = pattern.match(key)
                if res:
                    new_key = res.group(1) + res.group(2)
                    pretrained_dict[new_key] = pretrained_dict[key]
                    del pretrained_dict[key]
            netG.load_state_dict(pretrained_dict, strict=False)
            if self.opt.cuda:
                self.cuda = torch.device('cuda:0') # set externally. ToDo: set internally
                netG = netG.cuda()
            self.best_val_error = checkpoint['best_pred']

            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG
        else:
            print("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))


    def test(self):
        print('Test phase using {} split.'.format(self.opt.test_split))
        data_iter = iter(self.data_loader)
        self.netG.eval()
        total_iter = 0

        for it in tqdm(range(len(self.data_loader))):
            total_iter += 1
            input_cpu, target_cpu, depth_cpu = next(data_iter)#.next()
            
            # # #Save NP.Arry of input image VVVVV####
            # np.save('{}/input/{:04}.npy'.format(self.save_samples_path, it+1), input_cpu[0].permute(1, 2, 0))
            
            # # #Save NP.Arry of target image VVVVV####
            # np.save('{}/target/{:04}.npy'.format(self.save_samples_path, it+1), target_cpu[0].permute(1, 2, 0))
            
            # Save Np.Array of depth GT to use for evaluation
            depth_gt = depth_cpu[0,0].numpy().astype(np.uint16)
            np.save('{}/depth_gt/{:04}.npy'.format(self.save_samples_path, it+1), depth_gt)

            input_gpu = input_cpu.to(self.cuda)

            with torch.no_grad():
                depth_pred, aif_pred = self.netG.forward(input_gpu)
                
                #Save NP.Arry of outG_gpu GT VVVVV####             
                depth_pred_image = depth_pred.detach().cpu()[0,0].numpy().astype(np.uint16)
                np.save('{}/depth_pred/{:04}.npy'.format(self.save_samples_path, it+1), depth_pred_image)
    
                #Save NP.Arry of AIF PredictionsVVVVVVVVVVV
                self.save_rgb_as_png(aif_pred, '{}/aif_pred/{:04}.jpg'.format(self.save_samples_path, it+1))
                np.save('{}/aif_pred/{:04}.npy'.format(self.save_samples_path, it+1), aif_pred.detach().cpu()[0].permute(1, 2, 0))


    def create_G_network(self):
        netG = networks.define_G(self.opt.input_nc, self.opt.output_nc, 64, net_architecture=self.opt.net_architecture, opt=self.opt, gpu_ids='')
        # print(netG)
        return netG

    def save_depth_as_png(self, data, filename):
        """
        All depths are saved in np.uint8
        """
        data_np = data.data[0].cpu().float().numpy()
        data_np = data_np * self.opt.scale_to_mm
        data_np = data_np.astype(np.uint16)
        data_pil = Image.fromarray(np.squeeze(data_np))

        data_pil.save(filename)

    # def save_semantics_as_png(self, data, filename):
    #     from util.util import labels_to_colors
    #     # data_pil = Image.fromarray(np.squeeze(labels_to_colors(data, self.opt.color_palette).astype(np.)))
    #     # data_tensor = 
    #     data = data.cpu().data[0].numpy()
    #     if 'output' in filename:
    #         data = np.argmax(data, axis=0)
    #     data = data.astype(np.uint8)
    #     data_pil = Image.fromarray(data).convert('P')
    #     data_pil.putpalette(self.opt.color_palette)

        # data_pil.save(filename)

    def save_rgb_as_png(self, data, filename):
        data_np = data.data[0].cpu().float().numpy()
        data_np = np.transpose(data_np, (1,2,0))
        data_np = ((data_np + 1) / 2) * 255
        data_np = data_np.astype(np.uint8)
        data_pil = Image.fromarray(np.squeeze(data_np), mode='RGB')

        data_pil.save(filename)

    def save_as_png(self, tensor, filename, isRGB = False):
        if isRGB:
            self.save_rgb_as_png(data=tensor, filename=filename)
        # elif 'semantic' in filename:
        #     self.save_semantics_as_png(data=tensor, filename=filename)
        else:
            self.save_depth_as_png(data=tensor, filename=filename)
        # else:
        #     self.save_rgb_as_png(data=tensor, filename=filename)

    def create_save_folders(self, subfolders=['input', 'target', 'depth_gt', 'aif_pred', 'depth_pred']):
        if self.opt.save_samples:
            if self.opt.test:
                self.save_samples_path = os.path.join('results', self.opt.model, self.opt.name, self.opt.epoch)
                for subfolder in subfolders:
                    path = os.path.join(self.save_samples_path, subfolder)
                    os.system('mkdir -p {0}'.format(path))
                    # if 'input' not in subfolder:
                        # for task in self.opt.tasks:
                        #     path = os.path.join(self.save_samples_path, subfolder, task)
                        #     os.system('mkdir -p {0}'.format(path))

    def save_images(self, input, outputs, targets, index, phase='train'):
        # save other images
        self.save_as_png(input.data, '{}/input/input_{:04}.png'.format(self.save_samples_path, index))
        for i, target in enumerate(targets):
            self.save_as_png(outputs[i], '{}/output/{}/output_{}_{:04}.png'.format(self.save_samples_path, self.opt.tasks[i], self.opt.tasks[i], index))
            self.save_as_png(target, '{}/target/{}/target_{}_{:04}.png'.format(self.save_samples_path, self.opt.tasks[i], self.opt.tasks[i], index))
    
    def save_images(self, input, aif_pred, depth_pred, target_aif, target_depth, index, phase='train'):
        # save other images
        # self.save_as_png(input.data, '{}/input/input_{:04}.png'.format(self.save_samples_path, index), isRGB=True)
        # self.save_as_png(target_aif.data, '{}/input_aif/input_aif_{:04}.png'.format(self.save_samples_path, index), isRGB=True)
        self.save_as_png(target_depth.data, '{}/target/target_{:04}.png'.format(self.save_samples_path, index), isRGB=False)
        self.save_as_png(depth_pred.data, '{}/output/output_{:04}.png'.format(self.save_samples_path, index), isRGB=False)
        self.save_as_png(aif_pred.data, '{}/output_aif/output_aif_{:04}.png'.format(self.save_samples_path, index), isRGB=True)
        
