import os
import time
import torch
import torch.nn as nn
from collections import OrderedDict
from ipdb import set_trace as st

from math import sqrt
from tqdm import tqdm
from .train_model import TrainModel
from networks import networks

import util.pytorch_ssim as pytorch_ssim

from sklearn.metrics import confusion_matrix
import util.semseg.metrics.raster as metrics
import numpy as np
from torch.optim.lr_scheduler import ReduceLROnPlateau
# import sobel
# from piqa import SSIM
from torch import optim
# from sewar.full_ref import ssim
# import pytorch_ssim
# import torch
import torch.nn.functional as F

from torch.optim.lr_scheduler import StepLR

from Utils import Losses, GaussPSF

#from piq import ssim, SSIMLoss

#from piqa import SSIM

#class SSIMLoss(SSIM):
#    def forward(self, x, y):
#        return 1. - super().forward(x, y)
#def L_blur(images):
 #   beta = 2.5 # or any other value of your choice
  #  N = images.shape[0]
   # M = images.shape[1]*images.shape[2]
    #mu = images.mean()
    #loss = 0.0
    #for c in range(N):
           
     #   Lap_filter = torch.Tensor([[0,-1,0],[-1,4,-1],[0,-1,0]])
      #  dX = F.conv2d(images[c,:,:], Lap_filter, padding=1)
       # loss -= beta * torch.log(torch.sum(dX**2) / (M - mu**2))
   # blurr_loss /= N
    #return blurr_loss



# Be able to add many loss functions
class MultiTaskGen(TrainModel):
    def name(self):
        return 'MultiTask General Model'

    def initialize(self, opt):
        TrainModel.initialize(self, opt)
        
        if self.opt.resume:
            self.netG, self.optimG = self.load_network()
        elif self.opt.train:
            from os.path import isdir
            if isdir(self.opt.pretrained_path) and self.opt.pretrained:
                self.netG = self.load_weights_from_pretrained_model()
            else:
                self.netG = self.create_network()
            self.optimG = self.get_optimizerG(self.netG, self.opt.lr,
                                              weight_decay=self.opt.weightDecay)
            
            self.scheduler = StepLR(self.optimG, step_size=5, gamma=0.7169)

        self.generator = GaussPSF(self.opt.kernel_size, 1e-3, 10000, self.opt.pixel_size, self.opt.scale).cuda()
        self.generator = torch.nn.DataParallel(self.generator)
        self.generator = self.generator.cuda()

        self.generator_2 = GaussPSF(self.opt.kernel_size, 1e-3, 10000, self.opt.pixel_size, self.opt.scale).cuda()
        self.generator_2 = torch.nn.DataParallel(self.generator)
        self.generator_2 = self.generator_2.cuda()
        
            # self.criterion = self.create_reg_criterion()
        self.n_tasks = len(self.opt.tasks)
        self.lr_sc = ReduceLROnPlateau(self.optimG, 'min', patience=500)
        
        if self.opt.display_id > 0:
            self.errors = OrderedDict()
            self.current_visuals = OrderedDict()
        if 'depth' in self.opt.tasks:
            self.criterion_reg = self.get_regression_criterion()
            
            ############AIF
        # if 'aif' in self.opt.tasks:
        #     self.criterion_charbonnier = self.get_errors_charbonnier()
            
        if 'semantics' in self.opt.tasks:
            self.initialize_semantics()
        if 'instance' in self.opt.tasks:
            pass
        if 'normals' in self.opt.tasks:
            pass

    def initialize_semantics(self):
        from util.util import get_color_palette, get_dataset_semantic_weights
        self.global_cm = np.zeros((self.opt.n_classes-1, self.opt.n_classes-1))
        self.target = self.get_variable(torch.LongTensor(self.batchSize, self.opt.output_nc, self.opt.imageSize[0], self.opt.imageSize[1]))
        self.outG_np = None
        self.overall_acc = 0
        self.average_acc = 0
        self.average_iou = 0
        self.opt.color_palette = get_color_palette(self.opt.dataset_name)

        weights = self.get_variable(torch.FloatTensor(get_dataset_semantic_weights(self.opt.dataset_name)))
        self.cross_entropy = nn.CrossEntropyLoss(weight=weights)
    
    def train_batch(self):
        self._train_batch()

    def restart_variables(self):
        self.it = 0
        self.n_iterations = 0
        self.n_images = 0
        self.rmse = 0
        self.e_reg = 0
        self.norm_grad_sum = 0

    def mean_errors(self):
        if 'depth' in self.opt.tasks:
            rmse_epoch = self.rmse / self.n_images
            self.set_current_errors(RMSE=rmse_epoch)

    def get_errors_regression(self, target, output):
        e_regression = self.criterion_reg(output, target)
        if self.total_iter % self.opt.print_freq == 0:

            # gets valid pixels of output and target
            # if not self.opt.no_mask:
            #     (output, target), n_valid_pixls = self.apply_valid_pixels_mask(output, target, value=self.opt.mask_thres)

            with torch.no_grad():
                for k in range(output.shape[0]):
                    self.rmse += sqrt(self.mse_scaled_error(output[k], target[k], n_valid_pixls).item()) # mean through the batch
                    self.n_images += 1

            # self.set_current_visuals(depth_gt=target.data,
            #                             depth_pred=output.data)
            self.set_current_errors(L1=e_regression.item())
            
        return e_regression

    def get_errors_semantics(self, target, output, n_classes):
        # e_semantics = self.cross_entropy(output, target)
        if self.total_iter % self.opt.print_freq == 0:
            with torch.no_grad():
                target_sem_np = target.cpu().numpy()
                output_np = np.argmax(output.cpu().data.numpy(), axis=1)
                cm = confusion_matrix(target_sem_np.ravel(), output_np.ravel(), labels=list(range(n_classes)))
                self.global_cm += cm[1:,1:]

                # scores
                overall_acc = metrics.stats_overall_accuracy(self.global_cm)
                average_acc, _ = metrics.stats_accuracy_per_class(self.global_cm)
                average_iou, _ = metrics.stats_iou_per_class(self.global_cm)

                self.set_current_errors(OAcc=overall_acc, AAcc=average_acc, AIoU=average_iou)
                self.set_current_visuals(sem_gt=target.data[0].cpu().float().numpy(),
                                        sem_out=output_np[0])

            # return e_semantics
    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def get_errors_instance(self, target, output):
        pass

    def get_errors_normals(self, target, output):
        pass

#####Smoothing loss#######
    def get_smooth_loss(disp, img):
                 """Computes the smoothness loss for a disparity image
    The color image is used for edge-aware smoothness
    """
                 grad_disp_x = torch.abs(disp[:, :, :, :-1] - disp[:, :, :, 1:])
                 grad_disp_y = torch.abs(disp[:, :, :-1, :] - disp[:, :, 1:, :])

                 grad_img_x = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]), 1, keepdim=True)
                 grad_img_y = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]), 1, keepdim=True)

                 grad_disp_x *= torch.exp(-grad_img_x)
                 grad_disp_y *= torch.exp(-grad_img_y)
                 return grad_disp_x.mean() + grad_disp_y.mean()
    
##########################
    def _train_batch(self):
            input_cpu, target_cpu, depth_cpu = next(self.data_iter)

            input_data = input_cpu.to(self.device)
            input_data.requires_grad = True
            target_data = target_cpu.to(self.device)

            depth_pred, aif_pred = self.netG.forward(input_data)

            if (torch.isnan(depth_pred).any() or torch.isnan(aif_pred).any()):
                print("NaN at network output")

            ################## Training on iDFD #################################
            img_refoc_aperture_1 = self.generator(aif_pred, depth_pred, self.opt.focal_depth, self.opt.aperture, self.opt.focal_length)
            img_refoc_aperture_2 = self.generator(aif_pred, depth_pred, self.opt.focal_depth, self.opt.aperture_2, self.opt.focal_length)

            self.set_current_visuals(orig_aperture_1=input_data.data)
            self.set_current_visuals(orig_aperture_2=target_data.data)            
            self.set_current_visuals(aif_pred=aif_pred.data)
            self.set_current_visuals(depth_gt=depth_cpu.data)
            self.set_current_visuals(depth_pred=depth_pred.data)
            # # Loss for refoc image aperture f/2.8
            # # L1 Loss
            # l1 = nn.L1Loss()
            # l1_loss_aperture_1=l1(img_refoc_aperture_1, input_data)

            # Charb Loss
            charb_aperture_1 = self.charb(img_refoc_aperture_1, input_data)

            # SSIM Loss
            lssim_aperture_1 = 1 - pytorch_ssim.ssim(img_refoc_aperture_1, input_data)

            # # Loss for refoc image aperture f/10
            # # L1 Loss
            # l1 = nn.L1Loss()
            # l1_loss_aperture_2=l1(img_refoc_aperture_2, target_data)

            # Charb Loss
            charb_aperture_2 = self.charb(img_refoc_aperture_2, target_data)

            # SSIM Loss
            lssim_aperture_2 = 1 - pytorch_ssim.ssim(img_refoc_aperture_2, target_data)

            # # Smoothness loss
            #try first using the f\10 image as reference, than the predicted aif
            grad_aif_depth = self.gradient_aif_depth(aif_pred, depth_pred)

            self.loss_error = (charb_aperture_1 + 2*lssim_aperture_1) + 2*(charb_aperture_2 + 2*lssim_aperture_2) + 0.01 * grad_aif_depth

            ################## End of training on iDFD #################################
                
            # ################## Training on SimCol3D #################################
            # img_refoc_30mm = self.generator(aif_pred, depth_pred, torch.Tensor([100] * self.batchSize).float().cuda(), self.opt.aperture, self.opt.focal_length)
            # img_refoc_100mm = self.generator(aif_pred, depth_pred, torch.Tensor([30] * self.batchSize).float().cuda(), self.opt.aperture, self.opt.focal_length)

            # #self.set_current_visuals(img_refoc_30mm=img_refoc_30mm.data)
            # # self.set_current_visuals(target_100mm=target_cpu.data)
            # self.set_current_visuals(depth_pred=depth_pred.data)

            # # if (torch.isnan(img_refoc_100mm).any() or torch.isnan(img_refoc_100mm).any()):
            # #     print("NaN at refocalisation")
            
            # losses = []
        
            # target = target_cpu.to(self.device)
            
            # # aif = aif_cpu.to(self.device)

            # # # Loss for refoc image 30mm
            # l1 = nn.L1Loss()
            # l1_loss_100mm=l1(img_refoc_100mm, target)
            # # losses.append(l1_loss_100mm)
            # # # self.set_current_errors(L1_refoc=l1_loss_30mm.item())
            
            
            # ssim_value_100mm = pytorch_ssim.ssim(img_refoc_100mm, target)
            # lssim_100mm = 1 - ssim_value_100mm
            # # self.set_current_errors(SSIM_refoc=lssim_30mm.item())

            #  # Loss for refoc image 30mm
            # # charb_100mm = self.charb(img_refoc_100mm, target)
            # # self.set_current_errors(L1_refoc=l1_loss_30mm.item())
            
            
            # # ssim_value_100mm = pytorch_ssim.ssim(img_refoc_100mm, target)
            # # lssim_100mm = 1 - ssim_value_100mm
            # # self.set_current_errors(SSIM_refoc=lssim_30mm.item())


            # # # Loss for refoc image 100mm
            # l1 = nn.L1Loss()
            # l1_loss_30mm=l1(img_refoc_30mm, input_data)
            # # losses.append(l1_loss_30mm)
            # # #self.set_current_errors(L1_refoc=l1_loss_100mm.item())
            

            # # ssim_value_30mm = pytorch_ssim.ssim(img_refoc_30mm, input_data)
            # # lssim_30mm = 1 - ssim_value_30mm
            # # #self.set_current_errors(SSIM_refoc=lssim_100mm.item())

            #  # Loss for refoc image 100mm
            # # charb_30mm=self.charb(img_refoc_30mm, input_data)
            # # losses.append(charb_30mm)
            # # #self.set_current_errors(L1_refoc=l1_loss_100mm.item())
            

            # ssim_value_30mm = pytorch_ssim.ssim(img_refoc_30mm, input_data)
            # lssim_30mm = 1 - ssim_value_30mm
            # #self.set_current_errors(SSIM_refoc=lssim_100mm.item())

            # #Aif loss
            # # charb_aif=self.charb(aif_pred, aif)
            # #self.set_current_errors(L1_refoc=l1_loss_100mm.item())
            # # l1 = nn.L1Loss()
            # # l1_aif=l1(aif_pred, aif)

            # # ssim_value_aif = pytorch_ssim.ssim(aif_pred, aif)
            # # lssim_aif = 1 - ssim_value_aif

            # # # Sharpness loss for 30mm
            # # sharp_30mm = self.sharpness_loss(img_refoc_30mm, target)
            # # # Sharpness loss for 100mm
            # # sharp_100mm = self.sharpness_loss(img_refoc_100mm, input_data)

            # # Smoothness loss
            # # grad_aif_depth = self.gradient_aif_depth(aif_pred, depth) 

            # #Charb
            # # aif_charb = self.charb(aif, aif_pred)
            
    
            # #self.loss_error = (charb_aif + 2*lssim_aif) + (charb_30mm + 2*lssim_30mm) + (charb_100mm + 2*lssim_100mm) #+ 0.001*grad_aif_depth #+ aif_charb #+ 0.0001*sharp_30mm + 0.0001*sharp_100mm
            # self.loss_error = (l1_loss_30mm + 2*lssim_30mm) + 2*(l1_loss_100mm + 2*lssim_100mm)
            ################## End of Training on SimCol3D #################################

            # if (torch.isnan(self.loss_error)):
            #     print("NaN at loss calculation")

            self.set_current_errors(L2HDED=self.loss_error.item())

            self.optimG.zero_grad()
            self.loss_error.backward()
            
            ###Scheduler
            self.optimG.step()

            self.n_iterations += 1 # outG[0].shape[0]

            with torch.no_grad():
                # Calculate depth estimation error, just for visualisation
                l1 = nn.L1Loss()
                l1_depth = l1(depth_pred.to('cpu'), depth_cpu)
                self.set_current_errors(L1_true_depth=l1_depth.item())

       #this method is used to evaluate
    def evaluate(self, data_loader, epoch):
        if self.opt.validate and self.total_iter % self.opt.val_freq == 0:
            val_error = self.get_eval_error(data_loader)
            self.visualizer.display_errors(self.val_errors, epoch, float(self.it)/self.len_data_loader, phase='val')
            self.visualizer.save_errors_file(self.logfile_val)
            message = self.visualizer.print_errors(self.val_errors, epoch, self.it, len(data_loader), 0)
            print('[Validation] ' + message)
            self.visualizer.display_images(self.val_current_visuals, epoch=epoch, phase='val')
            is_best = self.best_val_error > self.val_errors[f'{"L2HDED"}']
            if is_best:     # and not self.opt.not_save_val_model:
                    print("Updating BEST model (epoch {}, iters {})\n".format(epoch, self.total_iter))
                    self.best_val_error = self.val_errors['L2HDED']
                    self.save_checkpoint(epoch, is_best)

    def gradient(self, inp):
        D_dy = inp[:, :, :, :] - F.pad(inp[:, :, :-1, :], (0, 0, 1, 0))
        D_dx = inp[:, :, :, :] - F.pad(inp[:, :, :, :-1], (1, 0, 0, 0))
        return D_dx, D_dy

    def sharpness(self, image):
        grad = self.gradient(image)
        mu = self.AVG(image) + 1e-8
        output = - (grad[0]**2 + grad[1]**2) - torch.abs((image - mu) / mu) - torch.pow(image - mu, 2)
        return output

    def sharpness_loss(self, input, pred):
            grad = self.gradient(input)
            ### avegare
            mu = F.avg_pool2d(input, self.opt.kernel_size, 1, self.opt.kernel_size // 2, count_include_pad=False) + 1e-8
            sharp_input = - (grad[0]**2 + grad[1]**2) - torch.abs((input - mu) / mu) - torch.pow(input - mu, 2)

            grad_pred = self.gradient(pred)
            ### avegare
            mu = F.avg_pool2d(input, self.opt.kernel_size, 1, self.opt.kernel_size // 2, count_include_pad=False) + 1e-8
            sharp_pred = - (grad[0]**2 + grad[1]**2) - torch.abs((pred - mu) / mu) - torch.pow(pred - mu, 2)
            l1 = nn.L1Loss()

            return l1(sharp_input, sharp_pred)
    
    def gradient_aif_depth(self, pred_aif, pred_depth):
        aif_grad = self.gradient(pred_aif)
        aif_grad_x_exp = torch.exp(-aif_grad[0].abs())
        aif_grad_y_exp = torch.exp(-aif_grad[1].abs())

        dx, dy = self.gradient(pred_depth)
        dD_x = dx.abs() * aif_grad_x_exp
        dD_y = dy.abs() * aif_grad_y_exp
        sm_loss = (dD_x + dD_y).mean()

        return sm_loss
    
    def charb(self, aif_pred, aif_gt):
        charb_diff = torch.add(aif_pred, -aif_gt)
        charb_error = torch.sqrt( charb_diff * charb_diff + 1e-12 )
        charb_elems = aif_gt.shape[0] * aif_gt.shape[1] * aif_gt.shape[2] * aif_gt.shape[3]  
        return torch.sum(charb_error) / charb_elems
                    

    # def evaluate(self, data_loader, epoch):
    #     if self.opt.validate and self.total_iter % self.opt.val_freq == 0:
    #         self.get_eval_error(data_loader)
    #         self.visualizer.display_errors(self.val_errors, epoch, float(self.it)/self.len_data_loader, phase='val')
    #         message = self.visualizer.print_errors(self.val_errors, epoch, self.it, len(data_loader), 0)
    #         print('[Validation] ' + message)
    #         self.visualizer.display_images(self.val_current_visuals, epoch=epoch, phase='val')

    def calculate_val_loss(self, data_loader):
        # print('In Val loss fnc \n')
        model = self.netG.train(False)
        aif_pred_list = list()
        aif_data_list = list()
        losses = np.zeros(self.n_tasks)
        aif_err = 0.0
        with torch.no_grad():
            pbar_val = range(len(data_loader))
            data_iter = iter(data_loader)
            for _ in pbar_val:
                #pbar_val.set_description('[Validation]')
                input_cpu, target_cpu, aif_cpu = next(data_iter)#.next()
                input_data = input_cpu.to(self.device)
                aif_data = aif_cpu.to(self.device)
                
                outG, aif_pred = model.forward(input_data)
                aif_pred_list.append(aif_pred)
                aif_data_list.append(aif_data)
                
                for i_task, task in enumerate(self.opt.tasks):
                    target = target_cpu[i_task].to(self.device)
                    if task == 'semantics':
                        target_np = target_cpu[i_task].data.numpy()
                        output_np = np.argmax(outG[i_task].cpu().data.numpy(), axis=1)
                        cm = confusion_matrix(target_np.ravel(), output_np.ravel(), labels=list(range(self.opt.outputs_nc[i_task])))
                        target_cpu[i_task] = target.data[0].cpu().float().numpy()
                        outG[i_task] = output_np[0]
                        loss, _ = metrics.stats_iou_per_class(cm[1:,1:])
                        losses[i_task] += loss
                    elif task == 'depth':
                        losses[i_task] += sqrt(nn.MSELoss()(target, outG[i_task]))
                        outG[i_task] = outG[i_task].data
                 
                
                        
                # TODO: ADD AIF ERROR
                ##L1
                l1 = nn.L1Loss()
                l1_loss=l1(aif_pred,aif_data)
                aif_err += l1_loss
                # total_loss = 0
                # for idx, value in enumerate(aif_pred_list):
                #     total_loss = total_loss + l1(aif_pred_list[idx], aif_data_list[idx])
                # aif_err += total_loss/len(aif_pred_list)
                ##Charb
                # charb_diff = torch.add(aif_pred, -aif_data)
                # charb_error = torch.sqrt( charb_diff * charb_diff + 1e-6 )
                # charb_elems = aif_data.shape[0] * aif_data.shape[1] * aif_data.shape[2] * aif_data.shape[3]  
                # loss_aif = torch.sum(charb_error) / charb_elems
                # aif_err += loss_aif
                # losses.append(self.criterion_reg(aif_data, aif_pred))
                # loss_aif =  self.get_errors_regression(aif_data, aif_pred)
                # aif_err += loss_aif
                return aif_err
        
    def get_eval_error(self, data_loader):
        model = self.netG.train(False)
        self.val_errors = OrderedDict()
        self.val_current_visuals = OrderedDict()
        aif_pred_list = list()
   
        L2hded_error = 0.0
        l1_true_depth = 0.0
        
        with torch.no_grad():
            pbar_val = tqdm(range(len(data_loader)))
            data_iter = iter(data_loader)
            for _ in pbar_val:
                pbar_val.set_description('[Validation]')
                input_cpu, target_cpu, depth_cpu = next(data_iter)#.next()

                input_data = input_cpu.to(self.device)
                target_data = target_cpu.to(self.device)
                #aif_data = aif_cpu.to(self.device)

                depth_pred, aif_pred = model.forward(input_data)

                ################## Validate on iDFD #################################
                img_refoc_aperture_1 = self.generator(aif_pred, depth_pred, self.opt.focal_depth_val, self.opt.aperture_val, self.opt.focal_length_val)
                img_refoc_aperture_2 = self.generator(aif_pred, depth_pred, self.opt.focal_depth_val, self.opt.aperture_val_2, self.opt.focal_length_val)

                self.val_current_visuals.update([("orig_aperture_1", input_data.data)])
                self.val_current_visuals.update([("orig_aperture_2", target_data.data)])
                self.val_current_visuals.update([("aif_pred", aif_pred.data)])
                self.val_current_visuals.update([("depth_gt", depth_cpu.data)])
                self.val_current_visuals.update([("depth_pred", depth_pred.data)])

                # # Loss for refoc image aperture f/2.8
                # # L1 Loss
                # l1 = nn.L1Loss()
                # l1_loss_aperture_1=l1(img_refoc_aperture_1, input_data)

                # Charb Loss
                charb_aperture_1 = self.charb(img_refoc_aperture_1, input_data)

                # SSIM Loss
                lssim_aperture_1 = 1 - pytorch_ssim.ssim(img_refoc_aperture_1, input_data)

                # # Loss for refoc image aperture f/10
                # # L1 Loss
                # l1 = nn.L1Loss()
                # l1_loss_aperture_2=l1(img_refoc_aperture_2, target_data)

                # Charb Loss
                charb_aperture_2 = self.charb(img_refoc_aperture_2, target_data)

                # SSIM Loss
                lssim_aperture_2 = 1 - pytorch_ssim.ssim(img_refoc_aperture_2, target_data)

                # # Smoothness loss
                #try first using the f\10 image as reference, than the predicted aif
                grad_aif_depth = self.gradient_aif_depth(aif_pred, depth_pred)

                self.loss_error = (charb_aperture_1 + 2*lssim_aperture_1) + 2*(charb_aperture_2 + 2*lssim_aperture_2) + 0.01 * grad_aif_depth

            ################## End of validation on iDFD #################################
                
                # ################## Validate on SimCol3D #################################
            #     aif_pred_list.append(aif_pred)

            #     img_refoc_30mm = self.generator(aif_pred, depth, torch.Tensor([100] * 1).float().cuda(), self.opt.aperture_val, self.opt.focal_length_val)
            #     img_refoc_100mm = self.generator(aif_pred, depth, torch.Tensor([30] * 1).float().cuda(), self.opt.aperture_val, self.opt.focal_length_val)

            #     # Loss for refoc image 30mm
            #     l1 = nn.L1Loss()
            #     l1_loss_100mm=l1(img_refoc_100mm, target_data)
            #     # # self.set_current_errors(L1_refoc=l1_loss_100mm.item())
                
            #     # # charb_100mm = self.charb(img_refoc_100mm, target_data)
            #     # # self.set_current_errors(L1_refoc=l1_loss_30mm.item())

            #     ssim_value_100mm = pytorch_ssim.ssim(img_refoc_100mm, target_data)
            #     lssim_100mm = 1 - ssim_value_100mm
            #     # self.set_current_errors(SSIM_refoc=lssim_100mm.item())

            #     # Loss for refoc image 100mm
            #     l1 = nn.L1Loss()
            #     l1_loss_30mm=l1(img_refoc_30mm, input_data)
            #     #self.set_current_errors(L1_refoc=l1_loss_100mm.item())
            #     # charb_30mm = self.charb(img_refoc_30mm, input_data)
                

            #     ssim_value_30mm = pytorch_ssim.ssim(img_refoc_30mm, input_data)
            #     lssim_30mm = 1 - ssim_value_30mm
            #     #self.set_current_errors(SSIM_refoc=lssim_100mm.item())

            #     # # Sharpness loss for 30mm
            #     # sharp_30mm = self.sharpness_loss(img_refoc_30mm, target)
            #     # # Sharpness loss for 100mm
            #     # sharp_100mm = self.sharpness_loss(img_refoc_100mm, input_data)

            #     # charb_aif = self.charb(aif_pred, aif_data)
                
            #     l1_aif = l1(aif_pred, aif_data)

            #     ssim_value_aif = pytorch_ssim.ssim(aif_pred, aif_data)
            #     lssim_aif = 1 - ssim_value_aif

            #     # grad_aif_depth = self.gradient_aif_depth(aif_pred, depth) 

            #     #Charb
            #     # aif_charb = self.charb(aif_data, aif_pred)
                
            #     self.loss_error =  (l1_loss_30mm + 2*lssim_30mm) + 2*(l1_loss_100mm + 2*lssim_100mm)

               
                
            # self.val_current_visuals.update([("input_100mm", input_cpu)])
            # self.val_current_visuals.update([("aif_pred", aif_pred)])
            # # self.val_current_visuals.update([("img_refoc_30mm", img_refoc_30mm)])
            # self.val_current_visuals.update([("target_100mm", target_cpu)])
            # self.val_current_visuals.update([("depth_pred", depth)])
            ################## End of validation on SimCol3D #################################
                L2hded_error += self.loss_error

                l1 = nn.L1Loss()
                l1_true_depth += l1(depth_pred.to('cpu'), depth_cpu)                                  

            self.val_errors.update([("L2HDED", L2hded_error.item() / len(data_loader))])
            self.val_errors.update([("l1_true_depth", l1_true_depth / len(data_loader))])

    def set_current_errors_string(self, key, value):
        self.errors.update([(key, value)])

    def set_current_errors(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.errors.update([(key, value)])

    def get_current_errors(self):
        return self.errors

    def get_current_errors_display(self):
        return self.errors

    def set_current_visuals(self, **k_dict_elements):
        for key, value in k_dict_elements.items():
            self.current_visuals.update([(key, value)])

    def get_current_visuals(self):
        return self.current_visuals

    def get_checkpoint(self, epoch):
        return ({'epoch': epoch,
                 'arch_netG': self.opt.net_architecture,
                 'state_dictG': self.netG.state_dict(),
                 'optimizerG': self.optimG,
                 'best_pred': self.best_val_error,
                 'tasks': self.opt.tasks,
                 'mtl_method': self.opt.mtl_method,
                #  'data_augmentation': self.opt.data_augmentation, # used before loading net
                 'n_classes': self.opt.output_nc,
                 })

    def load_network(self):
        if self.opt.epoch != 'latest' or self.opt.epoch != 'best':
            self.opt.epoch = self.opt.epoch.zfill(4)
        checkpoint_file = os.path.join(self.checkpoints_path, self.opt.epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
            self.start_epoch = checkpoint['epoch']
            self.opt.net_architecture = checkpoint['arch_netG']
            self.opt.n_classes = checkpoint['n_classes']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            netG = self.create_network()
            model_dict = netG.state_dict()
            pretrained_dict = checkpoint['state_dictG']
            model_shapes = [v.shape for k, v in model_dict.items()]
            exclude_model_dict = [k for k, v in pretrained_dict.items() if v.shape not in model_shapes]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in exclude_model_dict}
            model_dict.update(pretrained_dict)
            netG.load_state_dict(model_dict)
            optimG = checkpoint['optimizerG']
            self.best_val_error = checkpoint['best_pred']
            self.print_save_options()
            print("Loaded model from epoch {}".format(self.start_epoch))
            return netG, optimG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))

        # if os.path.isfile(checkpoint_file):
        #     checkpoint = torch.load(checkpoint_file)
        #     print("Loading {} checkpoint of model {} ...".format(self.opt.epoch, self.opt.name))
        #     self.start_epoch = checkpoint['epoch']
        #     self.opt.net_architecture = checkpoint['arch_netG']
        #     self.opt.n_classes = checkpoint['n_classes']
        #     self.opt.mtl_method = checkpoint['mtl_method']
        #     self.opt.tasks = checkpoint['tasks']
        #     netG = self.create_network()
        #     netG.load_state_dict(checkpoint['state_dictG'])
        #     optimG = checkpoint['optimizerG']
        #     self.best_val_error = checkpoint['best_pred']
        #     self.print_save_options()
        #     print("Loaded model from epoch {}".format(self.start_epoch))
        #     return netG, optimG
        # else:
        #     raise ValueError("Couldn't find checkpoint on path: {}".format(self.checkpoints_path + '/' + self.opt.epoch))

    def load_weights_from_pretrained_model(self):
        epoch = 'best'
        checkpoint_file = os.path.join(self.opt.pretrained_path, epoch + '.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            print("Loading {} checkpoint of model {} ...".format(epoch, self.opt.pretrained_path))
            self.opt.net_architecture = checkpoint['arch_netG']
            self.opt.n_classes = checkpoint['n_classes']
            self.opt.mtl_method = checkpoint['mtl_method']
            self.opt.tasks = checkpoint['tasks']
            netG = self.create_network()
            model_dict = netG.state_dict()
            pretrained_dict = checkpoint['state_dictG']
            model_shapes = [v.shape for k, v in model_dict.items()]
            exclude_model_dict = [k for k, v in pretrained_dict.items() if v.shape not in model_shapes]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and k not in exclude_model_dict}
            model_dict.update(pretrained_dict)
            netG.load_state_dict(model_dict)
            _epoch = checkpoint['epoch']
            # netG.load_state_dict(checkpoint['state_dictG'])
            print("Loaded model from epoch {}".format(_epoch))
            return netG
        else:
            raise ValueError("Couldn't find checkpoint on path: {}".format(self.pretrained_path + '/' + epoch))

    def to_numpy(self, data):
        return data.data.cpu().numpy()