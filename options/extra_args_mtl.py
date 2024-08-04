from .arguments import TrainOptions
import torch

class MTL_Options(TrainOptions):
    def __init__(self, mtl_method='eweights', tasks=['depth'], outputs_nc=[1], regression_loss='L1', alpha=0.5, 
                 depth_loss='L1', depth_reg=None, aif_loss='L1', aif_reg=None, 
                 depth_loss_coef=1, aif_loss_coef=1, depth_reg_coef=1, aif_reg_coef=1, 
                 kernel_size = 601, focal_lenght = 1, focal_depth = 1500, aperture = 2.5, aperture_2 = 2.5, pixel_size = 0.0044, scale=1 ,**kwargs):
        super().__init__(**kwargs)  # Initialize parent class (TrainOptions) with all additional arguments

        # MTL specific arguments
        self.mtl_method = mtl_method
        self.tasks = tasks
        self.outputs_nc = outputs_nc
        self.regression_loss = regression_loss
        self.alpha = alpha 
        self.depth_loss = depth_loss
        self.depth_reg= depth_reg
        self.aif_loss = aif_loss
        self.aif_reg = aif_reg
        self.depth_loss_coef = depth_loss_coef
        self.aif_loss_coef = aif_loss_coef        
        self.depth_reg_coef = depth_reg_coef
        self.aif_reg_coef = aif_reg_coef

        self.kernel_size = kernel_size 
        self.pixel_size = pixel_size
        self.scale = scale

        self.focal_length = torch.Tensor([focal_lenght] * self.batchSize).float().cuda()
        self.focal_length_val = torch.Tensor([focal_lenght] * 1).float().cuda()

        self.focal_depth =  torch.Tensor([focal_depth] * self.batchSize).float().cuda()
        self.focal_depth_val =  torch.Tensor([focal_depth] * 1).float().cuda()
        
        self.aperture =  torch.Tensor([aperture] * self.batchSize).float().cuda()
        self.aperture_val =  torch.Tensor([aperture] * 1).float().cuda()

        self.aperture_2 = torch.Tensor([aperture_2] * self.batchSize).float().cuda()
        self.aperture_val_2 =  torch.Tensor([aperture_2] * 1).float().cuda()