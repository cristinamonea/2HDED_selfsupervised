# simplified main
from options.extra_args_mtl import MTL_Options as TrainOptions
from dataloader.data_loader import CreateDataLoader

# Load options
opt = TrainOptions(dataroot=r".\datasets\Simcol3D",
                   name="simcol3d_480_mirrored_30-100_l1_loss_2_100mm_30mm", 
                   imageSize=[480], 
                   outputSize=[480],
                   resume=False,
                   train=False, 
                   validate=False,
                   test=True,
                   epoch='latest', 
                   visualize=False, 
                   test_split='test', 
                   train_split='train', 
                   val_split='val',
                   
                   aif_loss_coef=1,
                   depth_reg_coef=0,
                   aif_reg_coef=0,
                   use_dropout=False,
                   use_skips=True,
                   cuda=True, 
                   nEpochs=13,  # Updated to match 'nepochs'
                   batchSize=8,  # Updated to match 'batch_size'
                   init_method='normal', 
                   data_augmentation=["f", "f", "f", "f", "f"], 
                   display=True, 
                   dataset_name="simcol3d",
                   port=8097,  # Updated to match 'port'
                   display_id=100,  # Updated to match 'display_id'
                   display_freq=1, #80, 
                   print_freq=100, 
                   lr = 0.001,
                   checkpoints='checkpoints', 
                   save_samples=True, 
                   save_checkpoint_freq=1,  # Updated to match 'save_ckpt_freq'
                   scale_to_mm = 326.4, #devide by scale_to_mm
                   max_distance = 255,
                   val_freq= 1280, #2560, 
                   not_save_val_model=True, 
                   model='depth',  # Updated to match 'model'
                   net_architecture='D3net_multitask',  # Updated to match 'net_architecture'
                   workers=2, 
                   use_resize=False,
                   
                   kernel_size=51,
                   scale=3
                   )

# train model
if __name__ == '__main__':
    if opt.train or opt.resume:
        from models.mtl_train import MultiTaskGen as Model
        model = Model()
        model.initialize(opt)
        data_loader, val_loader = CreateDataLoader(opt)
        model.train(data_loader, val_loader=val_loader)
    elif opt.test:
        from models.mtl_test import MTL_Test as Model
        model = Model()
        model.initialize(opt)
        model.test()
