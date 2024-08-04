# simplified main
from options.extra_args_mtl import MTL_Options as TrainOptions
from dataloader.data_loader import CreateDataLoader

# Load options
opt = TrainOptions(dataroot=r"C:\Users\ubuntu\Desktop\Cristina\git repos\2HDED_selfsupervised\datasets\iDFD",
                   name="idfd_test", 
                   imageSize=[1051], 
                   outputSize=[1024],
                   use_crop=False,
                   resume=False,
                   train=True, 
                   validate=True,
                   test=False,
                   epoch='latest', 
                   visualize=False, 
                   test_split='test', 
                   train_split='train', 
                   val_split='val',
                   
                   use_dropout=False,
                   use_skips=True,
                   cuda=True, 
                   nEpochs=2,  # Updated to match 'nepochs'
                   batchSize=4,  # Updated to match 'batch_size'
                   init_method='normal', 
                   data_augmentation=["f", "f", "f", "f", "f"], 
                   display=True, 
                   dataset_name="iDFD",
                   port=8097,  # Updated to match 'port'
                   display_id=100,
                   display_freq=1, #80, 
                   print_freq=100, 
                   lr = 0.001,
                   checkpoints='checkpoints', 
                   save_samples=True, 
                   save_checkpoint_freq=1,  # Updated to match 'save_ckpt_freq'
                   scale_to_mm = 1, #devide by scale_to_mm, for idfd dataset values are already in mm
                   max_distance = 10000, #for idfd dataset is 10 meters
                   val_freq= 4, #2560, 
                   not_save_val_model=True, 
                   model='depth',  # Updated to match 'model'
                   net_architecture='D3net_multitask',  # Updated to match 'net_architecture'
                   workers=2, 
                   use_resize=False,
                   
                   kernel_size=51,
                   pixel_size=0.006,
                   scale=1,
                   focal_lenght=14,
                   focal_depth=1600,
                   aperture=2.8,
                   aperture_2=10,

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
