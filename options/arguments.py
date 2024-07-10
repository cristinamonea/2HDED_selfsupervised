class TrainOptions():
    def __init__(self, dataroot, name, imageSize=[256], outputSize=0, train=False, test=False, visualize=False, 
                 test_split='test', train_split='train', input_nc=3, output_nc=1, cuda=True, 
                 nEpochs=350,  batchSize=16, resume=False, epoch='latest', mask_thres=0.0, 
                 update_lr=False, init_method='normal', use_cudnn_benchmark=False, data_augmentation=["t", "f", "f", "f", "t"], 
                 display=False, port=8097, display_id=0, display_freq=50, print_freq=50, lr=0.0002, beta1=0.5, beta2=0.999,
                 weightDecay=0.0, optim='Adam', momentum=0.9, niter_decay=100, checkpoints='./checkpoints', 
                 save_samples=False, save_checkpoint_freq=30, validate=False, val_split='val', 
                 val_freq=1000, not_save_val_model=True, pretrained=False, pretrained_path='no_path', 
                 model='regression', net_architecture='DenseUNet', d_block_type='basic',
                 use_skips=False, use_dropout=False, workers=2, annotation=False, use_resize=False, use_crop=False, 
                 use_padding=False, padding=[12, 12, 1, 0], dataset_name='nyu', scale_to_mm=0.0, max_distance=10.0):

        self.dataroot = dataroot
        self.name = name
        self.imageSize = imageSize
        self.outputSize = outputSize
        self.train = train
        self.test = test
        self.visualize = visualize
        self.test_split = test_split
        self.train_split = train_split
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.cuda = cuda
        self.nEpochs = nEpochs
       # self.nThreads = nThreads
        self.batchSize = batchSize
        self.resume = resume
        self.epoch = epoch
       # self.no_mask = no_mask
        self.mask_thres = mask_thres
        self.update_lr = update_lr
        self.init_method = init_method
        self.use_cudnn_benchmark = use_cudnn_benchmark
        self.data_augmentation = data_augmentation
        self.display = display
        self.port = port
        self.display_id = display_id
        self.display_freq = display_freq
        self.print_freq = print_freq
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.weightDecay = weightDecay
        self.optim = optim
        self.momentum = momentum
        self.niter_decay = niter_decay
        self.checkpoints = checkpoints
        self.save_samples = save_samples
        self.save_checkpoint_freq = save_checkpoint_freq
        self.validate = validate
        self.val_split = val_split
        self.val_freq = val_freq
        self.not_save_val_model = not_save_val_model
        self.pretrained = pretrained
        self.pretrained_path = pretrained_path
        self.model = model
        self.net_architecture = net_architecture
        self.d_block_type = d_block_type
        self.use_skips = use_skips
        self.use_dropout = use_dropout
        self.workers = workers
        self.annotation = annotation
        self.use_resize = use_resize
        self.use_crop = use_crop
        self.use_padding = use_padding
        self.padding = padding
        self.dataset_name = dataset_name
        self.scale_to_mm = scale_to_mm
        self.max_distance = max_distance
