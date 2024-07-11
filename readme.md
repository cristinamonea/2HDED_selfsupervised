# 2HDED:NET self-supervised training script

## Environment configuration

To set up the environment, run the following commands:
```
conda create --name 2hded-env
conda activate 2hded-env

conda install conda-forge::tqdm
conda install visdom (le 06.03.24 - pip install visdom)
pip install -U scikit-learn

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #(cuda toolkit driver from nvidia v10.2?)
```

## Settings sourse directories

Change the images's sourse directory from [Dataset bank file](data loader/dataset_bank.py)
The folder structure of the dataset should be as follows:

<pre>
\datasets
    \<dataset_name>
        \rgb_F1
        \rgb_F2
        ...
        \aif
        \depth
</pre>

## Setting the loss functions for training

The loss functions for training are defined in \_train_batch method from [models/mtl_train.py](models/mtl_train.py)
The same loss function should be used for validation, in the method get_eval_error from [models/mtl_train.py](models/mtl_train.py)

### Saving results

The results can be saved as PNG or as NumPy arrays (TODO: save arrays of float16 instead of float32 for storage efficiency).

The code for saving images can be found in the test method in [models/mtl_test](models/mtl_test)

## Visualisation of results while training

In the \_train_batch method, you can set the images and errors to be displayed using Visdom by:
self.set_current_visuals(<key>=<image:torch.Tensor>.data) and self.set_current_errors(<key>=<error:torch.Tensor>.item())

For evaluation, use:
self.val_current_visuals.update([(<indentifier:string>, <image:torch.Tensor>)]) and self.val_errors.update([("<identifier:string>", <error:torch.Tensor>.item() / len(data_loader))])
