# ZR<sup>2</sup>ViM 

**ZR<sup>2</sup>ViM: A Recursive Vision Mamba Model for Boundary-Preserving Medical Image Segmentation**

## Main Environments

```bash
conda create -n zr2vim python=3.10 -y
conda activate zr2vim

# GPU（示例：CUDA 11.8）
conda install pytorch==2.1.2 torchvision==0.16.2 pytorch-cuda=11.8 -c pytorch -c nvidia -y
# 仅 CPU
conda install pytorch==2.1.2 torchvision==0.16.2 cpuonly -c pytorch -y

pip install -r requirements.txt
```


## Prepare  Datasets

- After preparing the datasets, you are supposed to put them into './data/', and the file format reference is as follows. (take the ISIC17 dataset as an example.) 

```
data/ISIC17
│   ├── images/         
│   ├── labels/         
│   └── annotations/    
│       ├── train.txt   
│       └── test.txt    
```

- Ensure your image and label files are named correctly:
  * Image files：`[filename].jpg`
  * Label files：`[filename]_segmentation.png`
- Please ensure that the training and test set filename lists do not include file extensions.

## Train

Use the following command to begin training:

```bash
python train_ZR2ViM.py --dataset <dataset_name> --crop_size 256 256 --batch_size 16 --epochs 150
```

### Main parameter description

- `--dataset`: Dataset Name
- `--crop_size`: Cutting size during training
- `--batch_size`: Training batch size
- `--val_batch_size`: Verify batch size
- `--epochs`: Number of training rounds
- `--lr`: Learning rate
- `--output_dir`: Output directory

## Test

Use the following command to start the test:

```bash
python test_ZR2ViM.py --dataset <dataset_name> --data_dir <path_to_data> --checkpoint <path_to_checkpoint>
```