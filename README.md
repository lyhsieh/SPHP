# SPHP: Sparse and Privacy-enhanced Representation for Human Pose Estimation
### [Project Page](https://lyhsieh.github.io/sphp/) | [Video](https://youtu.be/BdwL34Bd7e8?si=Pp-VUgmPCV9UH_wS) | [Paper](https://arxiv.org/abs/2309.09515) | [Data](https://forms.gle/wsfpLX6g7A1FDz5y5)

<br>

[Sparse and Privacy-enhanced Representation for Human Pose Estimation](https://lyhsieh.github.io/sphp/)  
 [Ting-Ying Lin]()\*<sup>1</sup>,
 [Lin-Yung Hsieh](https://lyhsieh.github.io/)\*<sup>1</sup>,
 [Fu-En Wang](https://fuenwang.phd/)<sup>1</sup>,
 [Wen-Shen Wuen]()<sup>2</sup>,
 [Min Sun](https://aliensunmin.github.io/)<sup>1</sup>,
 <br>
 <sup>1</sup>National Tsing Hua University, <sup>2</sup>Novatek Microelectronics Corp.
  \*denotes equal contribution  
in BMVC 2023

<div align="center">
<img src="img/sphp.gif">
</div>



## Setup

1. We recommend using [Anaconda](https://www.anaconda.com/download) to setup the environment.

    ```bash
    conda create --name sphp python=3.7
    ```

1. Install `torch`, `torchvision`, `torchaudio` from [PyTorch official site](https://pytorch.org/get-started/locally/) according to your CUDA version.

1. Modify dataset path in `config.yaml`. Change line 4 and line 11 to your path.

    ```yaml
    dataset_path: &dataset_path "PATH_TO_LABEL_DATA"    # Line 4
    calib_path: &calib_path "PATH_TO_calibrate.npy"    # Line 11
    ```

1. Install other required libraries in `requirements.txt`.
    ```bash
    pip install -r requirements.txt
    ```

1. If you want to utilize submanifold sparse convolution, please follow the setup instructions at [facebookresearch/SparseConvNet](https://github.com/facebookresearch/SparseConvNet) to install sparse convolution.

## Dataset Download

1. Fill out the [Google Form](https://forms.gle/wsfpLX6g7A1FDz5y5 ) and you will receive an agreement. 

1. Sign the agreement and send it back to us. Then we will share you the link of SPHP dataset. (It may take a few days.)

1. Download the dataset and put `Master.tar.gz` and `Slave.tar.gz` under the `data` folder.

1. Unzip the dataset. 
    ```bash
    cd data
    tar zxvf Master.tar.gz
    tar zxvf Slave.tar.gz
    ```
    Now, it should look like this. 
    ```bash
    .
    ├── data
    │   ├── calibrate.npy
    │   ├── Master
    │   │   ├── s01
    │   │   │   ├── 01
    │   │   │   │   ├── EDG 
    │   │   │   │   │   └── contains 300 png files (edge images)
    │   │   │   │   ├── MVH 
    │   │   │   │   │   └── contains 300 png files (horizontal motion vector)
    │   │   │   │   ├── MVV 
    │   │   │   │   │   └── contains 300 png files (vertical motion vector)
    │   │   │   │   └── pose_change 
    │   │   │   │       └── contains 300 npy files (ground truth labels)
    │   │   │   ├── 02
    │   │   │   ├── ...
    │   │   │   └── 16
    │   │   ├── s02
    │   │   ├── ...
    │   │   └── s16
    │   └── Slave
    ├── template
    │   ├── ...
    │   └── ...
    └── Utils
        ├── ...
        └── ...
    ```
    
    The file structure in `Slave` should be the same to `Master`.

## Training

### Basic usage

1. Choose a specification in `template` folder. The folders end with `_submanifold` utilize sparse convolution. Other folders use traditional convolution. Take edge modality under Unet backbone using sparse convolution as example.

    ```bash
    cd template/SPHP_Unet_edge_submanifold/
    ```

1. Run `main.py`
    ```bash
    python main.py --mode train
    ```

### Parser arguments

* `--mode`: training or testing mode, default mode is training.

* `--batch_size`: set batch size, default batch size is 32.

* `--device`: number of available GPUs, default value is 1.

    <details> <summary>Train on particular GPU</summary>

    To train on a particular GPU, insert the `CUDA_VISIBLE_DEVICES` before executing the command. Ensure consistency with the `--device` configuration.

    ```bash
    CUDA_VISIBLE_DEVICES=0,1,3 python main.py --mode train --device 3 --batch_size 64
    ```

    </details>

## Testing

1. Modify `config.yaml`. Change validation subject in line 22 to test subject in line 23.

    ```yaml
    # Line 22 (use these subject for training)
    subject: ['s06','s07','s08','s16','s17','s18','s26','s27','s28','s36','s37','s38'] 
    
    # Line 23 (use these subject for testing)
    subject: ['s09','s10','s19','s20','s29','s30','s39','s40'] 
    ```

1. Run `main.py`
    ```bash
    python main.py --mode val
    ```

## Citation
  >
    @inproceedings{lin2023sparse,
        title     = {Sparse and Privacy-enhanced Representation for Human Pose Estimation},
        author    = {Lin, Ting-Ying and Hsieh, Lin-Yung and Wang, Fu-En and Wuen, Wen-Shen and Sun, Min},
        booktitle = {British Machine Vision Conference (BMVC)},
        year      = {2023},
    }