# Eval Code for [CleanDIFT](https://github.com/CompVis/cleandift) in the setting of Telling Left from Right (GeoAware-SC)
This is a fork of their [original repo](https://github.com/Junyi42/GeoAware-SC) that includes the changes required to load our CleanDIFT checkpoints and perform evaluations.

> [!IMPORTANT]  
> The GeoAware-SC repo does not contain licensing information, making the licensing state of all the code taken from it unclear.


# Shortened version of the original Readme with modifications for CleanDIFT

## Environment Setup

To install the required dependencies, use the following commands:

```bash
conda create -n geo-aware python=3.9
conda activate geo-aware
conda install pytorch=1.13.1 torchvision=0.14.1 pytorch-cuda=11.6 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.6.1" libcusolver-dev
git clone git@github.com:Junyi42/GeoAware-SC.git 
cd GeoAware-SC
pip install -e .
```

PS: There are some common issues when installing Mask2Former. You might find [this issue](https://github.com/Junyi42/sd-dino/issues/11) helpful if you encounter any problems.

(Optional) You may want to install [xformers](https://github.com/facebookresearch/xformers) for efficient transformer implementation (which can significantly reduce the VRAM consumpution):

```
pip install xformers==0.0.16
```

You also want to install [SAM](https://github.com/facebookresearch/segment-anything) to extract the instance masks for adaptive pose alignment technique:

```
pip install git+https://github.com/facebookresearch/segment-anything.git
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

## Get Started

### Prepare the data

We provide the scripts to download the datasets in the `data` folder. To download specific datasets, use the following commands:

* SPair-71k: 
```bash
bash data/prepare_spair.sh
```

### Pre-extract the feature maps

To enable efficient evaluation, we pre-extract the feature maps of the datasets.


We provide a modified version of the feature extraction script that loads our released CleanDIFT weights.
First you need to download our weights and convert them from diffusers to the original ldm format. To do that, run:
```bash
python download_and_convert_model.py
```

Now you can run the feature extraction for SPair-71k using the CleanDift weights by running:

```bash
python preprocess_map.py ./data/SPair-71k/JPEGImages ./cleandift_sd15_unet_ldm.pt
```

For the SPair-71k dataset, it takes roughly 2 hours to extract the feature maps (for both the original and flipped images) on a single RTX 3090 GPU, and consumes around 90GB of disk space. 

### Pre-extract the instance masks

For the default adaptive pose alignment method which requires the source instance mask, we also pre-extract the masks of the dataset for efficiency. To do so, run the following commands:

* SPair-71k: 
```bash
python preprocess_mask_sam.py ./data/SPair-71k/JPEGImages
```


## Evaluation

In order to run the eval for zero-shot semantic correspondences on SPair-71k, execute:


```bash
python pck_train.py --config configs/eval_zero_shot_spair.yaml
```

This should yield the SOTA results we report in our [CleanDIFT paper](https://arxiv.org/pdf/2412.03439) in the zero-shot semantic correspondence section.


## Citation

If you find our work useful, please cite both our work on CleanDIFT as well as the original Telling Left from Right paper:

```BiBTeX
@misc{stracke2024cleandiftdiffusionfeaturesnoise,
      title={CleanDIFT: Diffusion Features without Noise}, 
      author={Nick Stracke and Stefan Andreas Baumann and Kolja Bauer and Frank Fundel and Björn Ommer},
      year={2024},
      eprint={2412.03439},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.03439}, 
}
@inproceedings{zhang2024telling,
  title={Telling Left from Right: Identifying Geometry-Aware Semantic Correspondence},
  author={Zhang, Junyi and Herrmann, Charles and Hur, Junhwa and Chen, Eric and Jampani, Varun and Sun, Deqing and Yang, Ming-Hsuan},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```

## Acknowledgement

Our code is largely based on the following open-source projects: [A Tale of Two Features](https://github.com/Junyi42/sd-dino), [Diffusion Hyperfeatures](https://github.com/diffusion-hyperfeatures/diffusion_hyperfeatures), [DIFT](https://github.com/Tsingularity/dift), [DenseMatching](https://github.com/PruneTruong/DenseMatching), and [SFNet](https://github.com/cvlab-yonsei/SFNet). Our heartfelt gratitude goes to the developers of these resources!
