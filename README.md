# Few-shot Generalized Category Discovery (ICMR 2025)
Official PyTorch Implementation of Few-shot Generalized Category Discovery (FSGCD).

## Get Started
### Dependencies
Code is tested on Linux with **PyTorch 1.10.0** and **CUDA 11.3**.

```bash
pip install -r requirements.txt
```

### Datasets
We recommend creating a folder 'datasets/' to store all the dataset files.

#### CIFAR10

Use pytorch built-in datasets in [CIFAR-10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10).

Or organize downloaded files as:
```
datasets
|---cifar-10-batches-py
|   |---batches.meta
|   |---data_batch_1
|   |---...
```

#### CIFAR100

Use pytorch built-in datasets in [CIFAR-100](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR100.html#torchvision.datasets.CIFAR100).

Or organize downloaded files as:
```
datasets
|---cifar-100-python
|   |---file.txt~
|   |---meta
|   |---...
```


#### ImageNet

Download ImageNet from [ImageNet](https://image-net.org/download.php). Then organize the validation set according to categories, referring to this [Script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh).

The organized dataset should be like:
```
datasets
|---ImageNet
|   |---train
|   |   |---n01440764
|   |   |---n01443537
|   |   |---...
|   |---val
|   |   |---n01440764
|   |   |---n01443537
|   |   |---...
```


#### CUB
Download CUB_200_2011 from [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/).

The organized dataset should be like:
```
datasets
|---CUB_200_2011
|   |---attributes
|   |   |---certainties.txt
|   |   |---...
|   |---images
|   |   |---001.Black_footed_Albatross
|   |   |---...
|   |---parts
|   |   |---part_click_locs.txt
|   |   |---...
|   |---bounding_boxes.txt
|   |---...
```

#### Stanford-Cars
The organized dataset should be like:
```
datasets
|---stanford_cars
|   |---cars_test
|   |   |---00001.jpg
|   |   |---...
|   |---cars_train
|   |   |---00001.jpg
|   |   |---...
|   |---devkit
|   |   |---cars_meta.mat
|   |   |---cars_test_annos_withlabels.mat
|   |   |---...
```

#### Herbarium19
Download small resized dataset of Herbarium Challenge 2019 - FGVC6 from [Herbarium19](https://www.kaggle.com/c/herbarium-2019-fgvc6).

The organized dataset should be like:
```
datasets
|---herbarium_19
|   |---small-test
|   |   |---00001.jpg
|   |   |---...
|   |---small-train
|   |   |---0
|   |   |   |---00000.jpg
|   |   |   |---...
|   |   |---...
|   |---small-validation
|   |   |---0
|   |   |   |---00000.jpg
|   |   |   |---...
|   |   |---...
```

### Pre-trained weights convertion
You can either:

* Download 'dino_vitbase16_pretrain.pth' from [DINO](https://github.com/facebookresearch/dino?tab=readme-ov-file), modify the .pth path in 'FSGCD/convert_weight.py' accordingly, and run the 'FSGCD/convert_weight.py' script to get modified pre-trained weights.

* Or download from this link: [weights](https://pan.baidu.com/s/1y0nQgASUDkNMQfZ2ZaNZFQ?pwd=sygz).


### Config Modification
* Modify paths in 'FSGCD/config.py' to match your datasets and pre-trained weights.

* Modify settings of bash files in 'FSGCD/bash_scripts/'.



## Run
Our code has been successfully tested on **1x RTX 3090** and **1x RTX 4090 GPU**.

To run our code:
```
# Replace {dataset} accordingly.
bash bash_scripts/{dataset}.sh
```

## Results
| **Dataset**       | **All** | **Old** | **New** |
|---------------|------------|---------------|-----------|
| CIFAR10 | 95.0 | 97.4 | 94.5 |
| CIFAR100 | 71.3 | 73.4 | 71.2 |
| ImageNet100 | 81.1 | 92.5 | 79.9 |
| CUB | 44.7 | 45.4 | 44.6 |
| Stanford Cars | 12.3 | 15.0 | 12.2 |
| Herbarium 19 | 15.5 | 14.7 | 15.6 |


## Acknowledgements
This implementation builds upon [GCD](https://github.com/sgvaze/generalized-category-discovery). We sincerely appreciate their contributions.

## Cite this paper
```
@inproceedings{ren2025fsgcd,
  title={Few-Shot Generalized Category Discovery With Retrieval-Guided Decision Boundary Enhancement},
  author={Ren, Yunhan and Luo, Feng and Huang, Siyu},
  booktitle={The 15th ACM International Conference on Multimedia Retrieval (ICMR)},
  year={2025},
}
```
