# HiSTGNN
This is a Pytorch implementation of the paper: HiSTGNN: Hierarchical Graph Neural Networks for Weather Forecasting.

## Running requirements
The basic dependencies are Python 3 and Torch 1.2.0. The others are specified in requirements.txt.

You can run the following commands to prepare the environment. We recommend the virtual conda environment.

```
conda create --name [myenv]
source activate [myenv]
pip install -r ./HiSTGNN/requirements.txt
```

## Data
We place the used data in [Google driver](https://drive.google.com/drive/folders/1OeadhQKI7a2aQuOAX5EzAXcwOUSZV92f?usp=sharing). You can download them and then put them in the following corresponding paths.

1. "./HiSTGNN/data/wfd_BJ"
2. "./HiSTGNN/data/wfd_Israel"
3. "./HiSTGNN/data/wfd_USA"


## Model Training
### wfd_BJ
```
bash BJ_train.sh
```
### wfd_Israel
```
bash ISR_train.sh
```
### wfd_USA
```
bash USA_train.sh
```

If you find our work helpful for your research, please consider citing our paper. :-)
```bibtex
@article{ma2023histgnn,
  title={HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting},
  author={Ma, Minbo and Xie, Peng and Teng, Fei and Wang, Bin and Ji, Shenggong and Zhang, Junbo and Li, Tianrui},
  journal={Information Sciences},
  volume={648},
  pages={119580},
  year={2023}
}
