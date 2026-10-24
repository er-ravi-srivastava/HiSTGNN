# HiSTGNN: Hierarchical Spatio-temporal Graph Neural Network for Weather Forecasting

This repository contains the PyTorch implementation and reproduction package for the paper:
> **HiSTGNN: Hierarchical Spatio-temporal Graph Neural Network for Weather Forecasting**  
> *Information Sciences*, Volume 648, 2023, 119580  
> Minbo Ma, Peng Xie, Fei Teng, Bin Wang, Shenggong Ji, Junbo Zhang, Tianrui Li

---

## 📊 Benchmark Results (Table 3: WD_BJ Performance)

| Methods | Metrics | TEMP | HUM | WIND | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HiSTGNN (Paper)** | **MAE** | **1.9533** | **9.8089** | **0.8671** | **4.2098** |
| | **RMSE** | **2.6353** | **14.0129** | **1.2634** | **5.9705** |
| | **MAPE (%)** | **32.49%** | **23.80%** | **52.40%** | **36.23%** |

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
conda create --name histgnn python=3.10 -y
conda activate histgnn

# Install PyTorch (CUDA or CPU)
pip install torch torchvision

# Dependencies
pip install numpy scipy pandas scikit-learn matplotlib
```

### 2. Download Dataset
Run the included download script to automatically fetch `test.dict`, `val.dict`, and `train.dict` from the author's public Google Drive storage directly into `HiSTGNN/data/wfd_BJ`:
```bash
python HiSTGNN/download_data.py
```

### 3. Reproduce Table 3 Results
To train the model on the `wfd_BJ` dataset and evaluate test performance:
```bash
cd HiSTGNN
python main.py \
  --device cuda:0 \
  --data data/wfd_BJ \
  --data_name BJ \
  --save ./save/BJ/ \
  --epochs 100 \
  --layers 3 \
  --gcn_depth 2 \
  --num_var 9 \
  --var_nodes 9 \
  --stat_nodes 10 \
  --var_node_dim 20 \
  --stat_node_dim 20 \
  --seq_in_len 28 \
  --seq_out_len 33 \
  --batch_size 64 \
  --tanhalpha 3 \
  --learning_rate 0.001 \
  --seed 101
```
*(On CPU-only machines, use `--device cpu`).*

---

## 🛠️ Reproduction Fixes Applied
1. **Target Variable Slicing (`util.py`)**: Fixed ground truth indexing from `:1` to `:3` to evaluate all three target metrics (`TEMP`, `HUM`, `WIND`) reported in Table 3.
2. **Output Kernel Size (`main.py`)**: Corrected output convolution kernel dimension `conv_k_size = (1, 7, 1)` to match `3` prediction heads instead of collapsing to `1`.
3. **Graceful Imports (`process_meteo.py`, `util.py`)**: Guarded optional libraries (`netCDF4`, `h5py`) to prevent import crashes during benchmark reproduction.
4. **Automated Data Downloader (`download_data.py`)**: Added direct 1-click Google Drive fetcher for seamless reproduction.

---

## 📖 Citation
```bibtex
@article{ma2023histgnn,
  title={HiSTGNN: Hierarchical spatio-temporal graph neural network for weather forecasting},
  author={Ma, Minbo and Xie, Peng and Teng, Fei and Wang, Bin and Ji, Shenggong and Zhang, Junbo and Li, Tianrui},
  journal={Information Sciences},
  volume={648},
  pages={119580},
  year={2023}
}
```
