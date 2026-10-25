# HiSTGNN: Hierarchical Spatio-temporal Graph Neural Network for Weather Forecasting

This repository contains the PyTorch implementation and reproduction package for the paper:
> **HiSTGNN: Hierarchical Spatio-temporal Graph Neural Network for Weather Forecasting**  
> *Information Sciences*, Volume 648, 2023, 119580  
> Minbo Ma, Peng Xie, Fei Teng, Bin Wang, Shenggong Ji, Junbo Zhang, Tianrui Li

---

## 📊 Benchmark Datasets & Results

### 1. WD_BJ (Beijing Weather Dataset - Table 3)
* **Configuration:** 10 Stations, 9 Observed Variables $\rightarrow$ 3 Target Variables (`TEMP`, `HUM`, `WIND`)
* **Time Windows:** 28 historical timesteps (28h) $\rightarrow$ 33 forecast timesteps (33h)

| Methods | Metrics | TEMP | HUM | WIND | Avg. |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **HiSTGNN (Paper)** | **MAE** | **1.9533** | **9.8089** | **0.8671** | **4.2098** |
| | **RMSE** | **2.6353** | **14.0129** | **1.2634** | **5.9705** |
| | **MAPE (%)** | **32.49%** | **23.80%** | **52.40%** | **36.23%** |

### 2. WD_Israel (Table 4)
* **Configuration:** 6 Stations, 4 Variables (`TEMP`, `HUM`, `WIND`, `PRESSURE`)
* **Time Windows:** 48 historical timesteps (48h) $\rightarrow$ 24 forecast timesteps (24h)

### 3. WD_USA (Table 5)
* **Configuration:** 13 Stations, 4 Variables (`TEMP`, `HUM`, `WIND`, `PRESSURE`)
* **Time Windows:** 48 historical timesteps (48h) $\rightarrow$ 24 forecast timesteps (24h)

---

## 🚀 Quick Start & Reproduction

### 1. Environment Setup
```bash
conda create --name histgnn python=3.10 -y
conda activate histgnn

# Install PyTorch with CUDA support (or CPU)
pip install torch torchvision

# Core scientific packages
pip install numpy scipy pandas scikit-learn matplotlib
```

### 2. Automated Dataset Downloader
Download all three datasets (or specific ones) with a single command from the official Google Drive archive:
```bash
# Download ALL 3 datasets (BJ, Israel, USA)
python HiSTGNN/download_data.py --dataset all

# Or individually:
# python HiSTGNN/download_data.py --dataset bj
# python HiSTGNN/download_data.py --dataset israel
# python HiSTGNN/download_data.py --dataset usa
```

### 3. Running Reproduction for All 3 Datasets

#### A. Beijing Dataset (`wfd_BJ`)
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

#### B. Israel Dataset (`wfd_Israel`)
```bash
cd HiSTGNN
python main.py \
  --device cuda:0 \
  --in_dim 1 \
  --epochs 100 \
  --layers 2 \
  --data data/wfd_Israel \
  --data_name Israel \
  --save ./save/Israel/ \
  --gcn_depth 2 \
  --num_var 4 \
  --var_nodes 4 \
  --stat_nodes 6 \
  --var_node_dim 5 \
  --stat_node_dim 10 \
  --seq_in_len 48 \
  --seq_out_len 24 \
  --batch_size 32 \
  --tanhalpha 2 \
  --learning_rate 0.001 \
  --seed 101
```

#### C. USA Dataset (`wfd_USA`)
```bash
cd HiSTGNN
python main.py \
  --device cuda:0 \
  --in_dim 2 \
  --epochs 100 \
  --layers 2 \
  --data data/wfd_USA \
  --data_name USA \
  --save ./save/USA/ \
  --gcn_depth 2 \
  --num_var 4 \
  --var_nodes 4 \
  --stat_nodes 13 \
  --var_node_dim 5 \
  --stat_node_dim 10 \
  --seq_in_len 48 \
  --seq_out_len 24 \
  --batch_size 16 \
  --tanhalpha 2 \
  --patient 5 \
  --learning_rate 0.001 \
  --seed 101
```
*(For CPU-only testing on any dataset, replace `--device cuda:0` with `--device cpu` and adjust batch size).*

---

## 🛠️ Key Bug Fixes Applied for Reproduction
1. **Target Variable Slicing (`util.py`)**: Fixed `wfd_BJ` label slicing from `:1` to `:3` so that all 3 target variables (`TEMP`, `HUM`, `WIND`) are computed and matched to Table 3.
2. **Convolution Kernel Projection (`main.py`)**: Corrected output convolution kernel dimension `conv_k_size = (1, 7, 1)` for `wfd_BJ` to properly project 9 variables to 3 target heads.
3. **Israel Dataset Loader Robustness (`util.py`)**: Added adaptive file discovery for `Israel_data.dict` vs `new_Israel_data.dict` to prevent `FileNotFoundError`.
4. **Library Import Guards (`process_meteo.py`, `util.py`)**: Protected optional dependencies (`netCDF4`, `h5py`) so standard environments don't crash on import.
5. **Unified Multi-Dataset Downloader (`download_data.py`)**: Automated downloading for all 3 datasets directly from Google Drive.

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
