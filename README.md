<p align="center">
  <img src="static/images/logo-cvn.png"
       style="width:180px; height:100px; object-fit:contain; background:white;
              border: 2px solid #ccc; border-radius: 6px; padding: 6px; margin: 0 12px;">
  <img src="static/images/logo-cs.png"
       style="width:180px; height:100px; object-fit:contain; background:white;
              border: 2px solid #ccc; border-radius: 6px; padding: 6px; margin: 0 12px;">
  <img src="static/images/universite-paris-saclay-logo.png"
       style="width:180px; height:100px; object-fit:contain; background:white;
              border: 2px solid #ccc; border-radius: 6px; padding: 6px; margin: 0 12px;">
  <img src="static/images/logo-inria.png"
       style="width:180px; height:100px; object-fit:contain; background:white;
              border: 2px solid #ccc; border-radius: 6px; padding: 6px; margin: 0 12px;">
</p>

# U-P3MG
The goal of the project is to optimize the hyperparameters of the P3MG algorithm using a deep learning approach.

---

### **A. Overview**
This repository contains the implementation of the U-P3MG model, a deep learning approach for spectroscopîc signal reconstruction in analytical chemistry. The model is designed to improve the quality of reconstitution of already established algorithm P3MG.

This repository includes tools for:
- Data preprocessing and augmentation
- Model training and evaluation
- Visualization and analysis of results

---

### **B. Getting Started**

#### **1. Clone the Repository**
```bash
git clone https://github.com/mehdikalla/U-P3MG.git
cd U-P3MG
```

#### **2. (Recommended) Install Mamba**
If Conda is not installed, use **Mambaforge** — it’s faster and fully compatible.

| Operating System | Architecture          | Installer                                                                                                                        |
| ---------------- | --------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Linux            | x86_64 (amd64)        | [Miniforge3-Linux-x86_64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh)       |
| macOS            | x86_64                | [Miniforge3-MacOSX-x86_64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-x86_64.sh)     |
| macOS            | arm64 (Apple Silicon) | [Miniforge3-MacOSX-arm64.sh](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh)       |
| Windows          | x86_64                | [Miniforge3-Windows-x86_64.exe](https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Windows-x86_64.exe) |

After installation, verify it works:

```bash
mamba --version
```

#### **3. Create the Environment**

```bash
mamba env create -f env.yml
```

#### **4. Activate the Environment**

```bash
conda activate U-P3MG
```

---

### **C. Dataset**

#### **1. Simulated Data**


---

### **D. Training and Evaluation**
---

#### **1. Configuration File**
The config.yaml file allows precise modification of the neural network and traditional algorithms hyperparameters (learning rate, epochs, number of layers and iterations ...).


#### **2. Train the Model **
To train the neural network models, you can run the following scripts :

```bash
# Train the P3MG unrolled model
./scripts/run.sh --config config.yaml --gpu 0 --train --model p3mg --strategy unrolling

# Train the ISTA unrolled model
./scripts/run.sh --config config.yaml --gpu 0 --train --model ista --strategy unrolling
```

#### **3. Evaluate the Model**
To evaluate a trained model, use the ```--test``` flag. Ensure that the checkpoint parameter in your config.yaml points to the correct trained weights.

```bash
# Evaluate the P3MG model
./scripts/run.sh --config config.yaml --gpu 0 --test --model p3mg --strategy unrolling

# Evaluate the ISTA model
./scripts/run.sh --config config.yaml --gpu 0 --train --model ista --strategy unrolling

```

#### **4. Run Random searches**
To run traditional iterative baseline algorithms, use the `random_search` strategy. This mode performs an algorithmic search over hyperparameters (such as Lambda and Tau) instead of training a neural network.

```bash
# Run random search for the P3MG model
./scripts/run.sh --config config.yaml --gpu 0 --full --model p3mg --strategy random_search

# Run random search for the ISTA model
./scripts/run.sh --config config.yaml --gpu 0 --full --model ista --strategy random_search
```

---
### **E. Results and Visualization**
---
#### **1. Visualize Results**
All generated plots and visualizations are automatically saved in the `plots/` subdirectory within your specific run folder (e.g., `./Results/p3mg_unrolling_<timestamp>/plots/`). 

Depending on the chosen mode and strategy, you will find:
* **`loss.png`**: Training and validation loss curves over epochs.
* **`best_sig.png` & `test_*_MSE.png`**: Visual comparisons between the true signal and the predicted signal. The framework automatically plots the Best, Worst, Median, and Mean reconstruction samples.
* **`learnt_lambda_curve.png` & `learnt_tau_heatmap.png`**: Evolution of the learned parameters (step sizes and thresholds) across the network layers. This is specific to the `unrolling` strategy and allows you to interpret the network's behavior.
* **`test_error_distribution.png`**: An histogram displaying the distribution of the chosen evaluation metric (MSE, SNR, or TSNR) across the entire test dataset.

#### **2. Analyze Performance**
Quantitative performance metrics are computed and saved during the evaluation phase :

* **Test Results Table (`test_results_table.txt`)**: Located in the `logs/` directory, this file provides a comprehensive statistical summary of the model's performance on the test set.

* **Oracle Statistics (`oracle_stats.json`)**: When running the `random_search` mode, this file records the absolute best theoretical performance and the optimal $( \lambda, \tau )$ hyperparameters for each individual sample in the test set.

---

### **Licensing**

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

### **Citation**

If you find this work useful in your research, please consider citing:

```
Coming soon...
```
