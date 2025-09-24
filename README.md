[Figure2.pdf](https://github.com/user-attachments/files/22509272/Figure2.pdf)
# I²CAR: Intra- and Inter-Variate Consistency Contrastive Adversarial Representation Learning
I²CAR is a lightweight framework for multivariate time series anomaly detection, addressing spatiotemporal entanglement and noise robustness. 
It decouples temporal and variable dependencies, then leverages contrastive–adversarial representation learning to separate noise from anomalies and enhance detection accuracy

## 🔹 Overall
I²CAR is a lightweight framework for **multivariate time series anomaly detection**, addressing spatiotemporal entanglement and noise contamination.  
It decouples **temporal** and **variable** dependencies, and introduces a dual strategy to achieve robust detection.

## 🔹 Architecture
- **Temporal-view GNN** models channel-independent temporal consistency.  
- **Variable-view GNN** models inter-variable consistency.  
- **Cross-view alignment** highlights anomalies as disruptions between the two views.  

## 🔹 Optimization
- **Contrastive learning** enlarges the boundary between noise and anomalies within each view.  
- **Adversarial learning** further generalizes normal and noisy samples, strengthening robustness to noise.  
- The joint loss balances intra-view separation and inter-view alignment for stable optimization.

## 🔹 Performance & Justification
- Achieves **state-of-the-art F1 scores** across four benchmark datasets (MSL, SMAP, PSM, SWaT).  
- Demonstrates the **lowest F1 drop (F1-N)** under noisy conditions, proving strong noise robustness.  
- Maintains **lightweight complexity** with fewer parameters and FLOPs, suitable for edge deployment.  



## Code Description
There are ten files/folders in the source.

- data_factory: The preprocessing folder/file. All datasets preprocessing codes are here.
- dataset: The dataset folder, and you can download all datasets [here](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR?usp=sharing).
- main.py: The main python file. You can adjustment all parameters in there.
- scripts: All datasets and ablation experiments scripts. You can reproduce the experiment results as get start shown.
- solver.py: Another python file. The training, validation, and testing processing are all in there. 
- utils: Other functions for data processing and model building.


## Get Start
1. Install Python 3.6, PyTorch >= 1.4.0.
2. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder ```./scripts```. You can reproduce the experiment results as follows:

```bash
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/SWAT.sh

## Acknowledgement
We appreciate the following github repos a lot for their valuable code:

https://github.com/thuml/Anomaly-Transformer
https://github.com/ahstat/affiliation-metrics-py
https://github.com/DAMO-DI-ML/KDD2023-DCdetector
