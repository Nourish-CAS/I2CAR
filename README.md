# Overview of the I2CAR framework
<img width="1115" height="351" alt="_cgi-bin_mmwebwx-bin_webwxgetmsgimg__ MsgID=1525722861549640544 skey=@crypt_576c8fe3_00332332b6a3f1de86231dab982da62c mmweb_appid=wx_webfilehelper" src="https://github.com/user-attachments/assets/68c5b31d-08b8-4f81-a39a-0ff74e12eaee" />


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
2. Download data. You can obtain all benchmarks from [Google Cloud](https://drive.google.com/drive/folders/1RaIJQ8esoWuhyphhmMaH-VCDh-WIluRR?usp=sharing). All the datasets are well pre-processed.
3. Train and evaluate. We provide the experiment scripts of all benchmarks under the folder ```./scripts```. You can reproduce the experiment results as follows:

```bash
bash ./scripts/SMD.sh
bash ./scripts/MSL.sh
bash ./scripts/SMAP.sh
bash ./scripts/PSM.sh
bash ./scripts/SWAT.sh

## Acknowledgement
We appreciate the following github repos a lot for their valuable code:

https://github.com/thuml/Anomaly-Transformer

https://github.com/ahstat/affiliation-metrics-py

https://github.com/DAMO-DI-ML/KDD2023-DCdetector
