# CAT
This repository is based on the paper "Wasserstein distance-based auto-encoder tracking", under reviewing in the journal NEPL.
## Environment
- Anaconda
- Pytorch
- visdom
```shell
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

## Run
You can train the Auto Encoder by yourself using:
https://github.com/wahahamyt/train_WAE.git. 

This code also integrated in this repository.
### Pretrained Auto_Encoder Weights
We also provided the pretrained auto encoder weights:
- Baidu Disk: 
    - 2048 z Dimension： https://pan.baidu.com/s/1yEFpTF9oOHFZtXAedxr0Ow Code: ```91pn```
    - 256 z Dimension: https://pan.baidu.com/s/1zMhTGUIYcqroXDKUFPvBqg Code: ```7t41```
- Google Drive: 
    - 2048 z Dimension： https://pan.baidu.com/s/1yEFpTF9oOHFZtXAedxr0Ow
    - 256 z Dimension: https://pan.baidu.com/s/1zMhTGUIYcqroXDKUFPvBqg

### Start tracking:
```shell
python tracking/run_tracker.py
```
## Evaluation
OTB protocal, LaSOT protocal, TC128 protocal

https://github.com/HengLan/LaSOT_Evaluation_Toolkit

http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html
