# Wasserstein distance-based auto-encoder tracking
This repository is based on the paper "Wasserstein distance-based auto-encoder tracking", under reviewing in the journal NEPL.

![image](https://github.com/wahahamyt/CAT/blob/master/data/Bird1.gif)

## Environment
- Anaconda
- Pytorch
- visdom
- Agumentor

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
    - 2048 z Dimension (2.3GB)：https://pan.baidu.com/s/1yEFpTF9oOHFZtXAedxr0Ow Code: ```91pn```
    - 256 z Dimension (493MB): https://pan.baidu.com/s/1zMhTGUIYcqroXDKUFPvBqg Code: ```7t41```
- Google Drive: 
    - 2048 z Dimension (2.3GB)：https://drive.google.com/file/d/1g1GMVQeEKSiBboLDm26qArRP5_g8O-dO/view?usp=sharing
    - 256 z Dimension (493MB): https://drive.google.com/file/d/1DeGPatiyGh52TWl4O7cOrdMZCFvDEvdT/view?usp=sharing

After downloaded the weights, it should be renamed as ```last```, then moved to the folder ```net/weights/```.

### Start tracking:
Pycharm is recommend for avoiding some path issues.
```shell
tracking/run_tracker.py
```
## Evaluation
OTB protocal: http://cvlab.hanyang.ac.kr/tracker_benchmark/datasets.html

LaSOT protocal: https://github.com/HengLan/LaSOT_Evaluation_Toolkit

TC128 protocal: http://www.dabi.temple.edu/~hbling/data/TColor-128/TColor-128.html
