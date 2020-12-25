# Fall-Detection-Network

### Introduction

This repo provides demo Pytorch implementation of the paper A Video-based Fall Detection Network by
Spatio-temporal Joint-point Model on Edge Devices. We extract and filtrate the time-series joint-point features to capture the spatial and temporal dynamics, which are subsequently fed to the tensorized spatio-temporal joint-point model to predict falls.

![](D:\学习资料\实验室相关文件\论文整理\示意图.png)

### Requirements

```
Pytorch >= 1.1.0
torchvision >= 0.3.0
visdom
nibabel
pandas
tqdm
matplotlib
opencv-python
ntpath
```

### Training

First step is preparing the training data, the fall detection dataset can be accessed by [Multiple cameras fall dataset](http://www.iro.umontreal.ca/~labimage/Dataset/) and [UR Fall Detection Dataset](http://fenix.univ.rzeszow.pl/~mkepski/ds/uf.html).

Second step is getting the frames by video clip at different FPS. (Recommend FPS higher than 8)

Then run the extracting codes for training.

```python
python demo_ectract.py
```

After getting the time-series joint-point features, we can feed them to the tensorized LSTM, which can been seen in [TT_LSTM/MULT_TT.py](TT_LSTM/MULT_TT.py), which can also attain the model file by optimizing the training parameters .
