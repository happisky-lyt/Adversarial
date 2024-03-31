# Adversarial Train

---
## <img src="images/logo_geochat.png" height="40">Overview

此项目是用于训练对抗样本的部分参考代码。为光学图像中的车辆目标生成贴纸形式的对抗样本，效果如图所示。

![img.png](img.png)

---
## Contents
- [Install](#install)
- [Train](#train)
- [Test](#test)
- [Other Excellent Repositories](#Other-Excellent-Repositories)
---
## Install

1. 克隆仓库到本地
```bash
git clone https://github.com/happisky-lyt/Adversarial.git
cd Adversarial
```

2. 创建虚拟环境
```Shell
conda create -n Adversarial python=3.8 -y
conda activate Adversarial
```

3. 安装依赖项
```
pip install -r requirements.txt  # 安装依赖项
```

---
## Train
### Training YOLO-V3 Detector
训练一个YOLO-V3车辆目标检测器。

**数据集**：[visdrone2019](https://github.com/VisDrone/VisDrone-Dataset)

- 注意：这里只训练车辆目标检测器，因此需要对数据集进行预处理，删除不包含车辆目标的图像和对应标签。

**关于yolo-v3的版本问题**：不同版本的yolo-v5主要是其中的部分函数有区别，因此可以使用最新版的yolo-v3训练检测器，但是后面训练对抗样本的训练需要采用相同版本的yolo-v3。

### Training Adversarial Examples
We train adversarial examples on NVIDIA GeForce RTX 3060 Laptop GPU with 6144MiB memory.

**训练数据**：存放在`train2`文件夹中。这里的数据是自己用无人机拍的。

**权重**：事先由yolo-v3在visdrone19中训练得到。

训练初始化设置在`patch_config.py`文件中。其中`BaseConfig.__init__()`中修改权重、训练集图片和标签地址、patchsize等信息。

训练对抗样本：
```
python train_patchv3_obj.py
```

---
## Test

测试初始化设置在`patch_config_test.py`文件中。

**测试数据**：存放在`test5`文件夹中。

测试对抗样本：
```
python test_patcher1_a.py
```
---
## Other Excellent Repositories

仅作参考！

[adversarial-yolo](https://github.com/KI-1-AI-Sec/adversarial-yolo)

[adversarial-yolov3-cowc](https://github.com/andrewpatrickdu/adversarial-yolov3-cowc)

[DPatch](https://github.com/veralauee/DPatch)
