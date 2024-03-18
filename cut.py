import os
from shutil import copy2

# 原始路径
image_original_path = r"E:\python\yolov3\cowc-m\datasets\DetectionPatches_256x256\DetectionPatches_256x256\car\image"
label_original_path = "E:\python\yolov3\cowc-m\datasets\DetectionPatches_256x256\DetectionPatches_256x256\car\label"
# 上级目录
parent_path = os.path.dirname(os.getcwd())
print(parent_path)
# 训练集路径
train_image_path = os.path.join(parent_path, "image_data/seed/train/images/")
train_label_path = os.path.join(parent_path, "image_data/seed/train/labels/")
# 测试集路径
test_image_path = os.path.join(parent_path, 'image_data/seed/test/images/')
test_label_path = os.path.join(parent_path, 'image_data/seed/test/labels/')


# 检查文件夹是否存在
def mkdir():
    if not os.path.exists(train_image_path):
        os.makedirs(train_image_path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)

    if not os.path.exists(test_image_path):
        os.makedirs(test_image_path)
    if not os.path.exists(test_label_path):
        os.makedirs(test_label_path)


def main():
    mkdir()
    # 复制移动图片数据
    all_image = os.listdir(image_original_path)
    for i in range(len(all_image)):
        if i % 10 != 0:
            copy2(os.path.join(image_original_path, all_image[i]), train_image_path)
        else:
            copy2(os.path.join(image_original_path, all_image[i]), test_image_path)

    # 复制移动标注数据
    all_label = os.listdir(label_original_path)
    for i in range(len(all_label)):
        if i % 10 != 0:
            copy2(os.path.join(label_original_path, all_label[i]), train_label_path)
        else:
            copy2(os.path.join(label_original_path, all_label[i]), test_label_path)


if __name__ == '__main__':
    main()