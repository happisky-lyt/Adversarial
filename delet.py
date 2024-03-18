import os

image_path = r'E:\python\train_patch\runs'


def get_filelist(dir):
    newDir = dir
    if os.path.isfile(dir):
        # if os.path.getsize(newDir) < 1000:
        #     os.remove(newDir)  # 删除小于1000字节的文件
        for s in os.listdir(dir):  # 遍历文件夹下所有文件
            newDir = os.path.join(dir, s)
            if newDir.endswith('.DESKTOP-8F8ADNP'):
                os.remove(newDir)
            get_filelist(newDir)  # 递归
    elif os.path.isdir(dir):
        for s in os.listdir(dir):  # 遍历文件夹下所有文件
            newDir = os.path.join(dir, s)
            if newDir.endswith('.DESKTOP-8F8ADNP'):
                os.remove(newDir)
            get_filelist(newDir)  # 递归


get_filelist(image_path)