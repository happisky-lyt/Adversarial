"""
此程序输出一个谢尔宾斯基地毯的图片
2020年3月23日
by littlefean
"""
from PIL import Image
from time import time


def put_color(img, x_p, y_p, long, color):
    """
    填充一定范围内的颜色
    :param x_p: 填充正方形的x位置
    :param y_p: 填充正方形的y位置
    :param img: 图片名
    :param long: 正方形的边长
    :param color: 填充颜色
    :return:
    """
    if long == 0:
        img.putpixel((x_p, y_p), color)
    for n in range(long):
        for m in range(long):
            img.putpixel((x_p + n, y_p + m), color)


if __name__ == '__main__':
    t1 = time()
    Level = 9
    im = Image.new('RGB', (3 ** Level, 3 ** Level), 'black')
    for i in range(Level):
        k = 0  # 换行计数
        px = 3 ** i
        py = 3 ** i
        for j in range(9 ** (Level - 1 - i)):
            put_color(im, px, py, int(3 ** i), (255, 255, 255))
            k += 1
            px += int(3 ** (i + 1))
            if k == int(3 ** (Level - 1 - i)):
                # 横向归零，竖向下一
                px = int(3 ** i)
                py += int(3 ** (i + 1))
                k = 0
    im.save(f'mat_{Level}.png')
    t2 = time()
    print(t2 - t1)
    input()