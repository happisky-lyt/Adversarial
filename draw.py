import numpy as np
import torch
import cv2
import os
def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    """
    将tensor保存为cv2格式
    :param input_tensor: 要保存的tensor
    :param filename: 保存的文件名
    """
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    # 复制一份
    input_tensor = input_tensor.clone().detach()
    # 到cpu
    input_tensor = input_tensor.to(torch.device('cpu'))
    # 反归一化
    # input_tensor = unnormalize(input_tensor)
    # 去掉批次维度
    input_tensor = input_tensor.squeeze()
    # 从[0,1]转化为[0,255]，再从CHW转为HWC，最后转为cv2
    input_tensor = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    # RGB转BRG
    input_tensor = cv2.cvtColor(input_tensor, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, input_tensor)
def draw(tensor1, tensor2):

    # a[1:2][1:4] = 0
    # print(a)

    imgsize = tensor2.shape[-1]
    eta = torch.full((3, imgsize, imgsize),0.5)
    list1 = []
    list2 = []
    batchsize = tensor1.shape[0]
    for p in range(batchsize):
        T1 = tensor1.clone()
        t1 = T1[p, :, :]
        lab = t1[0]
        T2 = tensor2.clone()
        t2 = T2[p, :, :, :]
        mask = np.zeros((imgsize, imgsize))
        T = np.zeros((imgsize, imgsize))
        for q in range(t1.shape[0]):
            if q == 0:
                continue
            else:
                x = int(t1[q,1]*imgsize)
                y = int(t1[q,2]*imgsize)
                z = int(t1[q,3]*imgsize)
                w = int(t1[q,4]*imgsize)
            if x<imgsize or y<imgsize or z<imgsize or w<imgsize :
                for i in range(imgsize):
                    for j in range(imgsize):
                        if i >= (x - z * 0.5) and i <= (x + z * 0.5) and j >= (y - w * 0.5) and j <= (y + w * 0.5):
                            T[j][i] = 1
            mask = mask + T

        # mask1 = torch.from_numpy(mask).unsqueeze(0).repeat(3, 1, 1)
        # save_p_img_batch = os.path.join("E:/python/train_patch/saved_patches/", 'mask1' + ".png")
        # save_image_tensor2cv2(mask1.unsqueeze(0), save_p_img_batch)


        mask1 = torch.clamp(torch.from_numpy(mask).unsqueeze(0).repeat(3,1,1), 0, 1)
        etat = (eta * mask1).to('cuda:0')
        # save_p_img_batch = os.path.join("E:/python/train_patch/saved_patches/", 'etat' + ".png")
        # save_image_tensor2cv2(etat.unsqueeze(0), save_p_img_batch)
        mask2 = (torch.ones_like(mask1)-mask1).to('cuda:0')
        # save_p_img_batch = os.path.join("E:/python/train_patch/saved_patches/", 'mask' + ".png")
        # save_image_tensor2cv2(mask2.unsqueeze(0), save_p_img_batch)
        t2 = t2 * mask2 + etat
        # save_p_img_batch = os.path.join("E:/python/train_patch/saved_patches/", 't2' + ".png")
        # save_image_tensor2cv2(t2.unsqueeze(0), save_p_img_batch)
        list1.append(torch.tensor(t2, dtype=torch.float32).unsqueeze(0))
        list2.append(lab.unsqueeze(0).unsqueeze(0))
    img_batch = torch.cat(list1, dim=0)
    lab_batch = torch.cat(list2, dim=0)




        # T=np.zeros((480, 480))
        # x=int(t[0])
        # y=int(t[1])
        # z=int(t[2])
        # w=int(t[3])
        # center_x = int((x+z)*0.5)
        # center_y = int((w+y)*0.5)
        # radius = int((0.05*(center_y**2+center_y**2)**0.5))
        # Mask = generate_mask(480 ,480 , radius, center_x, center_y)
        # for i in range(480):
        #     for j in range(480):
        #
        #         if i>=(x+(z-x)*0.15) and  i <=(z-(z-x)*0.15) and j>=(y+(w-y)*0.15) and j<=(w-(w-y)*0.15):
        #             T[j][i] = 1
        # mask = mask + T
    return img_batch, lab_batch


# def generate_mask(img_height, img_width, radius, center_x, center_y):
#     y, x = np.ogrid[0:img_height, 0:img_width]
#
#     # circle mask
#
#     mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2
#
#     return mask



