"""
Training code for Adversarial patch training


"""
import copy
import PIL
import numpy as np
import argparse


import load_data_test
from tqdm import tqdm
import cv2

from load_data_test import *
import gc
import matplotlib.pyplot as plt
from torch import autograd
from torchvision import transforms
from tensorboardX import SummaryWriter
import subprocess
from draw import draw
from FractalDimension import fractal_dimension
import grad_CAM
from yolov3.utils.loss import ComputeLoss

import weather_test
import patch_config_test
import sys
import time
from yolov3.models.experimental import attempt_load
from yolov3.utils.datasets import LoadStreams, LoadImages
from yolov3.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov3.utils.plots import Annotator, colors, save_one_box
from yolov3.utils.torch_utils import select_device

from yolov5.models.experimental import attempt_load
from yolov5.utils.datasets import LoadStreams, LoadImages
from yolov5.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, \
    apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov5.utils.plots import Annotator, colors, save_one_box
from yolov5.utils.torch_utils import select_device


def transbox(lab_batch):
    box1 = lab_batch[:, 0, 1:]
    box1 = box1 * 416
    box2 = box1.clone()
    box1[:, 0] = box2[:, 0] - 0.5 * box2[:, 2]
    box1[:, 1] = box2[:, 1] - 0.5 * box2[:, 3]
    box1[:, 2] = box2[:, 0] + 0.5 * box2[:, 2]
    box1[:, 3] = box2[:, 1] + 0.5 * box2[:, 3]
    return box1


def computeIOU(box1, box2):
    list = []
    # for i in range((box2.shape[0])):
    b1 = box1.unsqueeze(1).repeat(1, 10647, 1)
    b2 = box2
    b3 = b2.clone()
    b3[:, :, 0] = b2[:, :, 0] - 0.5 * b2[:, :, 2]
    b3[:, :, 1] = b2[:, :, 1] - 0.5 * b2[:, :, 3]
    b3[:, :, 2] = b2[:, :, 0] + 0.5 * b2[:, :, 2]
    b3[:, :, 3] = b2[:, :, 1] + 0.5 * b2[:, :, 3]
    # b1 = b1.repeat(10647, 1)
    left_column_max = 0.5 * ((b1[:, :, 0] + b3[:, :, 0]) + abs(b1[:, :, 0] - b3[:, :, 0]))
    right_column_min = 0.5 * ((b1[:, :, 2] + b3[:, :, 2]) - abs((b1[:, :, 2] - b3[:, :, 2])))
    up_row_max = 0.5 * ((b1[:, :, 1] + b3[:, :, 1]) + abs((b1[:, :, 1] - b3[:, :, 1])))
    down_row_min = 0.5 * ((b1[:, :, 3] + b3[:, :, 3]) - abs((b1[:, :, 3] - b3[:, :, 3])))
    a = torch.gt(left_column_max, right_column_min)
    b = torch.gt(up_row_max, down_row_min)
    b4 = b3[:, :, 0:4]
    x1 = b1[:, :, 2] - b1[:, :, 0]
    x2 = b1[:, :, 3] - b1[:, :, 1]
    x3 = b4[:, :, 2] - b4[:, :, 0]
    x4 = b4[:, :, 3] - b4[:, :, 1]
    S1 = (x1) * (x2)
    S2 = (x3) * (x4)
    S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    IOU = S_cross / (S1 + S2 - S_cross)
    c = IOU >= 0.8

    # xc = torch.gt(left_column_max, right_column_min) or torch.gt(up_row_max, down_row_min)
    # xc = 0.5*((b1[:, 0] + b3[: , 0]) + abs(b1[:, 0] - b3[: , 0])) >= 0.5 * ((b1[:, 2] + b3[:, 2]) - abs((b1[:, 2] - b3[:, 2]))) or 0.5*((b1[:,3] + b3[:, 3]) - abs((b1[:,3] - b3[:, 3]))) <= 0.5*((b1[:, 1] + b3[:, 1]) + abs((b1[:, 1] - b3[:, 1])))
    xc = (~(a | b)) & c
    for xi, x in enumerate(box2):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 10000 - x[xc[xi]].shape[0]))
        list.append(pad(x[xc[xi]]).unsqueeze(0))
    output = torch.cat(list, dim=0)
    return output
    #     list2 = []
    # list = []
    # b1 = box1.unsqueeze(1).repeat(1, 10647, 1)
    # b2 = box2
    # b3 = b2.clone()
    # c = (b1[:,:,2]**2 +b1[:,:,3]**2)*0.5
    # b4 = (b1[:,:,0]-b3[:,:,0])**2 + (b1[:,:,1]-b3[:,:, 1])**2
    # xc = torch.gt(c, b4)
    # for xi, x in enumerate(box2):  # image index, image inference
    #     # Apply constraints
    #     # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
    #     pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100-x[xc[xi]].shape[0]))
    #     list.append(pad(x[xc[xi]]).unsqueeze(0))
    # output = torch.cat(list, dim=0)
    # return output

    #     for box in b2:
    #         # a = b1[0].long()
    #         # b = box[0].long()
    #         # c = b1[1].long()
    #         # d = box[1].long()
    #         if ((b1[0].long()-box[0].long())**2 + (b1[1].long()-box[1].long())**2) <= 25:
    #         # left_column_max = torch.max(b1[0], box[0])
    #         # right_column_min = torch.min(b1[2], box[2])
    #         # up_row_max = torch.max(b1[1], box[1])
    #         # down_row_min = torch.min(b1[3], box[3])
    #         # if not(left_column_max >= right_column_min or down_row_min <= up_row_max):
    #             box = box.unsqueeze(0)
    #             list2.append(box)
    #     c = torch.cat(list2, dim=0)
    #     # ar = np.zeros((100, 6))
    #     pad = torch.nn.ZeroPad2d(padding=(0, 0, 0, 100-c.shape[0]))
    #     c = pad(c).unsqueeze(0)
    #     list1.append(c)
    # output = torch.cat(list1, dim=0)
    # return output

    # 两矩形有相交区域的情况

    # if torch.min(box1[2], box[2])-torch.max(box1[0], box[0]) > 0 and torch.min(box1[3], box[3])-torch.max(box1[1], box[1]) > 0:
    #     c1 = (torch.min(b1[2], box[2])-torch.max(b1[0], box[0])) * (torch.min(b1[3], box[3])-torch.max(b1[1], box[1]))
    #     c2 = (b1[2]-b1[0]) * (b1[3]-b1[1]) + (box[2]-box[0]) * (box[3]-box[1]) - c1
    #     if c1/c2 >0.8:
    #         return box2


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


def plot_detection(pred, savedir, save_p_img_batch, disc=None):
    colors = Colors()  # 用于画检测框
    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        if i > 0:
            break
        im = save_p_img_batch[0, :, :, :]  # 读取已经保存的对抗图像
        # im0 = img0.copy()
        im0 = (np.array(im.detach().cpu()) * 255).transpose(1, 2, 0).astype(np.uint8).copy()

        # im0 = im0.astype(int)
        save_path = os.path.join(savedir, '{}.png'.format(disc))
        # det = det.astype(int)
        annotator = Annotator(im0, line_width=1, example=str(['car']))
        if det != None:
            if len(det):  # 读取的是补全正方形的图片，故不需要对检测框进行缩放
                # Rescale boxes from img_size to im0 size
                # det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                # Write results
                for *xyxy, conf, cls in reversed(det):
                    c = int(cls)  # integer class
                    if c == 0:
                        label = f'car {conf:.2f}'
                        annotator.box_label(xyxy, label, color=colors(c, True))
            # Save results (image with detections)

            cv2.imwrite(save_path, cv2.cvtColor(im0, cv2.COLOR_RGB2BGR))
        else:
            pass
    return im0.transpose((2, 0, 1))[::-1]  # HWC->CHW,BGR->RGB


class PatchTrainer(object):
    def __init__(self, mode):
        self.config = patch_config_test.patch_configs[mode]()

        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = PatchTransformer().cuda()
        self.prob_extractor = MaxProbExtractor(0, 1, self.config).cuda()
        self.nps_calculator = NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation = TotalVariation().cuda()

    #     self.writer = self.init_tensorboard(mode)

    # def init_tensorboard(self, name=None):
    #     subprocess.Popen(['tensorboard', '--logdir', 'runs'])
    #     if name is not None:
    #         time_str = time.strftime("%Y%m%d-%H%M%S")
    #         return SummaryWriter(f'runs/{time_str}_{name}')
    #     else:
    #         return SummaryWriter()

    def train(self, opt):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        img_size = 640

        # Initialize
        device = select_device('')

        # Load model
        if opt.model == 'v5':
            model = attempt_load(self.config.weightfile_v5, map_location=device)  # load FP32 model
        if opt.model == 'v3':
            model = attempt_load(self.config.weightfile, map_location=device)

        compute_loss = ComputeLoss(model)
        stride = int(model.stride.max())  # model stride
        img_size = check_img_size(img_size, s=stride)  # check img_size
        # img_size = self.darknet_model.height   #416
        batch_size = self.config.batch_size  # 8
        n_epochs = 1
        max_lab = 20
        # savedir = "E:/python/kaiti/train_patch/testing/"
        # savedir = "D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/"

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu1 = self.read_image('saved_patches/v3/patchv3_2.jpg')  # 旋转20个度                  924test
        adv_patch_cpu2 = self.read_image('saved_patches/v3/random.jpg')  # 旋转20个度
        adv_patch_cpu3 = self.read_image('saved_patches/v3/patchdp.jpg')  # 旋转20个度
        adv_patch_cpu4 = self.read_image('saved_patches/v3/patch_thys.png')  # 旋转20个度
        adv_patch_cpu5 = self.read_image('saved_patches/v3/patchv3_2.jpg')  # 旋转20个度

        # adv_patch_cpu1 = self.read_image('test257/angle/print/random.png')  #旋转20个度    
        # adv_patch_cpu2 = self.read_image('test257/angle/patch_200_v5.png')  #旋转20个度 test1
        # adv_patch_cpu3 = self.read_image('test257/angle/patchobj_200_v5.png')
        # adv_patch_cpu4 = self.read_image('test257/angle/patchd_2002_v5.png')
        # adv_patch_cpu5 = self.read_image('test257/angle/patchamean_200_0.005_v5_2.png')

        # adv_patch_cpu = self.read_image("test257/test257new/patcha_753_1.png")

        # adv_patch_cpu.requires_grad_(True)
        mydataset = InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True)

        train_loader = torch.utils.data.DataLoader(mydataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=0)
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')

        # optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        # scheduler = self.config.scheduler_factory(optimizer)

        et0 = time.time()
        dorotate = True
        # p_img_random = torch.rand((batch_size, 3, img_size, img_size)).cuda()
        # p_img_random = p_img_random.unsqueeze(0).expand(batch_size, -1, -1, -1)
        # output_random, feature_random = model(p_img_random, visualize = True)
        # random_patch = torch.rand(adv_patch_cpu.size()).cuda()
        # random_patch = self.read_image("test257/patchamean257_200_fc.png")
        # adv_patch_cpu = adv_patch_cpu*0.7 + random_patch*0.3
        scale_dic = {'20': 294, '25': 230, '30': 191, '35': 164, '40': 140, '45': 123, '50': 111, '55': 101, '60': 93,
                     '65': 85, '70': 80,
                     '75': 75, '80': 70, '85': 66, '90': 63, '95': 60, '100': 57, '105': 54, '110': 52, '115': 51,
                     '120': 50}
        count_all= {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_1 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_6 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_7 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count_clean_8 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0,
                         '70': 0,
                         '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}

        ################################  v3_2  ################################
        count1_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count2_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count3_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count4_ours= {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count5_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count6_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count7_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count8_ours = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        ################################  random ################################
        count1_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count2_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count3_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count4_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count5_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count6_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count7_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count8_rd = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}

        ################################  dpatch  ################################
        count1_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count2_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count3_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count4_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count5_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count6_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count7_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count8_dp = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}

        ################################  thys  ################################
        count1_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count2_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count3_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count4_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count5_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count6_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count7_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        count8_ty = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
                    '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}

        sum_clean_1 = 0
        sum_clean_2 = 0
        sum_clean_3 = 0
        sum_clean_4 = 0
        sum_clean_5 = 0
        sum_clean_6 = 0
        sum_clean_7 = 0
        sum_clean_8 = 0

        sum1_ours = 0
        sum2_ours = 0
        sum3_ours = 0
        sum4_ours = 0
        sum5_ours = 0
        sum6_ours = 0
        sum7_ours = 0
        sum8_ours = 0

        sum1_rd = 0
        sum2_rd = 0
        sum3_rd = 0
        sum4_rd = 0
        sum5_rd = 0
        sum6_rd = 0
        sum7_rd = 0
        sum8_rd = 0

        sum1_dp = 0
        sum2_dp = 0
        sum3_dp = 0
        sum4_dp = 0
        sum5_dp = 0
        sum6_dp = 0
        sum7_dp = 0
        sum8_dp = 0

        sum1_ty = 0
        sum2_ty = 0
        sum3_ty = 0
        sum4_ty = 0
        sum5_ty = 0
        sum6_ty = 0
        sum7_ty = 0
        sum8_ty = 0

        # count1_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count2_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count3_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count4_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count5_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count6_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count7_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count8_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count9_2 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # sum3_2 = 0
        # sum1_2 = 0
        # sum2_2 = 0
        # sum4_2 = 0
        # sum5_2 = 0
        # sum6_2 = 0
        # sum7_2 = 0
        # sum8_2 = 0
        # sum9_2 = 0
        #
        # count1_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count2_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count3_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count4_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count5_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count6_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count7_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count8_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count9_3 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # sum3_3 = 0
        # sum1_3 = 0
        # sum2_3 = 0
        # sum4_3 = 0
        # sum5_3 = 0
        # sum6_3 = 0
        # sum7_3 = 0
        # sum8_3 = 0
        # sum9_3 = 0
        #
        # count1_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count2_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count3_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count4_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count5_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count6_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count7_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count8_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count9_4 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # sum3_4 = 0
        # sum1_4 = 0
        # sum2_4 = 0
        # sum4_4 = 0
        # sum5_4 = 0
        # sum6_4 = 0
        # sum7_4 = 0
        # sum8_4 = 0
        # sum9_4 = 0
        #
        # count1_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count2_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count3_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count4_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count5_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count6_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count7_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count8_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # count9_5 = {'20': 0, '25': 0, '30': 0, '35': 0, '40': 0, '45': 0, '50': 0, '55': 0, '60': 0, '65': 0, '70': 0,
        #             '75': 0, '80': 0, '85': 0, '90': 0, '95': 0, '100': 0, '105': 0, '110': 0, '115': 0, '120': 0}
        # sum3_5 = 0
        # sum1_5 = 0
        # sum2_5 = 0
        # sum4_5 = 0
        # sum5_5 = 0
        # sum6_5 = 0
        # sum7_5 = 0
        # sum8_5 = 0
        # sum9_5 = 0

        for epoch in range(n_epochs):
            ep_det_loss = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            # Ndet = 0
            bt0 = time.time()
            for i_batch, (img_batch, lab_batch, img_path, img_o) in tqdm(enumerate(train_loader),
                                                                         desc=f'Running epoch {epoch}',
                                                                         total=self.epoch_length):
                # print("i_batch",i_batch)
                # print("img_batch",img_batch.shape)
                # print("lab_batch",lab_batch)
                # Mask = []
                # print(img_path)
                list = []
                for i in range(len(img_path)):
                    list.append(torch.tensor([scale_dic[img_path[i].split('_')[1]]], dtype=torch.float))
                img_scale = torch.cat(list, dim=0).view(len(img_path), -1)

                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()

                    # 在图片上加mask
                    # img_batch, lab_batch = draw(lab_batch, img_batch)
                    # print(img_path[0].split('.')[0])
                    ori_shape = img_o.shape[2:]
                    final_shape = img_batch.shape[2:]
                    img_name_clean = img_path[0].split('.')[0]

                    # inputs_ori = {"image": img_batch, "origin_image":img_o,"height": img_size, "width": img_size, "ori_shape": ori_shape, "final_shape": final_shape, "path":img_path}
                    # grad_cam = grad_CAM.GradCAM(model, ori_shape, final_shape)

                    # save_p_img_batch = os.path.join("E:/python/kaiti/train_patch/output1/", img_path[0].split('.')[0]+ ".png")
                    # save_p_img_batch = os.path.join("E:/python/car_output2/val/images", img_path[0].split('.')[0]+ ".png")
                    # print(save_p_img_batch)
                    # save_image_tensor2cv2(img_batch[0, :, :, :].unsqueeze(0), save_p_img_batch)
                    # save_image_tensor2cv2(img_batch[0, :, :, :].unsqueeze(0), img_path)
                    # break
                    # output_clean = model(img_batch, visualize=False)
                    # output_clean = output_clean[0]
                    # pred_clean = non_max_suppression(output_clean, 0.7, 0.45, None, False, max_det=1000)
                    # count = pred_clean[0].size()[0]
                    # count3[img_path[i].split('_')[1]] += count
                    # sum3 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/",'clean/')
                    # plot_detection(pred_clean, savedir, img_batch, disc=img_name_clean)
                    # # del output_clean, pred_clean

                    # print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    # adv_patch = adv_patch_cpu.cuda()
                    # if epoch >= 300:
                    #     dorotate = True
                    adv_patch1 = adv_patch_cpu1.cuda()
                    adv_patch2 = adv_patch_cpu2.cuda()
                    adv_patch3 = adv_patch_cpu3.cuda()
                    adv_patch4 = adv_patch_cpu4.cuda()
                    adv_patch5 = adv_patch_cpu5.cuda()
                    adv_batch_t1, adv_batch_t2, adv_batch_t3, adv_batch_t4, adv_batch_t5 = self.patch_transformer(
                        adv_patch1, adv_patch2, adv_patch3, adv_patch4, adv_patch5, lab_batch, img_size, img_scale,
                        Noise=opt.noise, do_rotate=dorotate, rand_loc=True)

                    p_img_batch1 = self.patch_applier(img_batch, adv_batch_t1)
                    p_img_batch2 = self.patch_applier(img_batch, adv_batch_t2)
                    p_img_batch3 = self.patch_applier(img_batch, adv_batch_t3)
                    p_img_batch4 = self.patch_applier(img_batch, adv_batch_t4)
                    p_img_batch5 = self.patch_applier(img_batch, adv_batch_t5)
                    for i in range(p_img_batch1.size(0)):

                        weather_type = random.randint(0, 2)
                        # print('weather:', weather_type)

                        if weather_type == 0:
                            p_img_batch1[i, :, :, :], p_img_batch2[i, :, :, :], p_img_batch3[i, :, :, :], p_img_batch4[
                                                                                                          i, :, :,
                                                                                                          :], p_img_batch5[
                                                                                                              i, :, :,
                                                                                                              :] = weather_test.brighten(
                                p_img_batch1[i, :, :, :], p_img_batch2[i, :, :, :], p_img_batch3[i, :, :, :],
                                p_img_batch4[i, :, :, :], p_img_batch5[i, :, :, :], )
                        elif weather_type == 1:
                            # p_img_batch[i, :, : ,:] = weather.darken(p_img_batch[i, :, : ,:])
                            p_img_batch1[i, :, :, :], p_img_batch2[i, :, :, :], p_img_batch3[i, :, :, :], p_img_batch4[
                                                                                                          i, :, :,
                                                                                                          :], p_img_batch5[
                                                                                                              i, :, :,
                                                                                                              :] = weather_test.darken(
                                p_img_batch1[i, :, :, :], p_img_batch2[i, :, :, :], p_img_batch3[i, :, :, :],
                                p_img_batch4[i, :, :, :], p_img_batch5[i, :, :, :], )
                            # elif weather_type == 2:
                            # p_img_batch[i, :, : ,:] = weather.add_blur(p_img_batch[i, :, : ,:])
                            # elif weather_type == 3:
                            #     p_img_batch[i, :, : ,:] = weather.add_blur(p_img_batch[i, :, : ,:])

                            # elif weather_type == 2:
                            #     p_img_batch[i, :, : ,:] = weather.add_snow(p_img_batch[i, :, : ,:])
                            # elif weather_type == 3:
                            #     p_img_batch[i, :, : ,:] = weather.add_rain(p_img_batch[i, :, : ,:])
                            # elif weather_type == 4:
                            #     p_img_batch[i, :, : ,:] = weather.add_fog(p_img_batch[i, :, : ,:])
                            # elif weather_type == 5:
                            #     p_img_batch[i, :, : ,:] = weather.add_autumn(p_img_batch[i, :, : ,:])
                        elif weather_type == 2:
                            p_img_batch1[i, :, :, :] = p_img_batch1[i, :, :, :]
                            p_img_batch2[i, :, :, :] = p_img_batch2[i, :, :, :]
                            p_img_batch3[i, :, :, :] = p_img_batch3[i, :, :, :]
                            p_img_batch4[i, :, :, :] = p_img_batch4[i, :, :, :]
                            p_img_batch5[i, :, :, :] = p_img_batch5[i, :, :, :]

                    # p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))
                    p_img_batch1 = F.interpolate(p_img_batch1, (img_size, img_size))
                    clean_img_batch = F.interpolate(img_batch, (img_size, img_size))
                    p_img_batch2 = F.interpolate(p_img_batch2, (img_size, img_size))
                    p_img_batch3 = F.interpolate(p_img_batch3, (img_size, img_size))
                    p_img_batch4 = F.interpolate(p_img_batch4, (img_size, img_size))
                    p_img_batch5 = F.interpolate(p_img_batch5, (img_size, img_size))
                    # if True:

                    #     # minangle = -180 / 180 * math.pi
                    #     # maxangle = 180 / 180 * math.pi
                    #     # anglesize = 12
                    #     # angle = random.uniform(minangle, maxangle)
                    #     angle_int = random.randint(-3,3)
                    #     # anglesize = 12
                    #     # angle = random.uniform(minangle, maxangle)
                    #     angle = angle_int/3*math.pi
                    #     # angle = -0/3*math.pi
                    #     theta = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle) ,0]], dtype=torch.float)
                    #     grid = F.affine_grid(theta.unsqueeze(0).expand(p_img_batch1.size(0),-1,-1).cuda(), p_img_batch1.size())

                    #     p_img_batch1 = F.grid_sample(p_img_batch1, grid)
                    #     p_img_batch2 = F.grid_sample(p_img_batch2, grid)
                    #     p_img_batch3 = F.grid_sample(p_img_batch3, grid)
                    #     p_img_batch4 = F.grid_sample(p_img_batch4, grid)

                    # img_batch = F.grid_sample(img_batch, grid)

                    # img_name_proper = os.path.join("E:/python/train_patch/testing/",'proper_patched')
                    ############################ clean image ############################

                    ############################ clean image ############################
                    output_clean = model(img_batch, visualize=False)
                    output_clean = output_clean[0]
                    pred_clean = non_max_suppression(output_clean, 0.1, 0.45, None, False, max_det=1000) # 置信度0.1
                    count = pred_clean[0].size()[0]
                    count_clean_1[img_path[i].split('_')[1]] += count
                    sum_clean_1 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.1/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.2, 0.45, None, False, max_det=1000) # 置信度0.2
                    count = pred_clean[0].size()[0]
                    count_clean_2[img_path[i].split('_')[1]] += count
                    sum_clean_2 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.2/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.3, 0.45, None, False, max_det=1000) # 置信度0.3
                    count = pred_clean[0].size()[0]
                    count_clean_3[img_path[i].split('_')[1]] += count
                    sum_clean_3 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.3/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.4, 0.45, None, False, max_det=1000) # 置信度0.4
                    count = pred_clean[0].size()[0]
                    count_clean_4[img_path[i].split('_')[1]] += count
                    sum_clean_4 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.4/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.5, 0.45, None, False, max_det=1000) # 置信度0.5
                    count = pred_clean[0].size()[0]
                    count_clean_5[img_path[i].split('_')[1]] += count
                    sum_clean_5 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.5/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.6, 0.45, None, False, max_det=1000) # 置信度0.
                    count = pred_clean[0].size()[0]
                    count_clean_6[img_path[i].split('_')[1]] += count
                    sum_clean_6 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.6/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.7, 0.45, None, False, max_det=1000) # 置信度0.7
                    count = pred_clean[0].size()[0]
                    count_clean_7[img_path[i].split('_')[1]] += count
                    sum_clean_7 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.7/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    pred_clean = non_max_suppression(output_clean, 0.8, 0.45, None, False, max_det=1000) # 置信度0.8
                    count = pred_clean[0].size()[0]
                    count_clean_8[img_path[i].split('_')[1]] += count
                    sum_clean_8 += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/clean/", 'conf_0.8/')
                    if opt.plot:
                        plot_detection(pred_clean, savedir, clean_img_batch, disc=img_name_clean)

                    ############################ patchv3_2 ############################
                    output_proper = model(p_img_batch1, visualize=False)
                    output_proper = output_proper[0]
                    pred_proper = non_max_suppression(output_proper, 0.1, 0.45, None, False, max_det=1000) # 置信度0.1
                    count = pred_proper[0].size()[0]
                    count1_ours[img_path[i].split('_')[1]] += count
                    sum1_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.1/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.2, 0.45, None, False, max_det=1000) # 置信度0.2
                    count = pred_proper[0].size()[0]
                    count2_ours[img_path[i].split('_')[1]] += count
                    sum2_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.2/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.3, 0.45, None, False, max_det=1000) # 置信度0.3
                    count = pred_proper[0].size()[0]
                    count3_ours[img_path[i].split('_')[1]] += count
                    sum3_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.3/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.4, 0.45, None, False, max_det=1000)  # 置信度0.4
                    count = pred_proper[0].size()[0]
                    count4_ours[img_path[i].split('_')[1]] += count
                    sum4_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.4/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.5, 0.45, None, False, max_det=1000)  # 置信度0.5
                    count = pred_proper[0].size()[0]
                    count5_ours[img_path[i].split('_')[1]] += count
                    sum5_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.5/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.6, 0.45, None, False, max_det=1000)  # 置信度0.6
                    count = pred_proper[0].size()[0]
                    count6_ours[img_path[i].split('_')[1]] += count
                    sum6_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.6/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.7, 0.45, None, False, max_det=1000)  # 置信度0.7
                    count = pred_proper[0].size()[0]
                    count7_ours[img_path[i].split('_')[1]] += count
                    sum7_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.7/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.8, 0.45, None, False, max_det=1000)  # 置信度0.8
                    count = pred_proper[0].size()[0]
                    count8_ours[img_path[i].split('_')[1]] += count
                    sum8_ours += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/ours/", 'proper_patched_0.8/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch1, disc=img_name_clean)

                    ############################ random image ############################
                    output_proper = model(p_img_batch2, visualize=False)
                    output_proper = output_proper[0]
                    pred_proper = non_max_suppression(output_proper, 0.1, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count1_rd[img_path[i].split('_')[1]] += count
                    sum1_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.1/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.2, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count2_rd[img_path[i].split('_')[1]] += count
                    sum2_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.2/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.3, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count3_rd[img_path[i].split('_')[1]] += count
                    sum3_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.3/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.4, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count4_rd[img_path[i].split('_')[1]] += count
                    sum4_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.4/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.5, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count5_rd[img_path[i].split('_')[1]] += count
                    sum5_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.5/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.6, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count6_rd[img_path[i].split('_')[1]] += count
                    sum6_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.6/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.7, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count7_rd[img_path[i].split('_')[1]] += count
                    sum7_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.7/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.8, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count8_rd[img_path[i].split('_')[1]] += count
                    sum8_rd += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/random/", 'random_0.8/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch2, disc=img_name_clean)

                    ############################ Dpatch ############################
                    output_proper = model(p_img_batch3, visualize=False)
                    output_proper = output_proper[0]
                    pred_proper = non_max_suppression(output_proper, 0.1, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count1_dp[img_path[i].split('_')[1]] += count
                    sum1_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.1/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.2, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count2_dp[img_path[i].split('_')[1]] += count
                    sum2_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.2/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.3, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count3_dp[img_path[i].split('_')[1]] += count
                    sum3_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.3/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.4, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count4_dp[img_path[i].split('_')[1]] += count
                    sum4_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.4/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.5, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count5_dp[img_path[i].split('_')[1]] += count
                    sum5_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.5/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.6, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count6_dp[img_path[i].split('_')[1]] += count
                    sum6_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.6/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.7, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count7_dp[img_path[i].split('_')[1]] += count
                    sum7_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.7/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.8, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count8_dp[img_path[i].split('_')[1]] += count
                    sum8_dp += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/dpatch/", 'dpatch_0.8/')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch3, disc=img_name_clean)

                    ############################ Thys ############################
                    output_proper = model(p_img_batch4, visualize=False)
                    output_proper = output_proper[0]
                    pred_proper = non_max_suppression(output_proper, 0.1, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count1_ty[img_path[i].split('_')[1]] += count
                    sum1_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.1')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.2, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count2_ty[img_path[i].split('_')[1]] += count
                    sum2_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.2')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.3, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count3_ty[img_path[i].split('_')[1]] += count
                    sum3_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.3')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.4, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count4_ty[img_path[i].split('_')[1]] += count
                    sum4_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.4')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.5, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count5_ty[img_path[i].split('_')[1]] += count
                    sum5_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.5')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.6, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count6_ty[img_path[i].split('_')[1]] += count
                    sum6_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.6')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.7, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count7_ty[img_path[i].split('_')[1]] += count
                    sum7_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.7')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    pred_proper = non_max_suppression(output_proper, 0.8, 0.45, None, False, max_det=1000)
                    count = pred_proper[0].size()[0]
                    count8_ty[img_path[i].split('_')[1]] += count
                    sum8_ty += count
                    savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/thys/", 'thys_0.8')
                    if opt.plot:
                        plot_detection(pred_proper, savedir, p_img_batch4, disc=img_name_clean)

                    # 五
                    # output_proper = model(p_img_batch5, visualize=False)
                    # output_proper = output_proper[0]
                    # pred_proper = non_max_suppression(output_proper, 0.1, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count1_5[img_path[i].split('_')[1]] += count
                    # sum1_5 += count
                    # savedir = os.path.join("D:/Documents/bachelor/Graduation_Project/Project/train_patch/train_patch_a/digital_test/", 'proper_patched_0.1_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)

                    # pred_proper = non_max_suppression(output_proper, 0.2, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count2_5[img_path[i].split('_')[1]] += count
                    # sum2_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.2_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.3, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count3_5[img_path[i].split('_')[1]] += count
                    # sum3_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.3_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.4, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count4_5[img_path[i].split('_')[1]] += count
                    # sum4_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.4_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.5, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count5_5[img_path[i].split('_')[1]] += count
                    # sum5_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.5_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.6, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count6_5[img_path[i].split('_')[1]] += count
                    # sum6_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.6_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.7, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count7_5[img_path[i].split('_')[1]] += count
                    # sum7_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.7_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.8, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count8_5[img_path[i].split('_')[1]] += count
                    # sum8_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.8_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 0.9, 0.45, None, False, max_det=1000)
                    # count = pred_proper[0].size()[0]
                    # count9_5[img_path[i].split('_')[1]] += count
                    # sum9_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_0.9_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
                    #
                    # pred_proper = non_max_suppression(output_proper, 1, 0.45, None, False, max_det=1000)
                    # # count = pred_proper[0].size()[0]
                    # # count9_5[img_path[i].split('_')[1]] += count
                    # # sum9_5 += count
                    # savedir = os.path.join("E:/python/kaiti/train_patch/testing/", 'proper_patched_1_5/')
                    # if opt.plot:
                    #     plot_detection(pred_proper, savedir, p_img_batch5, disc=img_name_clean)
        path_all = 'test5/labels/'
        dirs_all = os.listdir(path_all)
        for i in range(len(dirs_all)):
            if dirs_all[i][-3:] != 'xml':
                key = dirs_all[i].split('_')[1]
                with open(os.path.join(path_all, dirs_all[i]), 'r') as f:
                    num_obj = len(f.readlines())
                    f.close()
                count_all[key] += num_obj

        num_obj_1 = count_all['25'] + count_all['30'] + count_all['35'] + count_all['40'] + count_all['45']
        num_obj_2 = count_all['50'] + count_all['55'] + count_all['60'] + count_all['65'] + count_all['70']
        num_obj_3 = count_all['75'] + count_all['80'] + count_all['85'] + count_all['90'] + count_all['95']
        num_obj_4 = count_all['100'] + count_all['105'] + count_all['110'] + count_all['115'] + count_all['120']

        ################################格式化输出################################
        print(f'Valid objects in testing images: {count_all}')
        print('----------------------------------------------clean 0.1-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.1 : {count_clean_1}')
        num_det_1_clean = count_clean_1['25'] + count_clean_1['30'] + count_clean_1['35'] + count_clean_1['40'] + count_clean_1['45']
        num_det_2_clean = count_clean_1['50'] + count_clean_1['55'] + count_clean_1['60'] + count_clean_1['65'] + count_clean_1['70']
        num_det_3_clean = count_clean_1['75'] + count_clean_1['80'] + count_clean_1['85'] + count_clean_1['90'] + count_clean_1['95']
        num_det_4_clean = count_clean_1['100'] + count_clean_1['105'] + count_clean_1['110'] + count_clean_1['115'] + count_clean_1['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.1: {sum(count_all.values()) - sum_clean_1}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.1-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.1 : {count1_ours}')
        num_det_1_ours = count1_ours['25'] + count1_ours['30'] + count1_ours['35'] + count1_ours['40'] + count1_ours['45']
        num_det_2_ours = count1_ours['50'] + count1_ours['55'] + count1_ours['60'] + count1_ours['65'] + count1_ours['70']
        num_det_3_ours = count1_ours['75'] + count1_ours['80'] + count1_ours['85'] + count1_ours['90'] + count1_ours['95']
        num_det_4_ours = count1_ours['100'] + count1_ours['105'] + count1_ours['110'] + count1_ours['115'] + count1_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.1: {sum(count_all.values()) - sum1_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.1-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.1 : {count1_rd}')
        num_det_1_rd = count1_rd['25'] + count1_rd['30'] + count1_rd['35'] + count1_rd['40'] + count1_rd['45']
        num_det_2_rd = count1_rd['50'] + count1_rd['55'] + count1_rd['60'] + count1_rd['65'] + count1_rd['70']
        num_det_3_rd = count1_rd['75'] + count1_rd['80'] + count1_rd['85'] + count1_rd['90'] + count1_rd['95']
        num_det_4_rd = count1_rd['100'] + count1_rd['105'] + count1_rd['110'] + count1_rd['115'] + count1_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.1: {sum(count_all.values()) - sum1_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.1-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.1 : {count1_dp}')
        num_det_1_dp = count1_dp['25'] + count1_dp['30'] + count1_dp['35'] + count1_dp['40'] + count1_dp['45']
        num_det_2_dp = count1_dp['50'] + count1_dp['55'] + count1_dp['60'] + count1_dp['65'] + count1_dp['70']
        num_det_3_dp = count1_dp['75'] + count1_dp['80'] + count1_dp['85'] + count1_dp['90'] + count1_dp['95']
        num_det_4_dp = count1_dp['100'] + count1_dp['105'] + count1_dp['110'] + count1_dp['115'] + count1_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.1: {sum(count_all.values()) - sum1_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.1-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.1 : {count1_ty}')
        num_det_1_ty = count1_ty['25'] + count1_ty['30'] + count1_ty['35'] + count1_ty['40'] + count1_ty['45']
        num_det_2_ty = count1_ty['50'] + count1_ty['55'] + count1_ty['60'] + count1_ty['65'] + count1_ty['70']
        num_det_3_ty = count1_ty['75'] + count1_ty['80'] + count1_ty['85'] + count1_ty['90'] + count1_ty['95']
        num_det_4_ty = count1_ty['100'] + count1_ty['105'] + count1_ty['110'] + count1_ty['115'] + count1_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.1: {sum(count_all.values()) - sum1_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.2-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.2 : {count_clean_2}')
        num_det_1_clean = count_clean_2['25'] + count_clean_2['30'] + count_clean_2['35'] + count_clean_2['40'] + count_clean_2['45']
        num_det_2_clean = count_clean_2['50'] + count_clean_2['55'] + count_clean_2['60'] + count_clean_2['65'] + count_clean_2['70']
        num_det_3_clean = count_clean_2['75'] + count_clean_2['80'] + count_clean_2['85'] + count_clean_2['90'] + count_clean_2['95']
        num_det_4_clean = count_clean_2['100'] + count_clean_2['105'] + count_clean_2['110'] + count_clean_2['115'] + count_clean_2['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.2: {sum(count_all.values()) - sum_clean_2}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.2-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.2 : {count2_ours}')
        num_det_1_ours = count2_ours['25'] + count2_ours['30'] + count2_ours['35'] + count2_ours['40'] + count2_ours['45']
        num_det_2_ours = count2_ours['50'] + count2_ours['55'] + count2_ours['60'] + count2_ours['65'] + count2_ours['70']
        num_det_3_ours = count2_ours['75'] + count2_ours['80'] + count2_ours['85'] + count2_ours['90'] + count2_ours['95']
        num_det_4_ours = count2_ours['100'] + count2_ours['105'] + count2_ours['110'] + count2_ours['115'] + count2_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.2: {sum(count_all.values()) - sum2_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.2-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.2 : {count2_rd}')
        num_det_1_rd = count2_rd['25'] + count2_rd['30'] + count2_rd['35'] + count2_rd['40'] + count2_rd['45']
        num_det_2_rd = count2_rd['50'] + count2_rd['55'] + count2_rd['60'] + count2_rd['65'] + count2_rd['70']
        num_det_3_rd = count2_rd['75'] + count2_rd['80'] + count2_rd['85'] + count2_rd['90'] + count2_rd['95']
        num_det_4_rd = count2_rd['100'] + count2_rd['105'] + count2_rd['110'] + count2_rd['115'] + count2_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.2: {sum(count_all.values()) - sum2_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.2-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.2 : {count2_dp}')
        num_det_1_dp = count2_dp['25'] + count2_dp['30'] + count2_dp['35'] + count2_dp['40'] + count2_dp['45']
        num_det_2_dp = count2_dp['50'] + count2_dp['55'] + count2_dp['60'] + count2_dp['65'] + count2_dp['70']
        num_det_3_dp = count2_dp['75'] + count2_dp['80'] + count2_dp['85'] + count2_dp['90'] + count2_dp['95']
        num_det_4_dp = count2_dp['100'] + count2_dp['105'] + count2_dp['110'] + count2_dp['115'] + count2_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.2: {sum(count_all.values()) - sum2_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.2-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.2 : {count2_ty}')
        num_det_1_ty = count2_ty['25'] + count2_ty['30'] + count2_ty['35'] + count2_ty['40'] + count2_ty['45']
        num_det_2_ty = count2_ty['50'] + count2_ty['55'] + count2_ty['60'] + count2_ty['65'] + count2_ty['70']
        num_det_3_ty = count2_ty['75'] + count2_ty['80'] + count2_ty['85'] + count2_ty['90'] + count2_ty['95']
        num_det_4_ty = count2_ty['100'] + count2_ty['105'] + count2_ty['110'] + count2_ty['115'] + count2_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.2: {sum(count_all.values()) - sum2_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.3-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.3 : {count_clean_3}')
        num_det_1_clean = count_clean_3['25'] + count_clean_3['30'] + count_clean_3['35'] + count_clean_3['40'] + count_clean_3['45']
        num_det_2_clean = count_clean_3['50'] + count_clean_3['55'] + count_clean_3['60'] + count_clean_3['65'] + count_clean_3['70']
        num_det_3_clean = count_clean_3['75'] + count_clean_3['80'] + count_clean_3['85'] + count_clean_3['90'] + count_clean_3['95']
        num_det_4_clean = count_clean_3['100'] + count_clean_3['105'] + count_clean_3['110'] + count_clean_3['115'] + count_clean_3['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.3: {sum(count_all.values()) - sum_clean_3}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.3-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.3 : {count3_ours}')
        num_det_1_ours = count3_ours['25'] + count3_ours['30'] + count3_ours['35'] + count3_ours['40'] + count3_ours['45']
        num_det_2_ours = count3_ours['50'] + count3_ours['55'] + count3_ours['60'] + count3_ours['65'] + count3_ours['70']
        num_det_3_ours = count3_ours['75'] + count3_ours['80'] + count3_ours['85'] + count3_ours['90'] + count3_ours['95']
        num_det_4_ours = count3_ours['100'] + count3_ours['105'] + count3_ours['110'] + count3_ours['115'] + count3_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.3: {sum(count_all.values()) - sum3_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.3-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.3 : {count3_rd}')
        num_det_1_rd = count3_rd['25'] + count3_rd['30'] + count3_rd['35'] + count3_rd['40'] + count3_rd['45']
        num_det_2_rd = count3_rd['50'] + count3_rd['55'] + count3_rd['60'] + count3_rd['65'] + count3_rd['70']
        num_det_3_rd = count3_rd['75'] + count3_rd['80'] + count3_rd['85'] + count3_rd['90'] + count3_rd['95']
        num_det_4_rd = count3_rd['100'] + count3_rd['105'] + count3_rd['110'] + count3_rd['115'] + count3_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.3: {sum(count_all.values()) - sum3_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.3-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.3 : {count3_dp}')
        num_det_1_dp = count3_dp['25'] + count3_dp['30'] + count3_dp['35'] + count3_dp['40'] + count3_dp['45']
        num_det_2_dp = count3_dp['50'] + count3_dp['55'] + count3_dp['60'] + count3_dp['65'] + count3_dp['70']
        num_det_3_dp = count3_dp['75'] + count3_dp['80'] + count3_dp['85'] + count3_dp['90'] + count3_dp['95']
        num_det_4_dp = count3_dp['100'] + count3_dp['105'] + count3_dp['110'] + count3_dp['115'] + count3_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.3: {sum(count_all.values()) - sum3_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.3-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.3 : {count3_ty}')
        num_det_1_ty = count3_ty['25'] + count3_ty['30'] + count3_ty['35'] + count3_ty['40'] + count3_ty['45']
        num_det_2_ty = count3_ty['50'] + count3_ty['55'] + count3_ty['60'] + count3_ty['65'] + count3_ty['70']
        num_det_3_ty = count3_ty['75'] + count3_ty['80'] + count3_ty['85'] + count3_ty['90'] + count3_ty['95']
        num_det_4_ty = count3_ty['100'] + count3_ty['105'] + count3_ty['110'] + count3_ty['115'] + count3_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.3: {sum(count_all.values()) - sum3_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.4-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.4 : {count_clean_4}')
        num_det_1_clean = count_clean_4['25'] + count_clean_4['30'] + count_clean_4['35'] + count_clean_4['40'] + count_clean_4['45']
        num_det_2_clean = count_clean_4['50'] + count_clean_4['55'] + count_clean_4['60'] + count_clean_4['65'] + count_clean_4['70']
        num_det_3_clean = count_clean_4['75'] + count_clean_4['80'] + count_clean_4['85'] + count_clean_4['90'] + count_clean_4['95']
        num_det_4_clean = count_clean_4['100'] + count_clean_4['105'] + count_clean_4['110'] + count_clean_4['115'] + count_clean_4['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.4: {sum(count_all.values()) - sum_clean_4}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.4-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.4 : {count4_ours}')
        num_det_1_ours = count4_ours['25'] + count4_ours['30'] + count4_ours['35'] + count4_ours['40'] + count4_ours['45']
        num_det_2_ours = count4_ours['50'] + count4_ours['55'] + count4_ours['60'] + count4_ours['65'] + count4_ours['70']
        num_det_3_ours = count4_ours['75'] + count4_ours['80'] + count4_ours['85'] + count4_ours['90'] + count4_ours['95']
        num_det_4_ours = count4_ours['100'] + count4_ours['105'] + count4_ours['110'] + count4_ours['115'] + count4_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.4: {sum(count_all.values()) - sum4_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.4-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.4 : {count4_rd}')
        num_det_1_rd = count4_rd['25'] + count4_rd['30'] + count4_rd['35'] + count4_rd['40'] + count4_rd['45']
        num_det_2_rd = count4_rd['50'] + count4_rd['55'] + count4_rd['60'] + count4_rd['65'] + count4_rd['70']
        num_det_3_rd = count4_rd['75'] + count4_rd['80'] + count4_rd['85'] + count4_rd['90'] + count4_rd['95']
        num_det_4_rd = count4_rd['100'] + count4_rd['105'] + count4_rd['110'] + count4_rd['115'] + count4_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.4: {sum(count_all.values()) - sum4_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.4-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.4 : {count4_dp}')
        num_det_1_dp = count4_dp['25'] + count4_dp['30'] + count4_dp['35'] + count4_dp['40'] + count4_dp['45']
        num_det_2_dp = count4_dp['50'] + count4_dp['55'] + count4_dp['60'] + count4_dp['65'] + count4_dp['70']
        num_det_3_dp = count4_dp['75'] + count4_dp['80'] + count4_dp['85'] + count4_dp['90'] + count4_dp['95']
        num_det_4_dp = count4_dp['100'] + count4_dp['105'] + count4_dp['110'] + count4_dp['115'] + count4_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.4: {sum(count_all.values()) - sum4_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.4-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.4 : {count4_ty}')
        num_det_1_ty = count4_ty['25'] + count4_ty['30'] + count4_ty['35'] + count4_ty['40'] + count4_ty['45']
        num_det_2_ty = count4_ty['50'] + count4_ty['55'] + count4_ty['60'] + count4_ty['65'] + count4_ty['70']
        num_det_3_ty = count4_ty['75'] + count4_ty['80'] + count4_ty['85'] + count4_ty['90'] + count4_ty['95']
        num_det_4_ty = count4_ty['100'] + count4_ty['105'] + count4_ty['110'] + count4_ty['115'] + count4_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.4: {sum(count_all.values()) - sum4_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.5-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.5 : {count_clean_5}')
        num_det_1_clean = count_clean_5['25'] + count_clean_5['30'] + count_clean_5['35'] + count_clean_5['40'] + count_clean_5['45']
        num_det_2_clean = count_clean_5['50'] + count_clean_5['55'] + count_clean_5['60'] + count_clean_5['65'] + count_clean_5['70']
        num_det_3_clean = count_clean_5['75'] + count_clean_5['80'] + count_clean_5['85'] + count_clean_5['90'] + count_clean_5['95']
        num_det_4_clean = count_clean_5['100'] + count_clean_5['105'] + count_clean_5['110'] + count_clean_5['115'] + count_clean_5['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.5: {sum(count_all.values()) - sum_clean_5}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.5-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.5 : {count5_ours}')
        num_det_1_ours = count5_ours['25'] + count5_ours['30'] + count5_ours['35'] + count5_ours['40'] + count5_ours['45']
        num_det_2_ours = count5_ours['50'] + count5_ours['55'] + count5_ours['60'] + count5_ours['65'] + count5_ours['70']
        num_det_3_ours = count5_ours['75'] + count5_ours['80'] + count5_ours['85'] + count5_ours['90'] + count5_ours['95']
        num_det_4_ours = count5_ours['100'] + count5_ours['105'] + count5_ours['110'] + count5_ours['115'] + count5_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.5: {sum(count_all.values()) - sum5_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.5-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.5 : {count5_rd}')
        num_det_1_rd = count5_rd['25'] + count5_rd['30'] + count5_rd['35'] + count5_rd['40'] + count5_rd['45']
        num_det_2_rd = count5_rd['50'] + count5_rd['55'] + count5_rd['60'] + count5_rd['65'] + count5_rd['70']
        num_det_3_rd = count5_rd['75'] + count5_rd['80'] + count5_rd['85'] + count5_rd['90'] + count5_rd['95']
        num_det_4_rd = count5_rd['100'] + count5_rd['105'] + count5_rd['110'] + count5_rd['115'] + count5_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.5: {sum(count_all.values()) - sum5_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.5-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.5 : {count5_dp}')
        num_det_1_dp = count5_dp['25'] + count5_dp['30'] + count5_dp['35'] + count5_dp['40'] + count5_dp['45']
        num_det_2_dp = count5_dp['50'] + count5_dp['55'] + count5_dp['60'] + count5_dp['65'] + count5_dp['70']
        num_det_3_dp = count5_dp['75'] + count5_dp['80'] + count5_dp['85'] + count5_dp['90'] + count5_dp['95']
        num_det_4_dp = count5_dp['100'] + count5_dp['105'] + count5_dp['110'] + count5_dp['115'] + count5_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.5: {sum(count_all.values()) - sum5_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.5-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.5 : {count5_ty}')
        num_det_1_ty = count5_ty['25'] + count5_ty['30'] + count5_ty['35'] + count5_ty['40'] + count5_ty['45']
        num_det_2_ty = count5_ty['50'] + count5_ty['55'] + count5_ty['60'] + count5_ty['65'] + count5_ty['70']
        num_det_3_ty = count5_ty['75'] + count5_ty['80'] + count5_ty['85'] + count5_ty['90'] + count5_ty['95']
        num_det_4_ty = count5_ty['100'] + count5_ty['105'] + count5_ty['110'] + count5_ty['115'] + count5_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.5: {sum(count_all.values()) - sum5_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.6-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.6 : {count_clean_6}')
        num_det_1_clean = count_clean_6['25'] + count_clean_6['30'] + count_clean_6['35'] + count_clean_6['40'] + count_clean_6['45']
        num_det_2_clean = count_clean_6['50'] + count_clean_6['55'] + count_clean_6['60'] + count_clean_6['65'] + count_clean_6['70']
        num_det_3_clean = count_clean_6['75'] + count_clean_6['80'] + count_clean_6['85'] + count_clean_6['90'] + count_clean_6['95']
        num_det_4_clean = count_clean_6['100'] + count_clean_6['105'] + count_clean_6['110'] + count_clean_6['115'] + count_clean_6['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.6: {sum(count_all.values()) - sum_clean_6}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.6-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.6 : {count6_ours}')
        num_det_1_ours = count6_ours['25'] + count6_ours['30'] + count6_ours['35'] + count6_ours['40'] + count6_ours['45']
        num_det_2_ours = count6_ours['50'] + count6_ours['55'] + count6_ours['60'] + count6_ours['65'] + count6_ours['70']
        num_det_3_ours = count6_ours['75'] + count6_ours['80'] + count6_ours['85'] + count6_ours['90'] + count6_ours['95']
        num_det_4_ours = count6_ours['100'] + count6_ours['105'] + count6_ours['110'] + count6_ours['115'] + count6_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.6: {sum(count_all.values()) - sum6_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.6-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.6 : {count6_rd}')
        num_det_1_rd = count6_rd['25'] + count6_rd['30'] + count6_rd['35'] + count6_rd['40'] + count6_rd['45']
        num_det_2_rd = count6_rd['50'] + count6_rd['55'] + count6_rd['60'] + count6_rd['65'] + count6_rd['70']
        num_det_3_rd = count6_rd['75'] + count6_rd['80'] + count6_rd['85'] + count6_rd['90'] + count6_rd['95']
        num_det_4_rd = count6_rd['100'] + count6_rd['105'] + count6_rd['110'] + count6_rd['115'] + count6_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.6: {sum(count_all.values()) - sum6_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.6-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.6 : {count6_dp}')
        num_det_1_dp = count6_dp['25'] + count6_dp['30'] + count6_dp['35'] + count6_dp['40'] + count6_dp['45']
        num_det_2_dp = count6_dp['50'] + count6_dp['55'] + count6_dp['60'] + count6_dp['65'] + count6_dp['70']
        num_det_3_dp = count6_dp['75'] + count6_dp['80'] + count6_dp['85'] + count6_dp['90'] + count6_dp['95']
        num_det_4_dp = count6_dp['100'] + count6_dp['105'] + count6_dp['110'] + count6_dp['115'] + count6_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.6: {sum(count_all.values()) - sum6_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.6-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.6 : {count6_ty}')
        num_det_1_ty = count6_ty['25'] + count6_ty['30'] + count6_ty['35'] + count6_ty['40'] + count6_ty['45']
        num_det_2_ty = count6_ty['50'] + count6_ty['55'] + count6_ty['60'] + count6_ty['65'] + count6_ty['70']
        num_det_3_ty = count6_ty['75'] + count6_ty['80'] + count6_ty['85'] + count6_ty['90'] + count6_ty['95']
        num_det_4_ty = count6_ty['100'] + count6_ty['105'] + count6_ty['110'] + count6_ty['115'] + count6_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.6: {sum(count_all.values()) - sum6_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.7-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.7 : {count_clean_7}')
        num_det_1_clean = count_clean_7['25'] + count_clean_7['30'] + count_clean_7['35'] + count_clean_7['40'] + count_clean_7['45']
        num_det_2_clean = count_clean_7['50'] + count_clean_7['55'] + count_clean_7['60'] + count_clean_7['65'] + count_clean_7['70']
        num_det_3_clean = count_clean_7['75'] + count_clean_7['80'] + count_clean_7['85'] + count_clean_7['90'] + count_clean_7['95']
        num_det_4_clean = count_clean_7['100'] + count_clean_7['105'] + count_clean_7['110'] + count_clean_7['115'] + count_clean_7['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.7: {sum(count_all.values()) - sum_clean_7}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.7-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.7 : {count7_ours}')
        num_det_1_ours = count7_ours['25'] + count7_ours['30'] + count7_ours['35'] + count7_ours['40'] + count7_ours['45']
        num_det_2_ours = count7_ours['50'] + count7_ours['55'] + count7_ours['60'] + count7_ours['65'] + count7_ours['70']
        num_det_3_ours = count7_ours['75'] + count7_ours['80'] + count7_ours['85'] + count7_ours['90'] + count7_ours['95']
        num_det_4_ours = count7_ours['100'] + count7_ours['105'] + count7_ours['110'] + count7_ours['115'] + count7_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.7: {sum(count_all.values()) - sum7_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.7-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.7 : {count7_rd}')
        num_det_1_rd = count7_rd['25'] + count7_rd['30'] + count7_rd['35'] + count7_rd['40'] + count7_rd['45']
        num_det_2_rd = count7_rd['50'] + count7_rd['55'] + count7_rd['60'] + count7_rd['65'] + count7_rd['70']
        num_det_3_rd = count7_rd['75'] + count7_rd['80'] + count7_rd['85'] + count7_rd['90'] + count7_rd['95']
        num_det_4_rd = count7_rd['100'] + count7_rd['105'] + count7_rd['110'] + count7_rd['115'] + count7_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.7: {sum(count_all.values()) - sum7_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.7-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.7 : {count7_dp}')
        num_det_1_dp = count7_dp['25'] + count7_dp['30'] + count7_dp['35'] + count7_dp['40'] + count7_dp['45']
        num_det_2_dp = count7_dp['50'] + count7_dp['55'] + count7_dp['60'] + count7_dp['65'] + count7_dp['70']
        num_det_3_dp = count7_dp['75'] + count7_dp['80'] + count7_dp['85'] + count7_dp['90'] + count7_dp['95']
        num_det_4_dp = count7_dp['100'] + count7_dp['105'] + count7_dp['110'] + count7_dp['115'] + count7_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.7: {sum(count_all.values()) - sum7_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.7-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.7 : {count7_ty}')
        num_det_1_ty = count7_ty['25'] + count7_ty['30'] + count7_ty['35'] + count7_ty['40'] + count7_ty['45']
        num_det_2_ty = count7_ty['50'] + count7_ty['55'] + count7_ty['60'] + count7_ty['65'] + count7_ty['70']
        num_det_3_ty = count7_ty['75'] + count7_ty['80'] + count7_ty['85'] + count7_ty['90'] + count7_ty['95']
        num_det_4_ty = count7_ty['100'] + count7_ty['105'] + count7_ty['110'] + count7_ty['115'] + count7_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.7: {sum(count_all.values()) - sum7_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')

        print('----------------------------------------------clean 0.8-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.8 : {count_clean_8}')
        num_det_1_clean = count_clean_8['25'] + count_clean_8['30'] + count_clean_8['35'] + count_clean_8['40'] + count_clean_8['45']
        num_det_2_clean = count_clean_8['50'] + count_clean_8['55'] + count_clean_8['60'] + count_clean_8['65'] + count_clean_8['70']
        num_det_3_clean = count_clean_8['75'] + count_clean_8['80'] + count_clean_8['85'] + count_clean_8['90'] + count_clean_8['95']
        num_det_4_clean = count_clean_8['100'] + count_clean_8['105'] + count_clean_8['110'] + count_clean_8['115'] + count_clean_8['120']
        recall_1_clean = num_det_1_clean / num_obj_1
        recall_2_clean = num_det_2_clean / num_obj_2
        recall_3_clean = num_det_3_clean / num_obj_3
        recall_4_clean = num_det_4_clean / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.8: {sum(count_all.values()) - sum_clean_8}={sum(count_all.values())}-({num_det_1_clean}+{num_det_2_clean}+{num_det_3_clean}+{num_det_4_clean}).')
        print(f'Recall of 25m-45m is {recall_1_clean}')
        print(f'Recall of 50m-70m is {recall_2_clean}')
        print(f'Recall of 75m-95m is {recall_3_clean}')
        print(f'Recall of 100m-120m is {recall_4_clean}')
        print('----------------------------------------------ours 0.8-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.8 : {count8_ours}')
        num_det_1_ours = count8_ours['25'] + count8_ours['30'] + count8_ours['35'] + count8_ours['40'] + count8_ours['45']
        num_det_2_ours = count8_ours['50'] + count8_ours['55'] + count8_ours['60'] + count8_ours['65'] + count8_ours['70']
        num_det_3_ours = count8_ours['75'] + count8_ours['80'] + count8_ours['85'] + count8_ours['90'] + count8_ours['95']
        num_det_4_ours = count8_ours['100'] + count8_ours['105'] + count8_ours['110'] + count8_ours['115'] + count8_ours['120']
        recall_1_ours = num_det_1_ours / num_obj_1
        recall_2_ours = num_det_2_ours / num_obj_2
        recall_3_ours = num_det_3_ours / num_obj_3
        recall_4_ours = num_det_4_ours / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.8: {sum(count_all.values()) - sum8_rd}={sum(count_all.values())}-({num_det_1_ours}+{num_det_2_ours}+{num_det_3_ours}+{num_det_4_ours}).')
        print(f'Recall of 25m-45m is {recall_1_ours}')
        print(f'Recall of 50m-70m is {recall_2_ours}')
        print(f'Recall of 75m-95m is {recall_3_ours}')
        print(f'Recall of 100m-120m is {recall_4_ours}')
        print('----------------------------------------------random 0.8-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.8 : {count8_rd}')
        num_det_1_rd = count8_rd['25'] + count8_rd['30'] + count8_rd['35'] + count8_rd['40'] + count8_rd['45']
        num_det_2_rd = count8_rd['50'] + count8_rd['55'] + count8_rd['60'] + count8_rd['65'] + count8_rd['70']
        num_det_3_rd = count8_rd['75'] + count8_rd['80'] + count8_rd['85'] + count8_rd['90'] + count8_rd['95']
        num_det_4_rd = count8_rd['100'] + count8_rd['105'] + count8_rd['110'] + count8_rd['115'] + count8_rd['120']
        recall_1_rd = num_det_1_rd / num_obj_1
        recall_2_rd = num_det_2_rd / num_obj_2
        recall_3_rd = num_det_3_rd / num_obj_3
        recall_4_rd = num_det_4_rd / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.8: {sum(count_all.values()) - sum8_dp}={sum(count_all.values())}-({num_det_1_rd}+{num_det_2_rd}+{num_det_3_rd}+{num_det_4_rd}).')
        print(f'Recall of 25m-45m is {recall_1_rd}')
        print(f'Recall of 50m-70m is {recall_2_rd}')
        print(f'Recall of 75m-95m is {recall_3_rd}')
        print(f'Recall of 100m-120m is {recall_4_rd}')
        print('----------------------------------------------dpatch 0.8-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.8 : {count8_dp}')
        num_det_1_dp = count8_dp['25'] + count8_dp['30'] + count8_dp['35'] + count8_dp['40'] + count8_dp['45']
        num_det_2_dp = count8_dp['50'] + count8_dp['55'] + count8_dp['60'] + count8_dp['65'] + count8_dp['70']
        num_det_3_dp = count8_dp['75'] + count8_dp['80'] + count8_dp['85'] + count8_dp['90'] + count8_dp['95']
        num_det_4_dp = count8_dp['100'] + count8_dp['105'] + count8_dp['110'] + count8_dp['115'] + count8_dp['120']
        recall_1_dp = num_det_1_dp / num_obj_1
        recall_2_dp = num_det_2_dp / num_obj_2
        recall_3_dp = num_det_3_dp / num_obj_3
        recall_4_dp = num_det_4_dp / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.8: {sum(count_all.values()) - sum8_dp}={sum(count_all.values())}-({num_det_1_dp}+{num_det_2_dp}+{num_det_3_dp}+{num_det_4_dp}).')
        print(f'Recall of 25m-45m is {recall_1_dp}')
        print(f'Recall of 50m-70m is {recall_2_dp}')
        print(f'Recall of 75m-95m is {recall_3_dp}')
        print(f'Recall of 100m-120m is {recall_4_dp}')
        print('----------------------------------------------thys 0.8-----------------------------------------------------------------------')
        print(f'Successfully detected at conf=0.8 : {count8_ty}')
        num_det_1_ty = count8_ty['25'] + count8_ty['30'] + count8_ty['35'] + count8_ty['40'] + count8_ty['45']
        num_det_2_ty = count8_ty['50'] + count8_ty['55'] + count8_ty['60'] + count8_ty['65'] + count8_ty['70']
        num_det_3_ty = count8_ty['75'] + count8_ty['80'] + count8_ty['85'] + count8_ty['90'] + count8_ty['95']
        num_det_4_ty = count8_ty['100'] + count8_ty['105'] + count8_ty['110'] + count8_ty['115'] + count8_ty['120']
        recall_1_ty = num_det_1_ty / num_obj_1
        recall_2_ty = num_det_2_ty / num_obj_2
        recall_3_ty = num_det_3_ty / num_obj_3
        recall_4_ty = num_det_4_ty / num_obj_4

        print(f'Total numbers of successful attacks at conf=0.8: {sum(count_all.values()) - sum8_ty}={sum(count_all.values())}-({num_det_1_ty}+{num_det_2_ty}+{num_det_3_ty}+{num_det_4_ty}).')
        print(f'Recall of 25m-45m is {recall_1_ty}')
        print(f'Recall of 50m-70m is {recall_2_ty}')
        print(f'Recall of 75m-95m is {recall_3_ty}')
        print(f'Recall of 100m-120m is {recall_4_ty}')
        print('################################################################################################################################')
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.2 : {count2_1}')
        # num_det_1_02 = count2_1['25'] + count2_1['30'] + count2_1['35'] + count2_1['40']
        # num_det_2_02 = count2_1['45'] + count2_1['50'] + count2_1['55'] + count2_1['60']
        # num_det_3_02 = count2_1['65'] + count2_1['70'] + count2_1['75'] + count2_1['80']
        # num_det_4_02 = count2_1['85'] + count2_1['90'] + count2_1['95'] + count2_1['100']
        # num_det_5_02 = count2_1['105'] + count2_1['110'] + count2_1['115'] + count2_1['120']
        # asr_1_02 = 1 - num_det_1_02/num_obj_1
        # asr_2_02 = 1 - num_det_2_02 / num_obj_2
        # asr_3_02 = 1 - num_det_3_02 / num_obj_3
        # asr_4_02 = 1 - num_det_4_02 / num_obj_4
        # asr_5_02 = 1 - num_det_5_02 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.1: {sum(count_all.values()) - sum2_1}={sum(count_all.values())}-({num_det_1_02}+{num_det_2_02}+{num_det_3_02}+{num_det_4_02}+{num_det_5_02}).')
        # print(f'ASR of 25m-40m is {asr_1_02}')
        # print(f'ASR of 45m-60m is {asr_2_02}')
        # print(f'ASR of 65m-80m is {asr_3_02}')
        # print(f'ASR of 85m-100m is {asr_4_02}')
        # print(f'ASR of 105m-120m is {asr_5_02}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.3 : {count3_1}')
        # num_det_1_03 = count3_1['25'] + count3_1['30'] + count3_1['35'] + count3_1['40']
        # num_det_2_03 = count3_1['45'] + count3_1['50'] + count3_1['55'] + count3_1['60']
        # num_det_3_03 = count3_1['65'] + count3_1['70'] + count3_1['75'] + count3_1['80']
        # num_det_4_03 = count3_1['85'] + count3_1['90'] + count3_1['95'] + count3_1['100']
        # num_det_5_03 = count3_1['105'] + count3_1['110'] + count3_1['115'] + count3_1['120']
        # asr_1_03 = 1 - num_det_1_03 / num_obj_1
        # asr_2_03 = 1 - num_det_2_03 / num_obj_2
        # asr_3_03 = 1 - num_det_3_03 / num_obj_3
        # asr_4_03 = 1 - num_det_4_03 / num_obj_4
        # asr_5_03 = 1 - num_det_5_03 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.3: {sum(count_all.values()) - sum3_1}={sum(count_all.values())}-({num_det_1_03}+{num_det_2_03}+{num_det_3_03}+{num_det_4_03}+{num_det_5_03}).')
        # print(f'ASR of 25m-40m is {asr_1_03}')
        # print(f'ASR of 45m-60m is {asr_2_03}')
        # print(f'ASR of 65m-80m is {asr_3_03}')
        # print(f'ASR of 85m-100m is {asr_4_03}')
        # print(f'ASR of 105m-120m is {asr_5_03}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.4 : {count4_1}')
        # num_det_1_04 = count4_1['25'] + count4_1['30'] + count4_1['35'] + count4_1['40']
        # num_det_2_04 = count4_1['45'] + count4_1['50'] + count4_1['55'] + count4_1['60']
        # num_det_3_04 = count4_1['65'] + count4_1['70'] + count4_1['75'] + count4_1['80']
        # num_det_4_04 = count4_1['85'] + count4_1['90'] + count4_1['95'] + count4_1['100']
        # num_det_5_04 = count4_1['105'] + count4_1['110'] + count4_1['115'] + count4_1['120']
        # asr_1_04 = 1 - num_det_1_04 / num_obj_1
        # asr_2_04 = 1 - num_det_2_04 / num_obj_2
        # asr_3_04 = 1 - num_det_3_04 / num_obj_3
        # asr_4_04 = 1 - num_det_4_04 / num_obj_4
        # asr_5_04 = 1 - num_det_5_04 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.4: {sum(count_all.values()) - sum4_1}={sum(count_all.values())}-({num_det_1_04}+{num_det_2_04}+{num_det_3_04}+{num_det_4_04}+{num_det_5_04}).')
        # print(f'ASR of 25m-40m is {asr_1_04}')
        # print(f'ASR of 45m-60m is {asr_2_04}')
        # print(f'ASR of 65m-80m is {asr_3_04}')
        # print(f'ASR of 85m-100m is {asr_4_04}')
        # print(f'ASR of 105m-120m is {asr_5_04}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.5 : {count5_1}')
        # num_det_1_05 = count5_1['25'] + count5_1['30'] + count5_1['35'] + count5_1['40']
        # num_det_2_05 = count5_1['45'] + count5_1['50'] + count5_1['55'] + count5_1['60']
        # num_det_3_05 = count5_1['65'] + count5_1['70'] + count5_1['75'] + count5_1['80']
        # num_det_4_05 = count5_1['85'] + count5_1['90'] + count5_1['95'] + count5_1['100']
        # num_det_5_05 = count5_1['105'] + count5_1['110'] + count5_1['115'] + count5_1['120']
        # asr_1_05 = 1 - num_det_1_05 / num_obj_1
        # asr_2_05 = 1 - num_det_2_05 / num_obj_2
        # asr_3_05 = 1 - num_det_3_05 / num_obj_3
        # asr_4_05 = 1 - num_det_4_05 / num_obj_4
        # asr_5_05 = 1 - num_det_5_05 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.5: {sum(count_all.values()) - sum5_1}={sum(count_all.values())}-({num_det_1_05}+{num_det_2_05}+{num_det_3_05}+{num_det_4_05}+{num_det_5_05}).')
        # print(f'ASR of 25m-40m is {asr_1_05}')
        # print(f'ASR of 45m-60m is {asr_2_05}')
        # print(f'ASR of 65m-80m is {asr_3_05}')
        # print(f'ASR of 85m-100m is {asr_4_05}')
        # print(f'ASR of 105m-120m is {asr_5_05}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.6 : {count6_1}')
        # num_det_1_06 = count6_1['25'] + count6_1['30'] + count6_1['35'] + count6_1['40']
        # num_det_2_06 = count6_1['45'] + count6_1['50'] + count6_1['55'] + count6_1['60']
        # num_det_3_06 = count6_1['65'] + count6_1['70'] + count6_1['75'] + count6_1['80']
        # num_det_4_06 = count6_1['85'] + count6_1['90'] + count6_1['95'] + count6_1['100']
        # num_det_5_06 = count6_1['105'] + count6_1['110'] + count6_1['115'] + count6_1['120']
        # asr_1_06 = 1 - num_det_1_06 / num_obj_1
        # asr_2_06 = 1 - num_det_2_06 / num_obj_2
        # asr_3_06 = 1 - num_det_3_06 / num_obj_3
        # asr_4_06 = 1 - num_det_4_06 / num_obj_4
        # asr_5_06 = 1 - num_det_5_06 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.6: {sum(count_all.values()) - sum6_1}={sum(count_all.values())}-({num_det_1_06}+{num_det_2_06}+{num_det_3_06}+{num_det_4_06}+{num_det_5_06}).')
        # print(f'ASR of 25m-40m is {asr_1_06}')
        # print(f'ASR of 45m-60m is {asr_2_06}')
        # print(f'ASR of 65m-80m is {asr_3_06}')
        # print(f'ASR of 85m-100m is {asr_4_06}')
        # print(f'ASR of 105m-120m is {asr_5_06}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.7 : {count7_1}')
        # num_det_1_07 = count7_1['25'] + count7_1['30'] + count7_1['35'] + count7_1['40']
        # num_det_2_07 = count7_1['45'] + count7_1['50'] + count7_1['55'] + count7_1['60']
        # num_det_3_07 = count7_1['65'] + count7_1['70'] + count7_1['75'] + count7_1['80']
        # num_det_4_07 = count7_1['85'] + count7_1['90'] + count7_1['95'] + count7_1['100']
        # num_det_5_07 = count7_1['105'] + count7_1['110'] + count7_1['115'] + count7_1['120']
        # asr_1_07 = 1 - num_det_1_07 / num_obj_1
        # asr_2_07 = 1 - num_det_2_07 / num_obj_2
        # asr_3_07 = 1 - num_det_3_07 / num_obj_3
        # asr_4_07 = 1 - num_det_4_07 / num_obj_4
        # asr_5_07 = 1 - num_det_5_07 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.7: {sum(count_all.values()) - sum7_1}={sum(count_all.values())}-({num_det_1_07}+{num_det_2_07}+{num_det_3_07}+{num_det_4_07}+{num_det_5_07}).')
        # print(f'ASR of 25m-40m is {asr_1_07}')
        # print(f'ASR of 45m-60m is {asr_2_07}')
        # print(f'ASR of 65m-80m is {asr_3_07}')
        # print(f'ASR of 85m-100m is {asr_4_07}')
        # print(f'ASR of 105m-120m is {asr_5_07}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.8 : {count8_1}')
        # num_det_1_08 = count8_1['25'] + count8_1['30'] + count8_1['35'] + count8_1['40']
        # num_det_2_08 = count8_1['45'] + count8_1['50'] + count8_1['55'] + count8_1['60']
        # num_det_3_08 = count8_1['65'] + count8_1['70'] + count8_1['75'] + count8_1['80']
        # num_det_4_08 = count8_1['85'] + count8_1['90'] + count8_1['95'] + count8_1['100']
        # num_det_5_08 = count8_1['105'] + count8_1['110'] + count8_1['115'] + count8_1['120']
        # asr_1_08 = 1 - num_det_1_08 / num_obj_1
        # asr_2_08 = 1 - num_det_2_08 / num_obj_2
        # asr_3_08 = 1 - num_det_3_08 / num_obj_3
        # asr_4_08 = 1 - num_det_4_08 / num_obj_4
        # asr_5_08 = 1 - num_det_5_08 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.8: {sum(count_all.values()) - sum8_1}={sum(count_all.values())}-({num_det_1_08}+{num_det_2_08}+{num_det_3_08}+{num_det_4_08}+{num_det_5_08}).')
        # print(f'ASR of 25m-40m is {asr_1_08}')
        # print(f'ASR of 45m-60m is {asr_2_08}')
        # print(f'ASR of 65m-80m is {asr_3_08}')
        # print(f'ASR of 85m-100m is {asr_4_08}')
        # print(f'ASR of 105m-120m is {asr_5_08}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=0.9 : {count9_1}')
        # num_det_1_09 = count9_1['25'] + count9_1['30'] + count9_1['35'] + count9_1['40']
        # num_det_2_09 = count9_1['45'] + count9_1['50'] + count9_1['55'] + count9_1['60']
        # num_det_3_09 = count9_1['65'] + count9_1['70'] + count9_1['75'] + count9_1['80']
        # num_det_4_09 = count9_1['85'] + count9_1['90'] + count9_1['95'] + count9_1['100']
        # num_det_5_09 = count9_1['105'] + count9_1['110'] + count9_1['115'] + count9_1['120']
        # asr_1_09 = 1 - num_det_1_09 / num_obj_1
        # asr_2_09 = 1 - num_det_2_09 / num_obj_2
        # asr_3_09 = 1 - num_det_3_09 / num_obj_3
        # asr_4_09 = 1 - num_det_4_09 / num_obj_4
        # asr_5_09 = 1 - num_det_5_09 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=0.9: {sum(count_all.values()) - sum9_1}={sum(count_all.values())}-({num_det_1_09}+{num_det_2_09}+{num_det_3_09}+{num_det_4_09}+{num_det_5_09}).')
        # print(f'ASR of 25m-40m is {asr_1_09}')
        # print(f'ASR of 45m-60m is {asr_2_09}')
        # print(f'ASR of 65m-80m is {asr_3_09}')
        # print(f'ASR of 85m-100m is {asr_4_09}')
        # print(f'ASR of 105m-120m is {asr_5_09}')
        #
        # print('---------------------------------------------------------------------------------------------------------------------')
        # print(f'Successfully attack at conf=1.0 : {count10_1}')
        # num_det_1_10 = count10_1['25'] + count10_1['30'] + count10_1['35'] + count10_1['40']
        # num_det_2_10 = count10_1['45'] + count10_1['50'] + count10_1['55'] + count10_1['60']
        # num_det_3_10 = count10_1['65'] + count10_1['70'] + count10_1['75'] + count10_1['80']
        # num_det_4_10 = count10_1['85'] + count10_1['90'] + count10_1['95'] + count10_1['100']
        # num_det_5_10 = count10_1['105'] + count10_1['110'] + count10_1['115'] + count10_1['120']
        # asr_1_10 = 1 - num_det_1_10 / num_obj_1
        # asr_2_10 = 1 - num_det_2_10 / num_obj_2
        # asr_3_10 = 1 - num_det_3_10 / num_obj_3
        # asr_4_10 = 1 - num_det_4_10 / num_obj_4
        # asr_5_10 = 1 - num_det_5_10 / num_obj_5
        #
        # print(f'Total numbers of successful attacks at conf=1.0: {sum(count_all.values()) - sum10_1}={sum(count_all.values())}-({num_det_1_10}+{num_det_2_10}+{num_det_3_10}+{num_det_4_10}+{num_det_5_10}).')
        # print(f'ASR of 25m-40m is {asr_1_10}')
        # print(f'ASR of 45m-60m is {asr_2_10}')
        # print(f'ASR of 65m-80m is {asr_3_10}')
        # print(f'ASR of 85m-100m is {asr_4_10}')
        # print(f'ASR of 105m-120m is {asr_5_10}')
        # print(sum2_1)
        # print(sum3_1)
        # print(sum4_1)
        # print(sum5_1)
        # print(sum6_1)
        # print(sum7_1)
        # print(sum8_1)
        # print(sum9_1)

        # print('count1_2', count1_2)
        # print('count2', count2_2)
        # print('count3', count3_2)
        # print('count4', count4_2)
        # print('count5', count5_2)
        # print('count6', count6_2)
        # print('count7', count7_2)
        # print('count8', count8_2)
        # print('count9', count9_2)

        # print(sum1_2)
        # print(sum2_2)
        # print(sum3_2)
        # print(sum4_2)
        # print(sum5_2)
        # print(sum6_2)
        # print(sum7_2)
        # print(sum8_2)
        # print(sum9_2)

        # print('count1_3', count1_3)
        # print('count2', count2_3)
        # print('count3', count3_3)
        # print('count4', count4_3)
        # print('count5', count5_3)
        # print('count6', count6_3)
        # print('count7', count7_3)
        # print('count8', count8_3)
        # print('count9', count9_3)

        # print('sum1_3', sum1_3)
        # print('sum2', sum2_3)
        # print('sum3', sum3_3)
        # print('sum4', sum4_3)
        # print('sum5', sum5_3)
        # print('sum6', sum6_3)
        # print('sum7', sum7_3)
        # print('sum8', sum8_3)
        # print('sum9', sum9_3)

        # print('count1_4', count1_4)
        # print('count2', count2_4)
        # print('count3', count3_4)
        # print('count4', count4_4)
        # print('count5', count5_4)
        # print('count6', count6_4)
        # print('count7', count7_4)
        # print('count8', count8_4)
        # print('count9', count9_4)

        # print(sum1_4)
        # print(sum2_4)
        # print(sum3_4)
        # print(sum4_4)
        # print(sum5_4)
        # print(sum6_4)
        # print(sum7_4)
        # print(sum8_4)
        # print(sum9_4)

        # print('count1_4', count1_5)
        # print('count2', count2_5)
        # print('count3', count3_5)
        # print('count4', count4_5)
        # print('count5', count5_5)
        # print('count6', count6_5)
        # print('count7', count7_5)
        # print('count8', count8_5)
        # print('count9', count9_5)

        # print(sum1_5)
        # print(sum2_5)
        # print(sum3_5)
        # print(sum4_5)
        # print(sum5_5)
        # print(sum6_5)
        # print(sum7_5)
        # print(sum8_5)
        # print(sum9_5)

    def generate_patch(self, type):
        """
        Generate a random patch as a starting point for optimization.

        :param type: Can be 'gray' or 'random'. Whether or not generate a gray or a random patch.
        :return:
        """
        if type == 'gray':
            adv_patch_cpu = torch.full((3, self.config.patch_size, self.config.patch_size), 0.5)
        elif type == 'random':
            adv_patch_cpu = torch.rand((3, self.config.patch_size, self.config.patch_size))

        return adv_patch_cpu

    def read_image(self, path):
        """
        Read an input image to be used as a patch

        :param path: Path to the image to be read.
        :return: Returns the transformed patch as a pytorch Tensor.
        """
        patch_img = Image.open(path).convert('RGB')
        tf = transforms.Resize((self.config.patch_size, self.config.patch_size))
        patch_img = tf(patch_img)
        tf = transforms.ToTensor()

        adv_patch_cpu = tf(patch_img)
        return adv_patch_cpu


def loss_feature(feature1, feature2, Mask):
    loss = torch.zeros(1, feature1[0].size(0)).cuda()
    for i in range(len(feature1)):
        loss = loss + (((feature1[i] - feature2[i]) * (feature1[i] - feature2[i]) * Mask[i]) / 65025).sum(
            axis=[1, 2, 3], keepdim=False)
    return loss


class Colors:
    # Ultralytics color palette https://ultralytics.com/
    def __init__(self):
        # hex = matplotlib.colors.TABLEAU_COLORS.values()
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb('#' + c) for c in hex]
        self.n = len(self.palette)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):  # rgb order (PIL)
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='v3', help='dataset.yaml path')
    # parser.add_argument('--patch', type=str, default='test257/test257new/patchamean_v5_yl0.0005.png', help='dataset.yaml path')
    parser.add_argument('--patch', type=str, default='test257/angle/patch_yl0.01.png', help='dataset.yaml path')
    parser.add_argument('--noise', type=str, default=False, help='dataset.yaml path')
    parser.add_argument('--plot', type=str, default=True, help='dataset.yaml path')

    # parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'E:/python/kaiti/yolov5/runs/train/exp28/weights/best28_v5.pt', help='model.pt path(s)')
    # parser.add_argument('--batch-size', type=int, default=2, help='batch size')
    # parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    # parser.add_argument('--conf-thres', type=float, default=0.001, help='confidence threshold')
    # parser.add_argument('--iou-thres', type=float, default=0.6, help='NMS IoU threshold')
    # parser.add_argument('--task', default='val', help='train, val, test, speed or study')
    # parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    # parser.add_argument('--workers', type=int, default=8, help='max dataloader workers (per RANK in DDP mode)')
    # parser.add_argument('--single-cls', action='store_true', help='treat as single-class dataset')
    # parser.add_argument('--augment', action='store_true', help='augmented inference')
    # parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    # parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    # parser.add_argument('--save-hybrid', action='store_true', help='save label+prediction hybrid results to *.txt')
    # parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    # parser.add_argument('--save-json', action='store_true', help='save a COCO-JSON results file')
    # parser.add_argument('--project', default=ROOT / 'runs/val', help='save to project/name')
    # parser.add_argument('--name', default='exp', help='save to project/name')
    # parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    # parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    # parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()

    return opt


def main(opt):
    if len(sys.argv) != 2:
        print('You need to supply (only) a configuration mode.')
        print('Possible modes are:')
        print(patch_config_test.patch_configs)

    # print("sys.argv[1]", sys.argv)
    trainer = PatchTrainer('paper_obj')
    trainer.train(opt)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
