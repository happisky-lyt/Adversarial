"""
Training code for Adversarial patch training


"""
import copy
import PIL
import numpy as np

import load_datac
import load_data
from tqdm import tqdm
import cv2
import argparse
from torch.autograd import Variable

from load_datac import *
from load_data import *
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
from yolov3.utils.datasets import create_dataloader
from yolov3.utils.general import (LOGGER, NCOLS, check_dataset, check_file, check_git_status, check_img_size,
                           check_requirements, check_suffix, check_yaml, colorstr, get_latest_run, increment_path,
                           init_seeds, intersect_dicts, labels_to_class_weights, labels_to_image_weights, methods,
                           one_cycle, print_args, print_mutation, strip_optimizer)

import weather

from torch.cuda import amp
from torch import optim

import patch_config
import sys
import time
from yolov3.models.experimental import attempt_load
from yolov3.utils.datasets import LoadStreams, LoadImages
from yolov3.utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from yolov3.utils.plots import Annotator, colors, save_one_box
from yolov3.utils.torch_utils import select_device

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
    b3[:,:, 0] = b2[:,:, 0] - 0.5 * b2[:,:, 2]
    b3[:,:, 1] = b2[:,:, 1] - 0.5 * b2[:,:, 3]
    b3[:,:, 2] = b2[:,:, 0] + 0.5 * b2[:,:, 2]
    b3[:,:, 3] = b2[:,:, 1] + 0.5 * b2[:,:, 3]
    # b1 = b1.repeat(10647, 1)
    left_column_max = 0.5*((b1[:,:, 0] + b3[: ,:, 0]) + abs(b1[:,:, 0] - b3[: ,:, 0]))
    right_column_min = 0.5 * ((b1[:,:, 2] + b3[:,:, 2]) - abs((b1[:,:, 2] - b3[:,:, 2])))
    up_row_max = 0.5*((b1[:,:, 1] + b3[:,:, 1]) + abs((b1[:,:, 1] - b3[:,:, 1])))
    down_row_min = 0.5 * ((b1[:,:, 3] + b3[:,:, 3]) - abs((b1[:,:, 3] - b3[:,:, 3])))
    a = torch.gt(left_column_max, right_column_min)
    b = torch.gt(up_row_max, down_row_min)
    b4 = b3[:,:,0:4]
    x1 = b1[:,:,2] - b1[:,:,0]
    x2 = b1[:,:,3] - b1[:,:,1]
    x3 = b4[:,:,2] - b4[:,:,0]
    x4 = b4[:,:,3] - b4[:,:,1]
    S1 = (x1) * (x2)
    S2 = (x3) * (x4)
    S_cross = (down_row_min - up_row_max) * (right_column_min - left_column_max)
    IOU = S_cross / (S1 + S2 - S_cross)
    c = IOU >=0.8

        # xc = torch.gt(left_column_max, right_column_min) or torch.gt(up_row_max, down_row_min)
        # xc = 0.5*((b1[:, 0] + b3[: , 0]) + abs(b1[:, 0] - b3[: , 0])) >= 0.5 * ((b1[:, 2] + b3[:, 2]) - abs((b1[:, 2] - b3[:, 2]))) or 0.5*((b1[:,3] + b3[:, 3]) - abs((b1[:,3] - b3[:, 3]))) <= 0.5*((b1[:, 1] + b3[:, 1]) + abs((b1[:, 1] - b3[:, 1])))
    xc = (~(a | b))&c
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


def plot_detection(pred, save_p_img_batch, disc=None):
    colors = Colors()  # 用于画检测框
    # Process predictions
    for i, det in enumerate(pred):  # detections per image
        if i>0:
            break
        im = save_p_img_batch[0, :, :, :]  # 读取已经保存的对抗图像
        # im0 = img0.copy()
        im0 = (np.array(im.detach().cpu())*255).transpose(1, 2, 0).astype(np.uint8).copy()

        # im0 = im0.astype(int)
        save_path = os.path.join("output/", '{}_detect.png'.format(disc))
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
        self.config = patch_config.patch_configs[mode]()

        # self.darknet_model = Darknet(self.config.cfgfile)
        # self.darknet_model.load_weights(self.config.weightfile)
        # self.darknet_model = self.darknet_model.eval().cuda() # TODO: Why eval?
        self.patch_applier = PatchApplier().cuda()
        self.patch_transformer = load_data.PatchTransformer().cuda()
        self.patch_transformerc = load_datac.PatchTransformer().cuda()
        self.prob_extractor = load_data.MaxProbExtractor(0, 1, self.config).cuda()
        self.prob_extractorc = load_datac.MaxProbExtractor(0, 1, self.config).cuda()
        self.nps_calculator = load_data.NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.nps_calculatorc = load_datac.NPSCalculator(self.config.printfile, self.config.patch_size).cuda()
        self.total_variation =  load_data.TotalVariation().cuda()
        self.total_variationc =  load_datac.TotalVariation().cuda()

        self.writer = self.init_tensorboard(mode)

    def init_tensorboard(self, name=None):
        subprocess.Popen(['tensorboard', '--logdir', 'runs'])
        if name is not None:
            time_str = time.strftime("%Y%m%d-%H%M%S")
            return SummaryWriter(f'runs/{time_str}_{name}')
        else:
            return SummaryWriter()
    

    
        


        






    def train(self, opt):
        """
        Optimize a patch to generate an adversarial example.
        :return: Nothing
        """
        img_size = 640

        # Initialize
        device = select_device('')
        min_contrast = 0.8
        max_contrast = 1.2
        min_brightness = -0.1
        max_brightness = 0.1
        noise_factor = 0.10

        # Load model
        model = attempt_load(self.config.weightfile , map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        img_size = check_img_size(img_size, s=stride)  # check img_size
        # img_size = self.darknet_model.height   #416
        batch_size = self.config.batch_size   #8
        n_epochs = 200
        max_lab = 25

        time_str = time.strftime("%Y%m%d-%H%M%S")

        # Generate stating point
        adv_patch_cpu = self.generate_patch("random")

        # save_p_img_batch = os.path.join("test257/random.png", img_path[0].split('\\')[-1])
        # save_p_img_batch = os.path.join("E:/python/kaiti/train_patch/saved_patches/", "random.png")
                    # print(save_p_img_batch)
        # save_image_tensor2cv2(img_batch[0, :, :, :].unsqueeze(0), save_p_img_batch)
                    # save_image_tensor2cv2(img_batch[0, :, :, :].unsqueeze(0), img_path)
        # save_image_tensor2cv2(adv_patch_cpu.unsqueeze(0), save_p_img_batch)  #

        # adv_patch_cpu = self.read_image("test100/patchfc.png")
        c = False
        if c:
            center_x = 100/2
            center_y = 100/2
            radius = 100/2
            adv_mask = generate_mask(150 ,150 , radius, center_x, center_y)
            adv_mask1 = torch.ones((150, 150)).expand(3, -1, -1)
            adv_mask = torch.mul(adv_mask, adv_mask1)
            adv_patch_cpu = torch.mul(adv_mask, adv_patch_cpu)


        adv_patch_cpu.requires_grad_(True)
        # mydataset = InriaDataset(self.config.img_dir, self.config.lab_dir, max_lab, img_size, shuffle=True)
        gs = max(int(model.stride.max()), 32)
        train_loader, dataset = create_dataloader(max_lab, self.config.img_dir, img_size, batch_size, gs, False,
                                              hyp=None, augment=False, cache=False, pad=0.0,
                      rect=False, rank=-1, workers=0, image_weights=False, quad=False, prefix='', shuffle=True)

        # train_loader = torch.utils.data.DataLoader(mydataset,
        #                                            batch_size=batch_size,
        #                                            shuffle=True,
        #                                            num_workers=0)

        
        self.epoch_length = len(train_loader)
        print(f'One epoch is {len(train_loader)}')
        compute_loss = ComputeLoss(model)

        optimizer = optim.Adam([adv_patch_cpu], lr=self.config.start_learning_rate, amsgrad=True)
        # scheduler = self.config.scheduler_factory(optimizer)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [90,160], gamma=0.1, last_epoch=-1)
        # scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60,110,160,210,260,310,360,410,460], gamma=0.1, last_epoch=-1)

        et0 = time.time()
        dorotate = True
        p_img_random = torch.rand((batch_size, 3, img_size, img_size)).cuda()
        # p_img_random = p_img_random.unsqueeze(0).expand(batch_size, -1, -1, -1)
        Feature = False
        if Feature:
            output_random, feature_random = model(p_img_random, visualize = True)
        else:
            output_random = model(p_img_random, visualize = False)
        nb = len(train_loader)

        scaler = amp.GradScaler(enabled=True)


        # scale_dic = {'20':294, '25':230, '30':191, '35':164, '40':140, '45':123, '50':111, '55':101, '60':93, '65':85, '70':80,
        #  '75':75, '80':70,'85':66, '90':63, '95':60, '100':56, '105':53, '110':51, '115':49, '120':48}
        
        scale_dic = {'20':294, '25':230, '30':191, '35':164, '40':140, '45':123, '50':111, '55':101, '60':93, '65':85, '70':80,
         '75':75, '80':70,'85':66, '90':63, '95':60, '100':57, '105':54, '110':52, '115':51, '120':50}
        

        for epoch in range(n_epochs):
            # model.train()
            ep_det_loss = 0
            ep_det_loss1 = 0
            ep_nps_loss = 0
            ep_tv_loss = 0
            ep_loss = 0
            # Ndet = 0
            bt0 = time.time()
            pbar = enumerate(train_loader)
            pbar = tqdm(pbar, total=nb, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
            # for i_batch, (img_batch, lab_batch, img_path, img_o) in tqdm(enumerate(train_loader), desc=f'Running epoch {epoch}',
            #                                             total=self.epoch_length):
            for i_batch, (img_batch, targets, img_path, _, lab_batch, img_o) in pbar:

                # img_scale= [0]*len(img_path)
                # for i in range(len(img_path)):
                #     img_scale[i] = scale_dic[img_path[i].split('_')[1]]

                # img_scale = torch.tensor([0],dtype=torch.float)
                list = []
                for i in range(len(img_path)):
                    list.append(torch.tensor([scale_dic[img_path[i].split('_')[1]]],dtype=torch.float))
                img_scale = torch.cat(list, dim = 0).view(len(img_path), -1)



            
                # print("i_batch",i_batch)
                # print("img_batch",img_batch.shape)
                # print("lab_batch",lab_batch)
                Mask = []
                # print(img_path)
                with autograd.detect_anomaly():
                    img_batch = img_batch.cuda()
                    lab_batch = lab_batch.cuda()

                    # 在图片上加mask
                    # img_batch, lab_batch = draw(lab_batch, img_batch)
                    # print(img_path[0].split('.')[0])






                    # ori_shape = img_o.shape[2:]
                    # final_shape = img_batch.shape[2:]

                    # inputs_ori = {"image": img_batch, "origin_image":img_o,"height": img_size, "width": img_size, "ori_shape": ori_shape, "final_shape": final_shape, "path":img_path}
                    # grad_cam = grad_CAM.GradCAM(model, ori_shape, final_shape)

                    # save_p_img_batch = os.path.join("E:/python/kaiti/train_patch/output1/", img_path[0].split('.')[0]+ ".png")
                    # .
                    # print(save_p_img_batch)
                    # save_image_tensor2cv2(img_batch[0, :, :, :].unsqueeze(0), save_p_img_batch)
                    # save_image_tensor2cv2(img_batch[0, :, :, :].unsqueeze(0), img_path)
                    # break


                    #print('TRAINING EPOCH %i, BATCH %i'%(epoch, i_batch))
                    adv_patch = adv_patch_cpu.cuda()
                    # if epoch >= 300:
                    # dorotate = True
                    if c:
                        adv_batch_t = self.patch_transformerc(adv_patch, lab_batch, img_size, adv_mask, Noise = opt.noise, do_rotate=dorotate, rand_loc=False)
                    else:
                        adv_batch_t = self.patch_transformer(adv_patch, lab_batch, img_size, img_scale, Noise = opt.noise, do_rotate=dorotate, rand_loc=False)
                    for i in range(adv_batch_t.size(0)):
                        # M = torch.zeros((3,640,640)).cuda()
                        # for j in range(adv_batch_t.size(1)):
                        #     M += adv_batch_t[i][j]
                        M= torch.sum(adv_batch_t[i], dim=0)
                        # save_image_tensor2cv2(M.unsqueeze(0), save_p_img_batch)


                    p_img_batch = self.patch_applier(img_batch, adv_batch_t)

                    # contrast = torch.cuda.FloatTensor(p_img_batch.size(0)).uniform_(min_contrast, max_contrast)
                    # contrast = contrast.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    # contrast = contrast.expand(-1, p_img_batch.size(-3), p_img_batch.size(-2), p_img_batch.size(-1))
                    # contrast = contrast.cuda()
                    # brightness = torch.cuda.FloatTensor(p_img_batch.size(0)).uniform_(min_brightness, max_brightness)
                    # brightness = brightness.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                    # brightness = brightness.expand(-1, p_img_batch.size(-3), p_img_batch.size(-2), p_img_batch.size(-1))
                    # brightness = brightness.cuda()
                    # # noise = torch.cuda.FloatTensor(p_img_batch.size()).uniform_(-1, 1) * noise_factor
                    # p_img_batch = p_img_batch * contrast + brightness
                    # p_img_batch = torch.clamp(p_img_batch, 0.000001, 0.99999)
                    # p_img_batch = F.interpolate(p_img_batch, (self.darknet_model.height, self.darknet_model.width))


                    for i in range(p_img_batch.size(0)):
                        
                            weather_type = random.randint(0,2) 
                            # print('weather:', weather_type)
                        
                            if weather_type == 0:
                                p_img_batch[i, :, : ,:] = weather.brighten(p_img_batch[i, :, : ,:])
                            elif weather_type == 1:
                                p_img_batch[i, :, : ,:] = weather.darken(p_img_batch[i, :, : ,:])
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
                                p_img_batch[i, :, : ,:] = p_img_batch[i, :, : ,:] 

                    p_img_batch = F.interpolate(p_img_batch, (img_size, img_size))


                    # if True:
                    #     # p_img_batch = transform(p_img_batch)
                    #     # img_batch = transform(img_batch)
                    #     # minangle = -180 / 180 * math.pi
                    #     # maxangle = 180 / 180 * math.pi
                    #     angle_int = random.randint(-3,3)
                    #     # anglesize = 12
                    #     # angle = random.uniform(minangle, maxangle)
                    #     angle = angle_int/3*math.pi
                    #     theta = torch.tensor([[math.cos(angle),math.sin(-angle),0],[math.sin(angle),math.cos(angle) ,0]], dtype=torch.float)
                    #     grid = F.affine_grid(theta.unsqueeze(0).expand(p_img_batch.size(0),-1,-1).cuda(), p_img_batch.size())
                    #     p_img_batch = F.grid_sample(p_img_batch, grid)
                    #     img_batch = F.grid_sample(img_batch, grid)


                    ''''''
                    # img = p_img_batch[0, :, :,]
                    # img = transforms.ToPILImage()(img.detach().cpu())
                    # img.show()
                    # impath = os.path.join("E:/python/train_patch/saved_patches/",str(i_batch)+".jpg")
                    #
                    # img.save(impath, dpi=(300, 300))
                    # save_p_img_batch = os.path.join("E:/python/train_patch/saved_patches/",str(i_batch)+".png")
                    # save_image_tensor2cv2(p_img_batch[0,:,:,:].unsqueeze(0), save_p_img_batch)  #
                    # box1 = transbox(lab_batch)
                    # continue

                    # output = self.darknet_model(p_img_batch)
                    # output, feature = model(p_img_batch, visualize=True)
                    ''''''
                    if Feature:
                        output, feature = model(p_img_batch, visualize=True)
                        output1, feature1 = model(img_batch,visualize = True)
                        output2 = output[1]
                        output = output[0]
                        output1 = output1[0]
                        # loss, _, lobj = compute_loss(output2, targets.to(device))
                        for i in range(len(feature1)):
                            feature[i] = feature[i].to(torch.float64)
                            mask = torch.where(feature[i] > 0.5, 1.0, feature[i])
                            mask = torch.where(feature[i] < 0.5, 0.0, feature[i])
                            # mask = mask*feature[i]
                            Mask.append(mask)
                        floss_positive = torch.mean(loss_feature(feature, feature1, Mask))*0.02
                        # floss_random = torch.mean(loss_feature(feature, feature_random, Mask))*0.005
                    else:
                        output = model(p_img_batch, visualize=False)
                        output2 = output[1]
                        output = output[0]
                        output1 = model(img_batch,visualize = False)
                        output1 = output1[0]
                    # newoutput = computeIOU(box1, output)


                    
                    
                    # for i in range(len(feature1)):
                    #     feature[i] = feature[i].to(torch.float64)
                    #     mask = torch.where(feature[i] > 0, 1.0, feature[i])
                    #     mask = torch.where(feature[i] < 0, 0.0, feature[i])
                    #     # mask = mask*feature[i]
                    #     Mask.append(mask)




                    # if epoch<50:
                    #     mth = 1
                    # else: mth = 2
                    
                    
                    max_prob = self.prob_extractor(output, opt.mth) #newoutput
                    # max_prob = self.prob_extractor(output, mth) #newoutput
                    pred1 = non_max_suppression(output1, 0.25, 0.45, None, False, max_det=1000)

                    pred = non_max_suppression(output, 0.25, 0.45, None, False, max_det=1000) #newoutput
                    img_patch = plot_detection(pred, p_img_batch,disc="img_patch")  # 由于只保存BatchSize中的一张照片，故只能读取到一张照片
                    # img_raw = plot_detection(pred1, img_batch, disc="img_raw")




                    # temp= np.zeros(2)
                    # score = np.zeros(2)
                    # scoret = torch.Tensor(score)
                    # # print(pred)
                    # for i, det in enumerate(pred):
                    #     if det.shape[0] > 0:
                    #         scoret[i] = det[:, 4].max()
                    #     else:
                    #         scoret[i] = 0


                    # score = np.zeros(16)
                    # scoret = torch.Tensor(score)
                    # # print(pred)
                    # for i, det in enumerate(pred):
                    #     if det.shape[0] > 0:
                    #         scoret[i] = det[:, 4].max()
                    #     else:
                    #         scoret[i] = 0
                    # xc = output[..., 4] > 0.25  #newoutput
                    #
                    # list = []
                    # for xi, x in enumerate(output):  # image index, image inference #newoutput
                    #     # Apply constraints
                    #     # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
                    #     list.append(x[xc[xi]])  # confidence
                    #
                    # temp = np.zeros(16)
                    #
                    # max_prob1 = torch.Tensor(temp)
                    # max_prob1.requires_grad = True
                    # max_prob1 = max_prob1.cuda()
                    # max_prob2 = torch.Tensor(temp)
                    # max_prob2.requires_grad = True
                    # max_prob2 = max_prob2.cuda()
                    # # output_cpu= output.cpu()
                    # i = 0
                    # ndet = 0
                    # for det in list:
                    #     if det.shape[0] > 0:
                    #         max_prob1[i] = det[:, 4].mean()
                    #         max_prob2[i] = det[:, 5].mean()
                    #         ndet = ndet +det.shape[0]
                    #     else:
                    #         max_prob1[i] = 0
                    #         max_prob2[i] = 0
                    #     i = i + 1



                    #只计算一个概率的损失
                    # temp = np.zeros(2)
                    # max_prob = torch.Tensor(temp)
                    # max_prob.requires_grad=True
                    # max_prob = max_prob.cuda()
                    # # output_cpu= output.cpu()
                    # for i in range(2):
                    #     max_prob[i] = output[i, :, 4].max()

                    # max_prob = self.prob_extractor(output)
                    loss_list = []
                    for i in range(p_img_batch.size(0)):

                        loss_list.append(model.compute_object_vanishing_gradient(Variable(p_img_batch[i].unsqueeze(0))).expand(1))

                    det_loss1 = torch.mean(torch.cat(loss_list)).squeeze(0)*4

                    if c:
                        nps = self.nps_calculatorc(adv_patch, adv_mask.cuda())
                        tv = self.total_variationc(adv_patch, adv_mask.cuda())
                    else:
                        nps = self.nps_calculator(adv_patch)
                        tv = self.total_variation(adv_patch)

                    # 分形维数
                    # patch_size1 = adv_patch.size(-1)
                    # adv_image = torch.zeros((patch_size1, patch_size1))
                    # adv_image = adv_patch.permute(1, 2, 0)[:, :, 0]*0.299 + adv_patch.permute(1, 2, 0)[:, :, 1]*0.587 + adv_patch.permute(1, 2, 0)[:, :, 2]*0.114
                    # adv_image = (adv_image.detach().cpu().numpy()*255).astype(np.uint8)
                    
                    
                    # fd = fractal_dimension(adv_image)



                    nps_loss = nps*0.01
                    tv_loss = tv*2.5

                    # floss_positive = torch.mean(loss_feature(feature, feature1, Mask))
                    # floss_random = torch.mean(loss_feature(feature, feature_random, Mask))

                    # fd_loss = (1/fd)*0.1
                    # det_loss = torch.mean(max_prob)
                    # det_loss = torch.sum(max_prob)
                    # if epoch > 600:
                    #     det_loss = torch.mean(max_prob)
                    # det_loss = 0.5*torch.max(max_prob1) + 0.5*torch.max(max_prob2)  # + 0.5*torch.max(scoret)
                    # det_loss = torch.sum(scoret)                    # if det_loss == 0:  # 如果当前图片的最大目标分数较低，跳过当前图片，不参与反向传播
                    #
                    #     continue
                    # loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())+ 1/floss_positive
                    # loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())  #+ 1/floss_positive*0.1 #+ floss_random*0.01 + 
                    # loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    if opt.loss_mode =='yl':
                        det_loss = torch.mean(max_prob)
                        loss, _, lobj = compute_loss(output2, targets.to(device))
                        # # loss.requires_grad_(True)
                        loss = 1/loss*0.001 + det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                        # loss = -loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                        # loss = 0.5-lobj  + torch.max(tv_loss, torch.tensor(0.1).cuda())
                        # loss = det_loss1 + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    else:
                        det_loss = torch.mean(max_prob)
                        loss = det_loss + nps_loss + torch.max(tv_loss, torch.tensor(0.1).cuda())
                    

                    
                    ep_det_loss += det_loss.detach().cpu().numpy()
                    ep_det_loss1 += det_loss1.detach().cpu().numpy()
                    ep_nps_loss += nps_loss.detach().cpu().numpy()
                    ep_tv_loss += tv_loss.detach().cpu().numpy()
                    ep_loss += loss
                    # Ndet = Ndet + ndet

                    loss.backward()
                    # scaler.scale(loss).backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    adv_patch_cpu.data.clamp_(0, 1)       #keep patch in image range

                    bt1 = time.time()
                    if i_batch%5 == 0:
                        iteration = self.epoch_length * epoch + i_batch 

                        self.writer.add_scalar('total_loss', loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss', det_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/det_loss1', det_loss1.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/nps_loss', nps_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('loss/tv_loss', tv_loss.detach().cpu().numpy(), iteration)
                        self.writer.add_scalar('misc/epoch', epoch, iteration)
                        self.writer.add_scalar('misc/learning_rate', optimizer.param_groups[0]["lr"], iteration)
                        if Feature:
                        # self.writer.add_scalar('loss/fd', fd, iteration)
                            self.writer.add_scalar('loss/floss_positive', floss_positive.detach().cpu().numpy(), iteration)
                            # self.writer.add_scalar('loss/floss_random', floss_random.detach().cpu().numpy(), iteration)

                        self.writer.add_image('patch', adv_patch_cpu, iteration)
                    # if i_batch + 1 >= len(train_loader):
                    #     print('\n')
                    # else:
                    #     del adv_batch_t, output, det_loss, p_img_batch, nps_loss, tv_loss, loss
                    #     torch.cuda.empty_cache()
                    # bt0 = time.time()
            et1 = time.time()
            ep_det_loss = ep_det_loss/len(train_loader)
            ep_det_loss1 = ep_det_loss1/len(train_loader)
            ep_nps_loss = ep_nps_loss/len(train_loader)
            ep_tv_loss = ep_tv_loss/len(train_loader)
            ep_loss = ep_loss/len(train_loader)

            # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
            # plt.imshow(im)
            # plt.savefig(f'pics/{time_str}_{self.config.patch_name}_{epoch}.png')
            # if epoch >= 500:
            # scheduler.step(ep_loss)
            scheduler.step()
            if True:
                print('  EPOCH NR: ', epoch)
                print('EPOCH LOSS: ', ep_loss)
                print('  DET LOSS: ', ep_det_loss)
                print('  DET LOSS1: ', ep_det_loss1)
                print('  NPS LOSS: ', ep_nps_loss)
                print('   TV LOSS: ', ep_tv_loss)
                print('EPOCH TIME: ', et1-et0)
                # print('      ndet: ', Ndet)
                # if ndet <=1500000:
                save_p_img_batch = os.path.join("saved_patches/test/", "patchv5_1.jpg")
                save_image_tensor2cv2(adv_patch_cpu.unsqueeze(0), save_p_img_batch)  #

                # im = transforms.ToPILImage('RGB')(adv_patch_cpu)
                # # plt.imshow(im)
                # # plt.show()
                # im.save("saved_patches/patchnew9.jpg", dpi=(300, 300))
                del adv_batch_t, output, det_loss, det_loss1, pred, max_prob, p_img_batch, nps_loss, tv_loss, loss
                # del adv_batch_t, output, det_loss, p_img_batch, nps_loss, tv_loss, loss
                torch.cuda.empty_cache()
            et0 = time.time()

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



def generate_mask(img_height, img_width, radius, center_x, center_y):
        y, x = np.ogrid[0:img_height, 0:img_width]

        # circle mask

        mask = (x - center_x) ** 2 + (y - center_y) ** 2 <= radius ** 2

        return torch.from_numpy(mask)



# def loss_feature(feature1, feature2, Mask):
#     loss = torch.zeros(1, feature1[0].size(0)).cuda()
#     for i in range(len(feature1)):
#         loss = loss + (((feature1[i]-feature2[i]) * (feature1[i]-feature2[i]) * Mask[i])/65025).sum(axis=[1, 2, 3], keepdim=False)
#     return loss


# def loss_feature(feature1, feature2,Mask):
#     loss = torch.zeros(1, feature1[0].size(0)).cuda()
#     L = 0
#     for i in range(len(feature2)):
#         mean = feature2[i].mean(axis=[1, 2, 3], keepdim = False).expand(feature2[i].size(3),-1).unsqueeze(0).expand(feature2[i].size(1),-1,-1).unsqueeze(0).expand(feature2[i].size(2), -1, -1 ,-1).transpose(0, 3).contiguous()
#         # mean = mean.expand(feature2[i].size(1),-1).unsqueeze(0).expand(feature2[i].size(2),-1,-1).unsqueeze(0).expand(feature2[i].size(3), -1, -1 ,-1)
#         # mean = mean.view(feature2[i].size(0), feature2[i].size(1), feature2[i].size(2), feature2[i].size(3))
#         f_loss = (feature1[i] - feature2[i])/mean
#         f_loss = (f_loss * f_loss).sum(axis=[1, 2, 3], keepdim=False)/(feature2[i].size(1) * feature2[i].size(2) * feature2[i].size(3))
#         loss = loss + f_loss
#         L += 1
#     loss = loss/L
#     return loss
def loss_feature(feature1, feature2,Mask):
    loss = torch.zeros(1, feature1[0].size(0)).cuda()
    L = 0
    for i in range(len(feature2)):
        mean = (feature2[i]*Mask[i]).mean(axis=[1, 2, 3], keepdim = False).expand(feature2[i].size(3),-1).unsqueeze(0).expand(feature2[i].size(1),-1,-1).unsqueeze(0).expand(feature2[i].size(2), -1, -1 ,-1).transpose(0, 3).contiguous()
        # mean = mean.expand(feature2[i].size(1),-1).unsqueeze(0).expand(feature2[i].size(2),-1,-1).unsqueeze(0).expand(feature2[i].size(3), -1, -1 ,-1)
        # mean = mean.view(feature2[i].size(0), feature2[i].size(1), feature2[i].size(2), feature2[i].size(3))
        f_loss = ((feature1[i] - feature2[i])*Mask[i])/mean
        f_loss = (f_loss * f_loss).sum(axis=[1, 2, 3], keepdim=False)/(Mask[i].sum(axis=[1, 2, 3], keepdim=False))
        loss = loss + f_loss
        L += 1
    loss = loss/L
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
    # parser.add_argument('--model', type=str, default='v5', help='dataset.yaml path')
    parser.add_argument('--patch', type=str, default='test257/test257new/patcha_v5.png', help='dataset.yaml path')  # 这一行代码没什么用
    parser.add_argument('--noise', type=bool, default=True, help='dataset.yaml path')
    parser.add_argument('--mth', type=int, default=2, help='dataset.yaml path')
    parser.add_argument('--loss_mode', type=str, default='yl', help='dataset.yaml path')
    
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
        print(patch_config.patch_configs)

    # print("sys.argv[1]", sys.argv)
    trainer = PatchTrainer('paper_obj')
    trainer.train(opt)

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


