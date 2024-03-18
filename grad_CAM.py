import cv2
import numpy as np
from torch.functional import split
# from build_utils import utils
import torch
# from build_utils import img_utils
from matplotlib import pyplot as plt
from PIL import Image
# from misc_functions import *
from yolov3.utils.general import non_max_suppression
import os
import torch.nn as nn
import grad_CAM
class GradCAM(object):
    """
    1: the network does not update gradient, input requires the update
    2: use targeted class's score to do backward propagation
    """

    def __init__(self, net, ori_shape, final_shape):
        self.net = net
        # self.layer_name = layer_name  #暂时不用
        self.ori_shape = ori_shape
        self.final_shape = final_shape
        self.feature = None
        self.gradient = None
        self.net.eval()
        #------给x rigister hook没成功，尝试另外一种方法----#
        self.handlers = []
        self._register_hook()
        #------给x rigister hook没成功，尝试另外一种方法----#
        
        #self.feature_extractor = FeatureExtractor(self.net, self.layer_name)


    def _get_features_hook(self, module, input, output):
        self.feature = output.detach()
        print("feature shape:{}".format(output.size()))

    
    def _get_grads_hook(self, module, input_grad, output_grad):
        self.gradient = output_grad[0].detach()
        print("gradient shape:{}".format(output_grad[0].size()))

    
    def _register_hook(self):
        
        # for name, module in self.net.named_modules():
        #     if name == self.layer_name:
        #         self.handlers.append(module.register_forward_hook(self._get_features_hook))
        #         self.handlers.append(module.register_backward_hook(self._get_grads_hook))
        # for idx, name in enumerate(self.net._modules['model']):#.items()
        #     if idx == 6:#10:
        #         if isinstance(name, nn.Sequential):
        #             layer_name_tmp=name._modules['7'].cv2#['3'].cv2
        #             if layer_name_tmp == self.layer_name:
        #                 self.handlers.append(layer_name_tmp.register_forward_hook(self._get_features_hook))
        #                 self.handlers.append(layer_name_tmp.register_backward_hook(self._get_grads_hook))
        for name, module in self.net.named_modules():
            if name == 'model.1':#yolov3:'model.22';yolov5:'model.20'
                self.handlers.append(module.register_forward_hook(self._get_features_hook))
                self.handlers.append(module.register_backward_hook(self._get_grads_hook))
                print(self.handlers)
                    
        # for i, module in enumerate(self.net.module.backbone.layer3.residual_7._modules.items()):
        #     if module[1] == self.net.module.backbone.layer3.residual_7.conv2:#module[1] == self.layer_name:
        #         self.handlers.append(module[1].register_forward_hook(self._get_features_hook))
        #         self.handlers.append(module[1].register_backward_hook(self._get_grads_hook))
            
    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()
    
    def imageRev(img):
        im1 = np.array(img)#zy:取消注释
        im1 = 255 - im1
        #im1 = Image.fromarray(im1)
        return im1

    def __call__(self, inputs, conf_thres, iou_thres,classes, agnostic_nms,
                                   max_det,augment):
        
        input_image = inputs['image']
        path= inputs['path'].split('\\')[-1].split('.')[0]
        pred=self.net(input_image,augment)[0]
        output_nonmax = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms,
                                   max_det=max_det)[0]#shape:torch.Size([3, 7])
        print('output_nonmax=',output_nonmax)
        #zy:这里不需要缩放
        # output_nonmax[:, :4] = utils.scale_coords(self.final_shape, output_nonmax[:, :4], self.ori_shape).round()
        ## 把检测到的框缩放到与原图大小对应的尺寸Rescale coords (xyxy) from final_shape to ori_shape
        
        if output_nonmax.shape[0] != 0:
            # pass
            scores = output_nonmax[:, 4]#获取置信度【读024.bmp时 经过非极大值抑制后没有目标，导致报错！！！
            scores = scores.unsqueeze(0)#          【读011.bmp时，三个目标只有一个由热力图】
            print('scores.shape=',scores.shape)
            # score = torch.max(scores)#获得最大置信度
            score,index = torch.max(scores,dim=1)
            idx = scores.argmax().cpu().numpy()#最大置信度对应的索引
            # one_hot_output = torch.FloatTensor(1, scores.size()[-1]).zero_().cuda()#全0 
            one_hot_output = torch.zeros_like(scores)     
            one_hot_output[0][index] = 1#最大置信度索引处值为1，其他为0,【实验发现，最大或最小索引值为1，对结果没有影响】
            # one_hot_output[0][:] = 1#由于三个框对应三个目标，都有用，这里全设为1
            print('one_hot_output=',one_hot_output)
        

            self.net.zero_grad()
            
            scores.backward(gradient=one_hot_output, retain_graph = True)
            # scores.backward(gradient=one_hot_output)

            #排除非极大值易知
            # bbb = pred[..., 4].max()
            # bbb.backward()
           
            #如果使用.backward()的tensor是一个标量，就可以省略gradient参数，而如果这个tensor是向量，则必须指明gradient参数
            #如果是(向量)矩阵对(向量)矩阵求导(tensor对tensor求导)，实际上是先求出Jacobian矩阵中每一个元素的梯度值(每一个元素
            # 的梯度值的求解过程对应上面的计算图的求解方法)，然后将这个Jacobian矩阵与grad_tensors参数对应的矩阵进行对应的点乘，
            # 得到最终的结果。
            #参考1：https://blog.csdn.net/Konge4/article/details/114955821 pytorch中backward函数的参数gradient作用的数学过程
            #参考2：https://www.cnblogs.com/zhouyang209117/p/11023160.html

            # self.gradient = self.net.module.get_activations_gradient()#获得特征层梯度
            # self.feature = self.net.module.get_activations_features()#获得特征层输出

            gradient = self.gradient.cpu().data.numpy()#
            feature=self.feature.cpu().data.numpy()#
            #测试
            # print(self.gradient)
            #gradient_tensor = torch.tensor(np.array(self.gradient[2]))
            #pooled_gradients = torch.mean(gradient_tensor, dim=[0, 2, 3])
            
            # target = self.feature[0].cpu().detach().numpy()[0]  # 【经检验，这里target和输入x完全一样，添加hook错误！！！不是想要的特定层的输出】
            # guided_gradients = self.gradient.cpu().detach().numpy()[0]
            target = feature[0] #
            guided_gradients = gradient[0] #

            #--------Grad-CAM系数---------#        
            weights = np.mean(guided_gradients, axis = (1, 2))  # take averages for each gradient
            #--------Grad-CAM系数---------# 

            #--------Grad-CAM++系数---------#
            # grads_power_2 = guided_gradients**2
            # grads_power_3 = grads_power_2*guided_gradients
            # # Equation 19 in https://arxiv.org/abs/1710.11063
            # sum_activations = np.sum(target, axis=(1, 2))
            # eps = 0.000001
            # aij = grads_power_2 / (2*grads_power_2 + 
            #     sum_activations[:,  None, None]*grads_power_3 + eps)
            # # Now bring back the ReLU from eq.7 in the paper,
            # # And zero out aijs where the activations are 0
            # aij = np.where(guided_gradients != 0, aij, 0)

            # weights_pp = np.maximum(guided_gradients, 0)*aij
            # weights_pp = np.sum(weights_pp, axis=(1, 2)) #weights of grad-cam++
            #--------Grad-CAM++系数---------#  

            # create empty numpy array for cam
            cam = np.ones(target.shape[1:], dtype = np.float32)#
            
            # multiply each grad-cam weight with its conv output and then, sum
            for i, w in enumerate(weights-1):
                cam =cam+ w * target[i, :, :]

            # multiply each grad-cam++ weight with its conv output and then, sum
            # for i, w in enumerate(weights_pp-1):
            #     cam =cam+ w * target[i, :, :]

            cam = np.maximum(cam, 0)#RELU

            if cam.max()==0 and cam.min()==0:
                with open(os.path.join('yolov3', "log_without_cam" + ".txt"), 'a') as f:
                    path=inputs['path'].split("\\")[-1].split(".")[0]
                    f.write(path)
                    f.write("\n")

            cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # normalize between 0-1
            # comment this line if use colormap
            # cam = 255 - cam
            # comment this line if use pixel matshow, otherwise cancel the comment
            cam = np.uint8(cam * 255)  # scale between 0-255 to visualize
            cam_tmp = cam.copy() #未缩放前的cam,【输出cam可知，hook特征层输出尺寸为:
                        #torch.Size([1, 512, 30, 30])，heatmap区域只有一个点，
                        # 这个点需要插值为原图像尺寸】            
            #
            
            '''
            cam = np.uint8(Image.fromarray(cam).resize((self.ori_shape[1],self.ori_shape[0]), Image.ANTIALIAS))/255
            
            original_image = Image.open('./img/4.png')
            I_array = np.array(original_image)
            original_image = Image.fromarray(I_array.astype("uint8"))
            save_class_activation_images(original_image, cam, 'cam-featuremap')
            '''           
            ################################## 
            # This is for pixel matplot method
            ##################################
            #把cam缩放到原图尺寸，采用Image.ANTIALIAS、Image.LANCZOS会出现旁瓣，Image.NEAREST不会出现旁瓣，但效果也不好，Image.BOX与Image.NEAREST类似
            # Image.BILINEAR、Image.HAMMING、Image.BICUBIC效果好些！综合来看可选择Image.BICUBIC插值方法
            # cam = np.uint8(Image.fromarray(cam).resize((self.ori_shape[1],self.ori_shape[0]), Image.ANTIALIAS))/255
            cam = np.uint8(Image.fromarray(cam).resize((self.ori_shape[1],self.ori_shape[0]), Image.BICUBIC))/255
            
            cam_all_255 = cam.copy()
            cam_all_255[cam != 0]=255  #把掩模中不为0的值全部设为255，可以避免cam插值为原图尺寸时出现的递减情况

            test_img = inputs['origin_image']#cv2.imread('./img/1.png')
            heatmap = cam.astype(np.float32)#shape:(480, 640)
            # heatmap = cv2.resize(heatmap, (test_img.shape[1], test_img.shape[0]))#zy:比赛图片经过特殊处理，不需要resize
            heatmap = np.uint8(255 * heatmap)#np.uint8()函数，但是这个函数仅仅是对原数据和0xff相与(和最低2字节数据相与)，这就容易导致如果原数据是大于255的，
                                            #那么在直接使用np.uint8()后，比第八位更大的数据都被截断了，比如:np.uint8(2000) =208
            cam_mask=heatmap #黑白heatmap
            heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)#彩色化heatmap
            # superimposed_img = heatmap  + test_img
            superimposed_img = heatmap * 0.6 + test_img #和原图叠加后的cam
            cv2.imwrite('CAM_OUT\composed_img\composed_img_{}.bmp'.format(str(path)), superimposed_img)
            cv2.imwrite('CAM_OUT\heatmap\heatmap_{}.bmp'.format(str(path)), heatmap)
            cv2.imwrite('CAM_OUT\cam_mask\cam_mask_{}.bmp'.format(str(path)), cam_mask)
            # cv2.imwrite('./image_mask.jpg', cam_mask[:,:,np.newaxis]*test_img)
            # cv2.imwrite('./ori_cam.jpg', cam_tmp)
            cv2.imwrite('CAM_OUT\cam_mask_all255\cam_mask_all255_{}.bmp'.format(str(path)), cam_all_255)
            

            # box = output_nonmax[idx][:4].cpu().detach().numpy().astype(np.int32)
            box = output_nonmax[:,:4].cpu().detach().numpy().astype(np.int32)

            '''
            #x1, y1, x2, y2 = box
            x1, y1, x2, y2 = box[:,0],box[:,1],box[:,2],box[:,3]
            ratio_x1 = x1 / test_img.shape[1]
            ratio_x2 = x2 / test_img.shape[0]
            ratio_y1 = y1 / test_img.shape[1]
            ratio_y2 = y2 / test_img.shape[0]
            # x1_cam = int(cam.shape[1] * ratio_x1)
            x1_cam = (cam.shape[1] * ratio_x1).astype(np.int32)   #python中标量可直接用int(a)进行类型转换，否则可用astype()
            x2_cam = (cam.shape[0] * ratio_x2).astype(np.int32)
            y1_cam = (cam.shape[1] * ratio_y1).astype(np.int32)
            y2_cam = (cam.shape[0] * ratio_y2).astype(np.int32)

            cam = cam[y1_cam:y2_cam, x1_cam:x2_cam]   #【报错！只有整数标量才能作为索引，后面用处不是很大，暂时未调试】
            cam = cv2.resize(cam, (x2 - x1, y2 - y1))
            '''
            # class_id = output[idx][-1].cpu().detach().numpy()
            class_id = output_nonmax[:, -1].cpu().detach().numpy()
            return cam, box, class_id
        
        else:
            print("cannot detect a car with the model!")
            return None,None,None

        














        '''
        pooled_gradients = torch.mean(self.gradient, dim=[0, 2, 3])
   
        gradient = self.gradient.cpu().data.numpy()  # [C,H,W]

        weight = np.mean(gradient, axis=(1, 2))  # [C]

        #self.feature = self.net.features
        #print(self.feature)
        #feature = self.feature[proposal_idx].cpu().data.numpy()  # [C,H,W]

        #feature = self.feature[idx].cpu().data
        feature = self.feature[0].data.numpy()
        print(pooled_gradients)
        print(torch.any(pooled_gradients != 0))
        print(pooled_gradients.shape)
        print(weight.shape)
        print(feature.shape)


        
        # pool the gradients across the channels
        #pooled_gradients = np.mean(gradient, dim=[0, 2, 3])

        # get the activations of the last convolutional layer
        activations = self.feature[0].detach()
        # weight the channels by corresponding gradients
        for i in range(26):
            #activations[:, i, :, :] *= weight[i]
            activations[:, i, :, :] *= pooled_gradients[i]

        print(activations.shape)
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()

        
        # relu on top of the heatmap
        # expression (2) in https://arxiv.org/pdf/1610.02391.pdf
        heatmap = np.maximum(heatmap, 0)

        # normalize the heatmap
        heatmap /= torch.max(heatmap)

        # draw the heatmap
        plt.matshow(heatmap.squeeze())
        plt.show()

        
        test_img = cv2.imread('./test7.png')
        heatmap = heatmap.numpy().astype(np.float32)
        heatmap = cv2.resize(heatmap, (test_img.shape[1], test_img.shape[0]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = heatmap * 0.6 + test_img
        cv2.imwrite('./map.jpg', superimposed_img)
        '''
        
