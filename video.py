# # # -*- coding: utf-8 -*-

# # import cv2
# # import os

# # def crop_video_by_width(input_video_path,out_video_path):
# #     # 判断视频是否存在
# #     if not os.path.exists(input_video_path):
# #         print('输入的视频文件不存在')

# #     # 获取

# #     video_read_cap = cv2.VideoCapture(input_video_path)

# #     input_video_width = int(video_read_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# #     input_video_height = int(video_read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# #     input_video_fps = int(video_read_cap.get(cv2.CAP_PROP_FPS))
# #     input_video_fourcc = int(cv2.VideoWriter_fourcc(*'XVID'))

# #     out_video_width = 512;
# #     out_video_height = 512;
# #     out_video_size = (int(out_video_width), int(out_video_height))

# #     video_write_cap = cv2.VideoWriter(out_video_path,input_video_fourcc,input_video_fps,out_video_size)

# #     while video_read_cap.isOpened():
# #         result, frame = video_read_cap.read()
# #         if not result:
# #             break

# #         # 裁剪到与原视频高度等宽的视频
# #         # diff = input_video_width - input_video_height
# #         input_video_width = int(input_video_width/4)
# #         diff = input_video_height - input_video_width
# #         diff = int(diff/2)
# #         crop_start_index = int(diff)
# #         crop_end_index = int(input_video_height - crop_start_index)
		
# # 		# 参数1 是高度的范围，参数2是宽度的范围
# #         target = frame[crop_start_index:crop_end_index,int(input_video_width*1.5):int(input_video_width*2.5)]

# #         # 再resize到512x512
# #         target = cv2.resize(target,(out_video_width,out_video_height))

# #         video_write_cap.write(target)
# #         cv2.imshow('target',target)
# #         cv2.waitKey(10)

# #     video_read_cap.release()
# #     video_write_cap.release()

# #     cv2.destroyAllWindows()

# import cv2
# import os

# def crop_video_by_width(input_video_path,out_video_path):
#     # 判断视频是否存在
#     if not os.path.exists(input_video_path):
#         print('输入的视频文件不存在')

#     # 获取

#     video_read_cap = cv2.VideoCapture(input_video_path)

#     input_video_width = int(video_read_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     input_video_height = int(video_read_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     input_video_fps = int(video_read_cap.get(cv2.CAP_PROP_FPS))
#     input_video_fourcc = int(cv2.VideoWriter_fourcc(*'mp4v'))

#     out_video_width = 512;
#     out_video_height = 512;
#     out_video_size = (int(out_video_width), int(out_video_height))

#     video_write_cap = cv2.VideoWriter(out_video_path,input_video_fourcc,input_video_fps,out_video_size)

#     while video_read_cap.isOpened():
#         result, frame = video_read_cap.read()
#         if not result:
#             break

#         # 裁剪到与原视频高度等宽的视频
#         diff = input_video_width - input_video_height
#         diff = int(diff/2)
#         crop_start_index = int(diff)
#         crop_end_index = int(diff + input_video_height)
		
# 		# 参数1 是高度的范围，参数2是宽度的范围
#         target = frame[0:int(input_video_height),crop_start_index:crop_end_index]

#         # 再resize到512x512
#         target = cv2.resize(target,(out_video_width,out_video_height))

#         video_write_cap.write(target)
#         cv2.imshow('target',target)
#         cv2.waitKey(10)

#     video_read_cap.release()
#     video_write_cap.release()

#     cv2.destroyAllWindows()


# if __name__ == '__main__':
#     crop_video_by_width('E:/python/kaiti/yolov3/data/DJI_0241.MP4', 'E:/python/kaiti/yolov3/data/video/result.mp4')

import os
import uuid
from ffmpy import FFmpeg
 
 
# 视频裁剪
def cut_out_video(path: str, output_dir: str, start_pix: tuple, size: tuple):
    video_path = os.listdir(path)
    for pth in video_path:
        pt = os.path.join(path, pth)
        ext = os.path.basename(pt).strip().split('.')[-1]
        if ext not in ['mp4','MP4', 'avi', 'flv']:
            raise Exception('format error')
        result = os.path.join(output_dir, '{}.{}'.format(uuid.uuid1().hex, ext))
        ff = FFmpeg(inputs={pt: None},
                    outputs={
                        result: '-vf crop={}:{}:{}:{} -y -threads 5 -preset ultrafast -strict -2'.format(size[0], size[1],
                                                                                                        start_pix[0],
                                                                                                        start_pix[1])})
        print(ff.cmd)
        ff.run()
    return result
 
 
if __name__ == '__main__':
    print(cut_out_video(r'E:/physical_attack/9.28/5', r'E:/python/kaiti/yolov5/data/9.28', (1800, 500), (960, 960)))#(1440, 600), (960, 960))
