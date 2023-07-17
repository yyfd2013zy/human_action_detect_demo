import cv2
import numpy as np
import time

from matplotlib import pyplot as plt
from tqdm import tqdm

from ultralytics import YOLO


import torch

from util.util import Util

# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
model = YOLO('yolov8x-pose.pt')
model.to(device)
# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)             # 框的 BGR 颜色
bbox_thickness = 2                   # 框的线宽
utils = Util()
# 框类别文字
bbox_labelstr = {
    'font_size':1,         # 字体大小
    'font_thickness':2,   # 字体粗细
    'offset_x':0,          # X 方向，文字偏移距离，向右为正
    'offset_y':-20,       # Y 方向，文字偏移距离，向下为正
}
# 关键点 BGR 配色
pointRadia = 5
kpt_color_map = {
    0:{'name':'Nose', 'color':[0, 0, 255], 'radius':pointRadia},                # 鼻尖
    1:{'name':'Right Eye', 'color':[255, 0, 0], 'radius':pointRadia},           # 右边眼睛
    2:{'name':'Left Eye', 'color':[255, 0, 0], 'radius':pointRadia},            # 左边眼睛
    3:{'name':'Right Ear', 'color':[0, 255, 0], 'radius':pointRadia},           # 右边耳朵
    4:{'name':'Left Ear', 'color':[0, 255, 0], 'radius':pointRadia},            # 左边耳朵
    5:{'name':'Right Shoulder', 'color':[193, 182, 255], 'radius':pointRadia},  # 右边肩膀
    6:{'name':'Left Shoulder', 'color':[193, 182, 255], 'radius':pointRadia},   # 左边肩膀
    7:{'name':'Right Elbow', 'color':[16, 144, 247], 'radius':pointRadia},      # 右侧胳膊肘
    8:{'name':'Left Elbow', 'color':[16, 144, 247], 'radius':pointRadia},       # 左侧胳膊肘
    9:{'name':'Right Wrist', 'color':[1, 240, 255], 'radius':pointRadia},       # 右侧手腕
    10:{'name':'Left Wrist', 'color':[1, 240, 255], 'radius':pointRadia},       # 左侧手腕
    11:{'name':'Right Hip', 'color':[140, 47, 240], 'radius':pointRadia},       # 右侧胯
    12:{'name':'Left Hip', 'color':[140, 47, 240], 'radius':pointRadia},        # 左侧胯
    13:{'name':'Right Knee', 'color':[223, 155, 60], 'radius':pointRadia},      # 右侧膝盖
    14:{'name':'Left Knee', 'color':[223, 155, 60], 'radius':pointRadia},       # 左侧膝盖
    15:{'name':'Right Ankle', 'color':[139, 0, 0], 'radius':pointRadia},        # 右侧脚踝
    16:{'name':'Left Ankle', 'color':[139, 0, 0], 'radius':pointRadia},         # 左侧脚踝
}

# 点类别文字
kpt_labelstr = {
    'font_size':1,             # 字体大小
    'font_thickness':1,       # 字体粗细
    'offset_x':10,             # X 方向，文字偏移距离，向右为正
    'offset_y':0,            # Y 方向，文字偏移距离，向下为正
}

# 骨架连接 BGR 配色
lineWidth = 2
skeleton_map = [
    {'srt_kpt_id':15, 'dst_kpt_id':13, 'color':[0, 100, 255], 'thickness':lineWidth},       # 右侧脚踝-右侧膝盖
    {'srt_kpt_id':13, 'dst_kpt_id':11, 'color':[0, 255, 0], 'thickness':lineWidth},         # 右侧膝盖-右侧胯
    {'srt_kpt_id':16, 'dst_kpt_id':14, 'color':[255, 0, 0], 'thickness':lineWidth},         # 左侧脚踝-左侧膝盖
    {'srt_kpt_id':14, 'dst_kpt_id':12, 'color':[0, 0, 255], 'thickness':lineWidth},         # 左侧膝盖-左侧胯
    {'srt_kpt_id':11, 'dst_kpt_id':12, 'color':[122, 160, 255], 'thickness':lineWidth},     # 右侧胯-左侧胯
    {'srt_kpt_id':5, 'dst_kpt_id':11, 'color':[139, 0, 139], 'thickness':lineWidth},        # 右边肩膀-右侧胯
    {'srt_kpt_id':6, 'dst_kpt_id':12, 'color':[237, 149, 100], 'thickness':lineWidth},      # 左边肩膀-左侧胯
    {'srt_kpt_id':5, 'dst_kpt_id':6, 'color':[152, 251, 152], 'thickness':lineWidth},       # 右边肩膀-左边肩膀
    {'srt_kpt_id':5, 'dst_kpt_id':7, 'color':[148, 0, 69], 'thickness':lineWidth},          # 右边肩膀-右侧胳膊肘
    {'srt_kpt_id':6, 'dst_kpt_id':8, 'color':[0, 75, 255], 'thickness':lineWidth},          # 左边肩膀-左侧胳膊肘
    {'srt_kpt_id':7, 'dst_kpt_id':9, 'color':[56, 230, 25], 'thickness':lineWidth},         # 右侧胳膊肘-右侧手腕
    {'srt_kpt_id':8, 'dst_kpt_id':10, 'color':[0,240, 240], 'thickness':lineWidth},         # 左侧胳膊肘-左侧手腕
    {'srt_kpt_id':1, 'dst_kpt_id':2, 'color':[224,255, 255], 'thickness':lineWidth},        # 右边眼睛-左边眼睛
    {'srt_kpt_id':0, 'dst_kpt_id':1, 'color':[47,255, 173], 'thickness':lineWidth},         # 鼻尖-左边眼睛
    {'srt_kpt_id':0, 'dst_kpt_id':2, 'color':[203,192,255], 'thickness':lineWidth},         # 鼻尖-左边眼睛
    {'srt_kpt_id':1, 'dst_kpt_id':3, 'color':[196, 75, 255], 'thickness':lineWidth},        # 右边眼睛-右边耳朵
    {'srt_kpt_id':2, 'dst_kpt_id':4, 'color':[86, 0, 25], 'thickness':lineWidth},           # 左边眼睛-左边耳朵
    {'srt_kpt_id':3, 'dst_kpt_id':5, 'color':[255,255, 0], 'thickness':lineWidth},          # 右边耳朵-右边肩膀
    {'srt_kpt_id':4, 'dst_kpt_id':6, 'color':[255, 18, 200], 'thickness':lineWidth}         # 左边耳朵-左边肩膀
]


def process_frame(img_bgr):
    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''

    # 记录该帧开始处理的时间
    start_time = time.time()

    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果

    # 预测框的个数
    num_bbox = len(results[0].boxes.cls)

    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')

    for idx in range(num_bbox):  # 遍历每个框

        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]

        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]

        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)

        # 写框类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
        img_bgr = cv2.putText(img_bgr,
                              bbox_label,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              bbox_labelstr['font_size'],
                              bbox_color,
                              bbox_labelstr['font_thickness'])

        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度


        lineIndex = 0
        # 画该框的骨架连接
        for skeleton in skeleton_map:
            lineIndex += 1
            # 获取起始点坐标
            srt_kpt_id = skeleton['srt_kpt_id']
            srt_kpt_x = bbox_keypoints[srt_kpt_id][0]
            srt_kpt_y = bbox_keypoints[srt_kpt_id][1]

            # 获取终止点坐标
            dst_kpt_id = skeleton['dst_kpt_id']
            dst_kpt_x = bbox_keypoints[dst_kpt_id][0]
            dst_kpt_y = bbox_keypoints[dst_kpt_id][1]

            utils.addPoseLineData(srt_kpt_x, srt_kpt_y, dst_kpt_x, dst_kpt_y)

            # 最后一个数据处理完毕,开始进行向量角计算
            if lineIndex == len(skeleton_map):
                utils.startCaculate()

            # 获取骨架连接颜色
            skeleton_color = skeleton['color']

            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']

            # 画骨架连接
            img_bgr = cv2.line(img_bgr,
                               (srt_kpt_x, srt_kpt_y),
                               (dst_kpt_x, dst_kpt_y),
                               color=skeleton_color,
                               thickness=skeleton_thickness)

        # 画该框的关键点
        for kpt_id in kpt_color_map:
            # 获取该关键点的颜色、半径、XY坐标
            kpt_color = kpt_color_map[kpt_id]['color']
            kpt_radius = kpt_color_map[kpt_id]['radius']
            kpt_x = bbox_keypoints[kpt_id][0]
            kpt_y = bbox_keypoints[kpt_id][1]

            # 画圆：图片、XY坐标、半径、颜色、线宽（-1为填充）
            img_bgr = cv2.circle(img_bgr, (kpt_x, kpt_y), kpt_radius, kpt_color, -1)

            # 写关键点类别文字：图片，文字字符串，文字左上角坐标，字体，字体大小，颜色，字体粗细
            kpt_label = str(kpt_id)  # 写关键点类别 ID（二选一）
            # kpt_label = str(kpt_color_map[kpt_id]['name']) # 写关键点类别名称（二选一）
            img_bgr = cv2.putText(img_bgr, kpt_label,
                                  (kpt_x + kpt_labelstr['offset_x'], kpt_y + kpt_labelstr['offset_y']),
                                  cv2.FONT_HERSHEY_SIMPLEX, kpt_labelstr['font_size'], kpt_color,
                                  kpt_labelstr['font_thickness'])

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1 / (end_time - start_time)

    # 在画面上写字：图片，字符串，左上角坐标，字体，字体大小，颜色，字体粗细
    FPS_string = 'FPS  ' + str(int(FPS))  # 写在画面上的字符串
    img_bgr = cv2.putText(img_bgr, FPS_string, (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

    return img_bgr


# 获取摄像头，传入0表示获取系统默认摄像头
cap = cv2.VideoCapture(0)
# 设置分辨率
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
# 检查是否成功设置分辨率
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("当前Camera分辨率：{}x{}".format(width, height))

# 打开cap
cap.open(0)

# 创建一个窗口并命名为'my_window'
cv2.namedWindow('my_window', cv2.WINDOW_NORMAL)
# 设置窗口大小
cv2.resizeWindow('my_window', 800, 600)  # 将窗口大小设置为800x600像素
# 无限循环，直到break被触发
while cap.isOpened():

    # 获取画面
    success, frame = cap.read()

    if not success:  # 如果获取画面不成功，则退出
        print('获取画面不成功，退出')
        break

    ## 逐帧处理
    frame = process_frame(frame)

    # 展示处理后的三通道图像
    cv2.imshow('my_window', frame)

    key_pressed = cv2.waitKey(60)  # 每隔多少毫秒毫秒，获取键盘哪个键被按下
    # print('键盘上被按下的键：', key_pressed)

    if key_pressed in [ord('q'), 27]:  # 按键盘上的q或esc退出（在英文输入法下）
        break

# 关闭摄像头
cap.release()

# 关闭图像窗口
cv2.destroyAllWindows()



