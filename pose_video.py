import cv2
import numpy as np
import time
from tqdm import tqdm

from ultralytics import YOLO

import matplotlib.pyplot as plt

import torch

from objecttracking.sort import Sort
from pose_detect.calculate_util import CaculateUtil
from pose_detect.util import Util


# 识别视频中人体关键点，并绘制点以及线
# 有 GPU 就用 GPU，没有就用 CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('device:', device)
# 载入预训练模型
# model = YOLO('yolov8n-pose.pt')
# model = YOLO('yolov8s-pose.pt')
# model = YOLO('yolov8m-pose.pt')
# model = YOLO('yolov8l-pose.pt')
model = YOLO('yolov8x-pose.pt')
# model = YOLO('yolov8n-pose.pt')
# 切换计算设备
model.to(device)
# 框（rectangle）可视化配置
bbox_color = (150, 0, 0)  # 框的 BGR 颜色
bbox_thickness = 2  # 框的线宽
utils = Util()
# 框类别文字
bbox_labelstr = {
    'font_size': 1,  # 字体大小
    'font_thickness': 2,  # 字体粗细
    'offset_x': 0,  # X 方向，文字偏移距离，向右为正
    'offset_y': -10,  # Y 方向，文字偏移距离，向下为正
}

tracker = Sort()

# 关键点 BGR 配色
kpt_color_map = {
    0: {'name': 'Nose', 'color': [0, 0, 255], 'radius': 6},  # 鼻尖
    1: {'name': 'Right Eye', 'color': [255, 0, 0], 'radius': 6},  # 右边眼睛
    2: {'name': 'Left Eye', 'color': [255, 0, 0], 'radius': 6},  # 左边眼睛
    3: {'name': 'Right Ear', 'color': [0, 255, 0], 'radius': 6},  # 右边耳朵
    4: {'name': 'Left Ear', 'color': [0, 255, 0], 'radius': 6},  # 左边耳朵
    5: {'name': 'Right Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 右边肩膀
    6: {'name': 'Left Shoulder', 'color': [193, 182, 255], 'radius': 6},  # 左边肩膀
    7: {'name': 'Right Elbow', 'color': [16, 144, 247], 'radius': 6},  # 右侧胳膊肘
    8: {'name': 'Left Elbow', 'color': [16, 144, 247], 'radius': 6},  # 左侧胳膊肘
    9: {'name': 'Right Wrist', 'color': [1, 240, 255], 'radius': 6},  # 右侧手腕
    10: {'name': 'Left Wrist', 'color': [1, 240, 255], 'radius': 6},  # 左侧手腕
    11: {'name': 'Right Hip', 'color': [140, 47, 240], 'radius': 6},  # 右侧胯
    12: {'name': 'Left Hip', 'color': [140, 47, 240], 'radius': 6},  # 左侧胯
    13: {'name': 'Right Knee', 'color': [223, 155, 60], 'radius': 6},  # 右侧膝盖
    14: {'name': 'Left Knee', 'color': [223, 155, 60], 'radius': 6},  # 左侧膝盖
    15: {'name': 'Right Ankle', 'color': [139, 0, 0], 'radius': 6},  # 右侧脚踝
    16: {'name': 'Left Ankle', 'color': [139, 0, 0], 'radius': 6},  # 左侧脚踝
}

# 点类别文字
kpt_labelstr = {
    'font_size': 0.5,  # 字体大小
    'font_thickness': 1,  # 字体粗细
    'offset_x': 10,  # X 方向，文字偏移距离，向右为正
    'offset_y': 0,  # Y 方向，文字偏移距离，向下为正
}

# 骨架连接 BGR 配色
skeleton_map = [
    {'srt_kpt_id': 15, 'dst_kpt_id': 13, 'color': [0, 100, 255], 'thickness': 2},  # 右侧脚踝-右侧膝盖
    {'srt_kpt_id': 13, 'dst_kpt_id': 11, 'color': [0, 255, 0], 'thickness': 2},  # 右侧膝盖-右侧胯
    {'srt_kpt_id': 16, 'dst_kpt_id': 14, 'color': [255, 0, 0], 'thickness': 2},  # 左侧脚踝-左侧膝盖
    {'srt_kpt_id': 14, 'dst_kpt_id': 12, 'color': [0, 0, 255], 'thickness': 2},  # 左侧膝盖-左侧胯
    {'srt_kpt_id': 11, 'dst_kpt_id': 12, 'color': [122, 160, 255], 'thickness': 2},  # 右侧胯-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 11, 'color': [139, 0, 139], 'thickness': 2},  # 右边肩膀-右侧胯
    {'srt_kpt_id': 6, 'dst_kpt_id': 12, 'color': [237, 149, 100], 'thickness': 2},  # 左边肩膀-左侧胯
    {'srt_kpt_id': 5, 'dst_kpt_id': 6, 'color': [152, 251, 152], 'thickness': 2},  # 右边肩膀-左边肩膀
    {'srt_kpt_id': 5, 'dst_kpt_id': 7, 'color': [148, 0, 69], 'thickness': 2},  # 右边肩膀-右侧胳膊肘
    {'srt_kpt_id': 6, 'dst_kpt_id': 8, 'color': [0, 75, 255], 'thickness': 2},  # 左边肩膀-左侧胳膊肘
    {'srt_kpt_id': 7, 'dst_kpt_id': 9, 'color': [56, 230, 25], 'thickness': 2},  # 右侧胳膊肘-右侧手腕
    {'srt_kpt_id': 8, 'dst_kpt_id': 10, 'color': [0, 240, 240], 'thickness': 2},  # 左侧胳膊肘-左侧手腕
    {'srt_kpt_id': 1, 'dst_kpt_id': 2, 'color': [224, 255, 255], 'thickness': 2},  # 右边眼睛-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 1, 'color': [47, 255, 173], 'thickness': 2},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 0, 'dst_kpt_id': 2, 'color': [203, 192, 255], 'thickness': 2},  # 鼻尖-左边眼睛
    {'srt_kpt_id': 1, 'dst_kpt_id': 3, 'color': [196, 75, 255], 'thickness': 2},  # 右边眼睛-右边耳朵
    {'srt_kpt_id': 2, 'dst_kpt_id': 4, 'color': [86, 0, 25], 'thickness': 2},  # 左边眼睛-左边耳朵
    {'srt_kpt_id': 3, 'dst_kpt_id': 5, 'color': [255, 255, 0], 'thickness': 2},  # 右边耳朵-右边肩膀
    {'srt_kpt_id': 4, 'dst_kpt_id': 6, 'color': [255, 18, 200], 'thickness': 2}  # 左边耳朵-左边肩膀
]


def process_frame(time_second, fps, img_bgr):
    print("==========开始处理")

    # 在视频左上角显示出当前的秒数
    text_position = (20, 50)  # 文字位置坐标 (x, y)
    text_color = (0, 255, 0)  # 文字颜色为绿色
    font_scale = 1  # 字体大小缩放因子
    font_thickness = 2  # 字体线宽

    img_bgr = cv2.putText(img_bgr,
                          str(time_second),
                          text_position,
                          cv2.FONT_HERSHEY_SIMPLEX,
                          font_scale,
                          text_color,
                          font_thickness)

    '''
    输入摄像头画面 bgr-array，输出图像 bgr-array
    '''
    results = model(img_bgr, verbose=False)  # verbose设置为False，不单独打印每一帧预测结果

    # 预测框的个数-人数
    num_bbox = len(results[0].boxes.cls)
    print("当前人数", num_bbox)

    # 预测框的 xyxy 坐标
    bboxes_xyxy = results[0].boxes.xyxy.cpu().numpy().astype('uint32')

    # 关键点的 xy 坐标
    bboxes_keypoints = results[0].keypoints.data.cpu().numpy().astype('uint32')

    # 人体追踪
    # filtered_indices = np.where(results[0].boxes.conf.cpu().numpy() > 0.5)[0]
    # 筛选出置信度大于5的人体框
    # boxes = results[0].boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
    # 暂时不筛选
    # 取出
    tracks = tracker.update(bboxes_xyxy.astype(int))
    tracks = tracks.astype(int)

    for xmin, ymin, xmax, ymax, track_id in tracks:
        img_bgr = cv2.putText(img_bgr, text=f"Id: {track_id}", org=(xmin, ymin - 55), fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=2, color=(0, 255, 0), thickness=2)
        # img_bgr = cv2.rectangle(img_bgr, pt1=(xmin, ymin), pt2=(xmax, ymax), color=(0, 255, 0), thickness=2)

    for idx in range(num_bbox):  # 遍历每个框
        bbox_keypoints = bboxes_keypoints[idx]  # 该框所有关键点坐标和置信度

        # vinda 自定义数据
        headposeStatus = ""  # 头部状态
        sitOrStandStatus = ""  # 站立还是坐下
        riseHandStatus = ""  # 是否举手
        lineIndex = 0

        # 获取当前人员框的的追踪ID 根据人体框的近似值进行取值
        print(f"此人框数据 {bboxes_xyxy[idx]}")
        id_list = []
        other_arrays = []
        for xmin, ymin, xmax, ymax, track_id in tracks:
            print(f"遍历追踪ID数据 track_id:{track_id} xmin:{xmin} ymin:{ymin} xmax:{xmax} ymax:{ymax}")
            this_list = []
            this_list.append(xmin)
            this_list.append(ymin)
            this_list.append(xmax)
            this_list.append(ymax)
            id_list.append(track_id)
            other_arrays.append(this_list)

        #判断最接近的素组
        closest_idx =CaculateUtil.find_closest_array(bboxes_xyxy[idx], other_arrays)
        if closest_idx!=-1:
            print(f"最接近数组:{closest_idx} 人员id:{id_list[closest_idx]}")
        else:
            print(f"最接近数组:{closest_idx} 人员id丢失")

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

            #这里循环遍历了单个人的骨架
            utils.addPoseLineData(srt_kpt_x, srt_kpt_y, dst_kpt_x, dst_kpt_y)

            def my_callback(sitOrStand, headAction, riseHand):
                # 在回调函数中处理接收到的参数-tips:这里1s回调一次
                nonlocal headposeStatus
                nonlocal sitOrStandStatus
                nonlocal riseHandStatus
                headposeStatus = headAction
                sitOrStandStatus = sitOrStand
                riseHandStatus = riseHand

            # 最后一个数据处理完毕,开始进行向量角计算
            if lineIndex == len(skeleton_map):
                utils.startCaculate(id_list[closest_idx],time_second,fps,my_callback)

            # 获取骨架连接颜色
            skeleton_color = skeleton['color']

            # 获取骨架连接线宽
            skeleton_thickness = skeleton['thickness']

            # 画骨架连接
            img_bgr = cv2.line(img_bgr, (srt_kpt_x, srt_kpt_y), (dst_kpt_x, dst_kpt_y), color=skeleton_color,
                               thickness=skeleton_thickness)

        # ------------------------------------------------绘制人体框-----------------------------------------
        # 获取该框坐标
        bbox_xyxy = bboxes_xyxy[idx]
        # 获取框的预测类别（对于关键点检测，只有一个类别）
        bbox_label = results[0].names[0]
        # 画框
        img_bgr = cv2.rectangle(img_bgr, (bbox_xyxy[0], bbox_xyxy[1]), (bbox_xyxy[2], bbox_xyxy[3]), bbox_color,
                                bbox_thickness)
        # 显示是否抬头
        img_bgr = cv2.putText(img_bgr,
                              headposeStatus,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'], bbox_xyxy[1] + bbox_labelstr['offset_y']),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              bbox_labelstr['font_size'],
                              bbox_color,
                              bbox_labelstr['font_thickness'])
        # 显示站立还是坐下
        img_bgr = cv2.putText(img_bgr,
                              sitOrStandStatus,
                              (bbox_xyxy[0] + bbox_labelstr['offset_x'],
                               bbox_xyxy[1] + bbox_labelstr['offset_y'] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX,
                              bbox_labelstr['font_size'],
                              bbox_color,
                              bbox_labelstr['font_thickness'])

        # 是否举手
        if (riseHandStatus != "none"):
            img_bgr = cv2.putText(img_bgr,
                                  riseHandStatus,
                                  (bbox_xyxy[0] + bbox_labelstr['offset_x'],
                                   bbox_xyxy[1] + bbox_labelstr['offset_y'] - 40),
                                  cv2.FONT_HERSHEY_SIMPLEX,
                                  bbox_labelstr['font_size'],
                                  bbox_color,
                                  bbox_labelstr['font_thickness'])

        # 画骨骼关键点
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
    print("==========处理完毕")
    return img_bgr


# 视频逐帧处理代码模板
# 不需修改任何代码，只需定义process_frame函数即可
# 同济子豪兄 2021-7-10

def generate_video(input_path='video/1.mp4'):
    filehead = input_path.split('/')[-1]
    output_path = "out-" + filehead

    print('视频开始处理', input_path)

    # 获取视频总帧数
    cap = cv2.VideoCapture(input_path)
    frame_count = 0
    while (cap.isOpened()):
        success, frame = cap.read()
        frame_count += 1
        if not success:
            break
    cap.release()
    print('视频总帧数为', frame_count)

    # cv2.namedWindow('Crack Detection and Measurement Video Processing')
    cap = cv2.VideoCapture(input_path)
    frame_size = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # 获取视频帧率
    fps = cap.get(cv2.CAP_PROP_FPS)

    # 计算视频总时长
    duration = frame_count / fps
    print(f"视频帧率:{fps} 视频总时长:{duration} 秒")

    out = cv2.VideoWriter(output_path, fourcc, fps, (int(frame_size[0]), int(frame_size[1])))

    # 进度条绑定视频总帧数
    with tqdm(total=frame_count - 1) as pbar:
        try:
            while (cap.isOpened()):
                success, frame = cap.read()
                if not success:
                    break

                # 处理帧
                frame_path = './temp_frame.png'
                cv2.imwrite(frame_path, frame)
                try:
                    time_per_frame = 1 / fps
                    time_in_seconds = int(pbar.n * time_per_frame)
                    print(f"分析此帧数据：{time_in_seconds} 秒")
                    frame = process_frame(time_in_seconds,fps, frame)
                except:
                    print('error')
                    pass

                if success == True:
                    # 将实时帧处理结果显示出来
                    cv2.imshow('Video Processing', frame)
                    out.write(frame)

                    # 进度条更新一帧
                    pbar.update(1)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except:
            print('中途中断')
            pass

    cv2.destroyAllWindows()
    out.release()
    cap.release()
    print('视频已保存', output_path)


generate_video(input_path='video/Ip22.mp4')
