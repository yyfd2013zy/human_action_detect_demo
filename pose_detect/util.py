# pose_detect.py
# 用于进行向量角计算
import math
import numpy as np

from pose_detect.pose.head_pose import HeadPoseCHeck
from pose_detect.pose.rise_hand import RiseHandCheck
from pose_detect.pose.site_stand import SiteStandCHeck


class Util:
    # 0  右侧脚踝-右侧膝盖
    # 1  右侧膝盖-右侧胯
    # 2  左侧脚踝-左侧膝盖
    # 3  左侧膝盖-左侧胯
    # 4  右侧胯-左侧胯
    # 5  右边肩膀-右侧胯
    # 6  左边肩膀-左侧胯
    # 7  右边肩膀-左边肩膀
    # 8  右边肩膀-右侧胳膊肘
    # 9  左边肩膀-左侧胳膊肘
    # 10 右侧胳膊肘-右侧手腕
    # 11 左侧胳膊肘-左侧手腕
    # 12 右边眼睛-左边眼睛
    # 13 鼻尖-左边眼睛
    # 14 鼻尖-左边眼睛
    # 15 右边眼睛-右边耳朵
    # 16 左边眼睛-左边耳朵
    # 17 右边耳朵-右边肩膀
    # 18 左边耳朵-左边肩膀
    poseLineDatas = []
    headPoseUtil = HeadPoseCHeck()
    siteStandUtil = SiteStandCHeck()
    riseHandCheck = RiseHandCheck()

    def __init__(self):
        pass

    def addPoseLineData(self, x, y, x1, y1):
        self.poseLineDatas.append((x, y, x1, y1))
        # print("add body line data")

    def startCaculate(self,time_second, callback):
        print("开始计算 当前帧有", len(self.poseLineDatas), "条线")
        # 判断是站立还是坐着
        standOrSit = self.siteStandUtil.checkStandOrSitPose(self.poseLineDatas[2], self.poseLineDatas[3])

        # 判断是否举手
        riseHand = self.riseHandCheck.checkRiseHand(self.poseLineDatas[8], self.poseLineDatas[10],self.poseLineDatas[9],self.poseLineDatas[11])

        # 判断头部姿态
        # 计算左边眼睛-左边耳朵 与 左边耳朵-左边肩膀 的向量角度
        headPose = self.headPoseUtil.checkHeadPose(self.poseLineDatas[16], self.poseLineDatas[18])
        print("头部姿态:", headPose)
        callback(standOrSit,headPose,riseHand)
        self.poseLineDatas = []
