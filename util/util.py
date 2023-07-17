# util.py
# 用于进行向量角计算
import math
import numpy as np

from util.pose.head_pose import HeadPoseCHeck
from util.pose.rise_hand import RiseHandCheck
from util.pose.site_stand import SiteStandCHeck


class Util:
    # 右侧脚踝-右侧膝盖
    # 右侧膝盖-右侧胯
    # 左侧脚踝-左侧膝盖
    # 左侧膝盖-左侧胯
    # 右侧胯-左侧胯
    # 右边肩膀-右侧胯
    # 左边肩膀-左侧胯
    # 右边肩膀-左边肩膀
    # 右边肩膀-右侧胳膊肘
    # 左边肩膀-左侧胳膊肘
    # 右侧胳膊肘-右侧手腕
    # 左侧胳膊肘-左侧手腕
    # 右边眼睛-左边眼睛
    # 鼻尖-左边眼睛
    # 鼻尖-左边眼睛
    # 右边眼睛-右边耳朵
    # 左边眼睛-左边耳朵
    # 右边耳朵-右边肩膀
    # 左边耳朵-左边肩膀
    poseLineDatas = []
    headPoseUtil = HeadPoseCHeck()
    siteStandUtil = SiteStandCHeck()
    riseHandCheck = RiseHandCheck()

    def __init__(self):
        pass

    def addPoseLineData(self, x, y, x1, y1):
        self.poseLineDatas.append((x, y, x1, y1))
        # print("add body line data")

    def startCaculate(self):
        print("开始计算 当前帧有", len(self.poseLineDatas), "条线")
        #判断是站立还是坐着
        standOrSit = self.siteStandUtil.checkStandOrSitPose(self.poseLineDatas[2],self.poseLineDatas[3])

        #判断是否举手
        riseHand = self.riseHandCheck.checkRiseHand(self.poseLineDatas[8],self.poseLineDatas[10])

        #判断头部姿态
        # 计算左边眼睛-左边耳朵 与 左边耳朵-左边肩膀 的向量角度
        headPose =  self.headPoseUtil.checkHeadPose(self.poseLineDatas[16], self.poseLineDatas[18])
        print("头部姿态:",headPose)
        self.poseLineDatas = []


