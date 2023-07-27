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
    # 用于判断1s内的人员姿态
    timeRecord = -1
    # 1s内所有trackID
    bodyTrackidInOneSecond = {}
    # 1s内所有trackID对应坐下动作的数据次数
    siteInOneSecond = {}
    standInOneSecond = {}
    headUpInOneSecond = {}
    headFrontInOneSecond = {}
    headDownInOneSecond = {}

    def __init__(self):
        pass

    # track_id 当前这个人的id，tips:这个函数会在调用了19次，完整添加了一个人的所有骨骼连线后。调用startCaculate(开始对当前帧的姿态进行判断)
    def addPoseLineData(self, x, y, x1, y1):
        self.poseLineDatas.append((x, y, x1, y1))
        # print("add body line data")

    def startCaculate(self, track_id, time_second, fps, callback):
        """
           开始估计这一帧的人体姿态。

           参数：
           track_id : 当前追踪id
           time_second : 当前秒数
           fps :  当前视频帧数

           返回 :
           callback : 实时回调这一阵的动作姿态
        """
        print("开始计算 当前track_id:", track_id, " 当前秒数:", time_second, " 视频帧数:", fps)
        # 判断是站立还是坐着
        standOrSit = self.siteStandUtil.checkStandOrSitPose(self.poseLineDatas[2], self.poseLineDatas[3])

        # 判断是否举手
        riseHand = self.riseHandCheck.checkRiseHand(self.poseLineDatas[8], self.poseLineDatas[10],
                                                    self.poseLineDatas[9], self.poseLineDatas[11])

        # 判断头部姿态
        # 计算左边眼睛-左边耳朵 与 左边耳朵-左边肩膀 的向量角度
        headPose = self.headPoseUtil.checkHeadPose(self.poseLineDatas[16], self.poseLineDatas[18])
        print("头部姿态:", headPose)
        callback(standOrSit, headPose, riseHand)
        self.poseLineDatas = []

        if time_second != self.timeRecord:
            print("=========秒数发生改变===========")
            if time_second == 0:
                print("第0秒不处理")
            else:
                print("处理第", time_second - 1, "秒数据")
                print("追踪track id 数据", self.bodyTrackidInOneSecond)
                print("坐姿  数据", self.siteInOneSecond)
                print("站姿  数据", self.standInOneSecond)
                print("抬头  数据", self.headUpInOneSecond)
                print("平视  数据", self.headFrontInOneSecond)
                print("低头  数据", self.headDownInOneSecond)
                self.startCheckThisSecondPose()
            self.timeRecord = time_second
        else:
            print("封装当前帧数据")
            if track_id in self.bodyTrackidInOneSecond:
                # 如果存在，将对应的值加1
                self.bodyTrackidInOneSecond[track_id] += 1
            else:
                # 如果不存在，设置值为1
                self.bodyTrackidInOneSecond[track_id] = 1

            # 站立还是坐着姿态
            if standOrSit == "sit":  # 坐姿
                if track_id in self.siteInOneSecond:
                    # 如果存在，将对应的值加1
                    self.siteInOneSecond[track_id] += 1
                else:
                    # 如果不存在，设置值为1
                    self.siteInOneSecond[track_id] = 1
            elif standOrSit == "stand":  # 站姿
                if track_id in self.standInOneSecond:
                    # 如果存在，将对应的值加1
                    self.standInOneSecond[track_id] += 1
                else:
                    # 如果不存在，设置值为1
                    self.standInOneSecond[track_id] = 1
            # 头部姿态
            if headPose == "head up":  # 抬头
                if track_id in self.headUpInOneSecond:
                    # 如果存在，将对应的值加1
                    self.headUpInOneSecond[track_id] += 1
                else:
                    # 如果不存在，设置值为1
                    self.headUpInOneSecond[track_id] = 1
            elif headPose == "head front":  # 看向前方
                if track_id in self.headFrontInOneSecond:
                    # 如果存在，将对应的值加1
                    self.headFrontInOneSecond[track_id] += 1
                else:
                    # 如果不存在，设置值为1
                    self.headFrontInOneSecond[track_id] = 1
            elif headPose == "head down":  # 低头
                if track_id in self.headDownInOneSecond:
                    # 如果存在，将对应的值加1
                    self.headDownInOneSecond[track_id] += 1
                else:
                    # 如果不存在，设置值为1
                    self.headDownInOneSecond[track_id] = 1

    def startCheckThisSecondPose(self):
        """
          开始预估1s内的姿态数据
        """
        self.bodyTrackidInOneSecond = {}
        self.siteInOneSecond = {}
        self.standInOneSecond = {}
        self.headUpInOneSecond = {}
        self.headFrontInOneSecond = {}
        self.headDownInOneSecond = {}
        pass
