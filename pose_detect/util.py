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

    def startCaculate(self, track_id, time_second, fps, one_frame_callback, one_second_callback):
        """
           开始估计这一帧的人体姿态。

           参数：
           track_id : 当前追踪id
           time_second : 当前秒数
           fps :  当前视频帧数

           返回 :
           one_frame_callback : 实时回调这一帧的动作姿态
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
        one_frame_callback(standOrSit, headPose, riseHand)
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
                """
                   !!!重要函数，判断当前这一s的人员以及动作
                """
                self.startCheckThisSecondPose(fps=fps, time=(time_second - 1), one_second_callback=one_second_callback)
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

    def startCheckThisSecondPose(self, fps, time, one_second_callback):
        """
          开始预估1s内的姿态数据
          threshold代表人体置信度-由百分比*帧率得出
        """
        # 判断的一些置信度的数值
        # 人体追踪稳定性置信度
        bodyTrackPercent = 0.6
        bodyTrackThreshold = fps * bodyTrackPercent
        # 坐姿和站立的置信度
        siteStandPercent = 0.6
        siteStandThreshold = fps * siteStandPercent
        # 头部姿态判断置信度小一些，因为头部姿态变化较快，较微小
        headPosePercent = 0.5
        headPosdThreshold = fps * headPosePercent

        # 此秒内的识别数据记录
        bodyTrackCount = 0
        siteSumCount = 0
        standSumCount = 0
        headUpSumCount = 0
        headFrontSumCount = 0
        headDownSumCount = 0

        print(f"<<<<<<<<<<<<<<<<<<<<< 开始分析第{time}s数据 fps:{fps} "
              f"\n percent:{bodyTrackPercent} "
              f"\n bodyThreshold:{bodyTrackThreshold}"
              f"\n siteStandThreshold:{siteStandThreshold}"
              f"\n headPosdThreshold:{headPosdThreshold}"
              )
        for key, value in self.bodyTrackidInOneSecond.items():
            if value > bodyTrackThreshold:
                print(f"第{time}s id:{key} 稳定识别")
                bodyTrackCount += 1
                # 认为这1s稳定识别,那么进行姿态判断
                siteCount = self.siteInOneSecond.get(key, -1)
                standCount = self.standInOneSecond.get(key, -1)
                # 此秒内坐姿以及站姿均大于阈值数量，那么取这两个值中较大值进行判定
                if siteCount >= siteStandThreshold and standCount >= siteStandThreshold:
                    if siteCount > standCount:
                        siteSumCount += 1
                        print("坐姿")
                    else:
                        standSumCount += 1
                        print("站姿")
                elif siteCount >= siteStandThreshold:
                    siteSumCount += 1
                    print("坐姿")
                elif standCount >= siteStandThreshold:
                    standSumCount += 1
                    print("站姿")
                else:
                    print("坐姿站姿无法判断")

                # 此秒内头部姿态判定
                headUpCount = self.headUpInOneSecond.get(key, -1)
                headFrontCount = self.headFrontInOneSecond.get(key, -1)
                headDownCount = self.headDownInOneSecond.get(key, -1)
                if headUpCount >= headPosdThreshold and headFrontCount >= headPosdThreshold and headDownCount >= headPosdThreshold:
                    # 比较这三个值中的最大值，并输出对应姿态
                    max_count = max(headUpCount, headFrontCount, headDownCount)
                    if max_count == headUpCount:
                        headUpSumCount += 1
                        print("抬头")
                    elif max_count == headFrontCount:
                        headFrontSumCount += 1
                        print("正视")
                    else:
                        headDownSumCount += 1
                        print("低头")
                # 判断是否有两个值大于threshold
                elif headUpCount >= headPosdThreshold and headFrontCount >= headPosdThreshold:
                    # 比较这两个值中的最大值，并输出对应姿态
                    max_count = max(headUpCount, headFrontCount)
                    if max_count == headUpCount:
                        headUpSumCount += 1
                        print("抬头")
                    else:
                        headFrontSumCount += 1
                        print("正视")
                elif headUpCount >= headPosdThreshold and headDownCount >= headPosdThreshold:
                    # 比较这两个值中的最大值，并输出对应姿态
                    max_count = max(headUpCount, headDownCount)
                    if max_count == headUpCount:
                        headUpSumCount += 1
                        print("抬头")
                    else:
                        headDownSumCount += 1
                        print("低头")
                elif headFrontCount >= headPosdThreshold and headDownCount >= headPosdThreshold:
                    # 比较这两个值中的最大值，并输出对应姿态
                    max_count = max(headFrontCount, headDownCount)
                    if max_count == headFrontCount:
                        headFrontSumCount += 1
                        print("正视")
                    else:
                        headDownSumCount += 1
                        print("低头")
                # 判断是否只有一个值大于threshold
                elif headUpCount >= headPosdThreshold:
                    headUpSumCount += 1
                    print("抬头")
                elif headFrontCount >= headPosdThreshold:
                    headFrontSumCount += 1
                    print("正视")
                elif headDownCount >= headPosdThreshold:
                    headDownSumCount += 1
                    print("低头")
                else:
                    print("头部姿态无法判")

        self.bodyTrackidInOneSecond = {}
        self.siteInOneSecond = {}
        self.standInOneSecond = {}
        self.headUpInOneSecond = {}
        self.headFrontInOneSecond = {}
        self.headDownInOneSecond = {}
        print(f"第{time}s"
              f"\n 识别人数:{bodyTrackCount}"
              f"\n 坐姿人数:{siteSumCount}"
              f"\n 站姿人数:{standSumCount}"
              f"\n 抬头人数:{headUpSumCount}"
              f"\n 平视人数:{headFrontSumCount}"
              f"\n 低头人数:{headDownSumCount}"
              )
        try:
            one_second_callback(time, bodyTrackCount, siteSumCount, standSumCount, headUpSumCount, headFrontSumCount,
                                headDownSumCount)
        except Exception as e:
            # 当发生异常时，异常信息会被捕获到这里，并打印出来
            print("发生了异常：", e)

        print(f"<<<<<<<<<<<<<<<<<<<<< 第{time}s数据分析完毕")
        pass
