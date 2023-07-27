import logging
import math

from pose_detect.calculate_util import CaculateUtil


class RiseHandCheck:

    def __init__(self):
        pass

    # 检测是否举手  8:左边肩膀-左右侧胳膊肘  10:左侧胳膊肘-左侧手腕
    # 检测是否举手  9:右边肩膀-右侧胳膊肘  11:右侧胳膊肘-右侧手腕
    def checkRiseHand(self, line8, line10, line9, line11):
        #print("右手举手检测", line11)
        # step1 先判断右手手腕是否高于手肘
        if int(line11[3]) < int(line11[1]):
            #print("右边-手腕抬起")
            # step2 判断 line11 右侧胳膊肘-右侧手腕 这条线是否与y轴平行
            angle = CaculateUtil.calculate_angle_with_y(int(line11[0]), int(line11[1]), int(line11[2]), int(line11[3]))
            #print("右侧胳膊肘-右侧手腕 线段与y轴的夹角为:", angle, "度")
            if angle > 75 or angle < -75:
                return "rise hand"
        else:
            #print("左手举手检测", line11)
            if int(line10[3]) < int(line10[1]):
                #print("左边-手腕抬起")
                # step2 判断 line11 右侧胳膊肘-右侧手腕 这条线是否与y轴平行
                angle = CaculateUtil.calculate_angle_with_y(int(line10[0]), int(line10[1]), int(line10[2]),
                                                            int(line10[3]))
                #print("左侧胳膊肘-左侧手腕 线段与y轴的夹角为:", angle, "度")
                if angle > 75 or angle < -75:
                    return "rise hand"
            return "none"

        pass
