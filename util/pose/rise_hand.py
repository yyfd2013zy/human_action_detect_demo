import logging
import math

from util.calculate_util import CaculateUtil


class RiseHandCheck:

    def __init__(self):
        pass

    # 检测是否举手  8:左边肩膀-左右侧胳膊肘  10:左侧胳膊肘-左侧手腕
    # 检测是否举手  9:右边肩膀-右侧胳膊肘  11:右侧胳膊肘-右侧手腕
    def checkRiseHand(self, line8, line10, line9, line11):
        print("line 11 data", line11)

        # step1 先判断右手手腕是否高于手肘
        if int(line11[3]) < int(line11[1]):
            print("右边-手腕抬起")
            # step2 判断 line11 右侧胳膊肘-右侧手腕 这条线是否与y轴平行
            angle = self.calculate_angle(int(line11[0]), int(line11[1]), int(line11[2]), int(line11[3]))
            print("右侧胳膊肘-右侧手腕 线段与y轴的夹角为:", angle, "度")
            if angle > 75 or angle< -75:
                return "rise hand"

        else:
            print("右边-手腕放下")
            return "none"



        # step1 判断 右侧手腕 是否在 右侧胳膊肘 之上
        # print("line 10 data", line10)
        # if int(line10[3]) > int(line10[1]):
        #     print("手腕抬起")
        # else:
        #     print("手腕放下")

        # 右边胳膊
        rightArmAngle = CaculateUtil.calculate_angle((int(line8[0]), int(line8[1]), int(line8[2]), int(line8[3])),
                                                     (int(line10[0]), int(line10[1]), int(line10[2]), int(line10[3])))
        arm = 180 - rightArmAngle
        print("arm angle", arm)
        '''
        if headAngle < 110:
            return "低头"
        elif headAngle < 140:
            return "平视"
        elif headAngle < 160:
            return "抬头"
        else:
            return("仰头")
        '''
        return "none"
        pass

    def calculate_angle(slef, x1, y1, x2, y2):
        # 计算斜率
        if x2 - x1 != 0:
            slope = (y2 - y1) / (x2 - x1)
        else:
            slope = float('inf')  # 斜率为正无穷

        # 计算夹角（以弧度为单位）
        angle_radians = math.atan(slope)

        # 将弧度转换为角度
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees
