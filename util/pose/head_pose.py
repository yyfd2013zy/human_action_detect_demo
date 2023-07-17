from util.calculate_util import CaculateUtil


class HeadPoseCHeck:

    def __init__(self):
        pass

    #检测头部姿态
    def checkHeadPose(self, line24, line46):
        # 计算两条线之间的夹角
        # angle_1 = math.degrees(math.atan((point_3_cx - point_4_cx) / (point_3_cy - point_4_cy)))
        angle = CaculateUtil.calculate_angle((int(line24[0]), int(line24[1]), int(line24[2]), int(line24[3])),
                                     (int(line46[0]), int(line46[1]), int(line46[2]), int(line46[3])))
        # angle1 = self.angle_between_points((352, 234),( 291, 219),( 202, 370))
        # 可行
        # angle1 = self.angle_between_points((x1, y1), (x2, y2), (x4, y4))
        # 这里应该计算一下鼻子与两个肩膀连线的高度差，添加一个距离参数进行判断
        headAngle = 180 - angle
        print("head angle",headAngle)
        if headAngle > 160:
            return "head up"
        elif headAngle > 120:
            return "head front"
        elif headAngle > 80:
            return "head down"
        else:
            return("none")
        pass