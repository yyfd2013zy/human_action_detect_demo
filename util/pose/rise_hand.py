from util.calculate_util import CaculateUtil


class RiseHandCheck:

    def __init__(self):
        pass

        # 检测是否举手
    def checkRiseHand(self, line68, line810):
        # 计算两条线之间的夹角
        # angle_1 = math.degrees(math.atan((point_3_cx - point_4_cx) / (point_3_cy - point_4_cy)))
        angle = CaculateUtil.calculate_angle((int(line68[0]), int(line68[1]), int(line68[2]), int(line68[3])),
                                     (int(line810[0]), int(line810[1]), int(line810[2]), int(line810[3])))
        # angle1 = self.angle_between_points((352, 234),( 291, 219),( 202, 370))
        # 可行
        # angle1 = self.angle_between_points((x1, y1), (x2, y2), (x4, y4))
        # 这里应该计算一下鼻子与两个肩膀连线的高度差，添加一个距离参数进行判断
        arm = 180 - angle
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
        return "作何"
        pass