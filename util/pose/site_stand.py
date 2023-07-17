from util.calculate_util import CaculateUtil


class SiteStandCHeck:

    def __init__(self):
        pass

    #检测头部姿态
    def checkStandOrSitPose(self, line1214, line1416):
        # 计算两条线之间的夹角
        # angle_1 = math.degrees(math.atan((point_3_cx - point_4_cx) / (point_3_cy - point_4_cy)))
        angle = CaculateUtil.calculate_angle((int(line1214[0]), int(line1214[1]), int(line1214[2]), int(line1214[3])),
                                     (int(line1416[0]), int(line1416[1]), int(line1416[2]), int(line1416[3])))
        # angle1 = self.angle_between_points((352, 234),( 291, 219),( 202, 370))
        # 可行
        # angle1 = self.angle_between_points((x1, y1), (x2, y2), (x4, y4))
        # 这里应该计算一下鼻子与两个肩膀连线的高度差，添加一个距离参数进行判断
        legAngle = 180 - angle
        print("leg angle", legAngle)
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