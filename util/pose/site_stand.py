from util.calculate_util import CaculateUtil


class SiteStandCHeck:

    def __init__(self):
        pass

    #检测头部姿态
    def checkStandOrSitPose(self, line1214, line1416):
        # 计算两条线之间的夹角

        angle = CaculateUtil.calculate_angle((int(line1214[0]), int(line1214[1]), int(line1214[2]), int(line1214[3])),
                                     (int(line1416[0]), int(line1416[1]), int(line1416[2]), int(line1416[3])))
        # 这里应该计算一下鼻子与两个肩膀连线的高度差，添加一个距离参数进行判断
        legAngle = 180 - angle
        print("leg angle", legAngle)

        if legAngle > 150:
            return "stand"
        else:
            return("sit")

        return "sit"
        pass