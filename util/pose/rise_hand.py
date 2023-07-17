from util.calculate_util import CaculateUtil


class RiseHandCheck:

    def __init__(self):
        pass

        # 检测是否举手
    def checkRiseHand(self, line68, line810):
        #右边胳膊
        rightArmAngle = CaculateUtil.calculate_angle((int(line68[0]), int(line68[1]), int(line68[2]), int(line68[3])),
                                     (int(line810[0]), int(line810[1]), int(line810[2]), int(line810[3])))
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