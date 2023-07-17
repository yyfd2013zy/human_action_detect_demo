import math

import numpy as np

#角度计算类
class CaculateUtil:
    @staticmethod
    def calculate_angle(line1, line2):
        # 将点坐标转换为向量表示形式
        vector1 = np.array([line1[2] - line1[0], line1[3] - line1[1]])
        vector2 = np.array([line2[2] - line2[0], line2[3] - line2[1]])

        # 计算向量的模的乘积
        magnitude_product = np.linalg.norm(vector1) * np.linalg.norm(vector2)

        # 计算向量的点积
        dot_product = np.dot(vector1, vector2)

        # 计算夹角（弧度）
        angle_rad = np.arccos(dot_product / magnitude_product)

        # 将弧度转换为角度
        angle_deg = np.degrees(angle_rad)

        return angle_deg

    # 计算几何角
    @staticmethod
    def angle_between_points(a, b, c):
        # Calculate vectors AB and BC
        AB = [b[0] - a[0], b[1] - a[1]]
        BC = [c[0] - b[0], c[1] - b[1]]

        # Calculate dot product of AB and BC
        dot_product = AB[0] * BC[0] + AB[1] * BC[1]

        # Calculate magnitudes of AB and BC
        magnitude_AB = math.sqrt(AB[0] ** 2 + AB[1] ** 2)
        magnitude_BC = math.sqrt(BC[0] ** 2 + BC[1] ** 2)

        # Calculate angle in radians
        angle_radians = math.acos(dot_product / (magnitude_AB * magnitude_BC))

        # Convert angle to degrees
        angle_degrees = math.degrees(angle_radians)

        return angle_degrees