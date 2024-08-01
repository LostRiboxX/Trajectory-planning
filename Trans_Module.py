import math
import numpy as np

class Global_Planner_Trans_Module:
    def __init__(self):
        pass

    def vel_ps_2_ms(self, speed):
        return [speed[0] * 2 / 1025, speed[1] * 2 / 1030, 0]

    def calculate_omni_wheel_velocities(self, speed, wheel_radius=0.04):  # Чисто
        matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                           [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                           [1 / 0.13, 1 / 0.13, 1 / 0.13]])
        r = wheel_radius
        X = np.array(speed)
        X1 = np.dot(1 / r, X)
        mat = np.dot(1 / 3, matrix)
        mat_inv = np.linalg.inv(mat)
        Y1 = np.matmul(mat_inv, X1)
        return np.dot(16, Y1)

    def calculate_vxvywz_from_omniwheels_vels(self, speed, wheels_radius=0.04):  # Чисто
        r = wheels_radius

        matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                           [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                           [1 / 0.13, 1 / 0.13, 1 / 0.13]])
        m1 = np.dot(1 / 3, matrix)
        mat = np.matmul(m1, np.array(speed) / 16)
        vxvywz = np.dot(r, mat)
        return vxvywz

    def coords_from_m_2_p(self, x_m, y_m):
        return x_m * 1025 / 2, y_m * 1030 / 2

    def coords_from_p_2_m(self, x_m, y_m):
        return x_m * 2 / 1025, y_m * 2 / 1030

    def coords_from_p_2_m_mass(self, coords):
        x_p, y_p = coords[0], coords[1]
        return [x_p * 2 / 1025, y_p * 2 / 1030]

    def coords_from_m_2_p_mass(self, coords):
        x_m, y_m = coords[0], coords[1]
        return [round(x_m * 1025 / 2), round(y_m * 1030 / 2)]

