import math
import numpy as np
from keras import models


class Global_Planner_NeurophysicalModel_Module():
    def __init__(self):

        self.mean = np.array([-0.04328374, -0.04481319, 0.08801374, 1.99927798])
        self.std = np.array([270.13250383, 270.11489776, 270.03726287, 0.81634887])

        self.mean_velocity_current_green = np.array([-3.70464561, 1.81659984, 2.1912304])
        self.std_velocity_current_green = np.array([204.12715797, 194.17465242, 200.22339528])

        self.mean_velocity_current_grey = np.array([-3.15935342, 1.9343096, 1.65945736])
        self.std_velocity_current_grey = np.array([196.32761694, 199.25209513, 191.52317769])

        self.mean_for_slippage_green = np.array([0.73784104, 0.78476592, 0.73912061])
        self.std_for_slippage_green = np.array([0.31336431, 0.30759276, 0.32479139])

        self.mean_for_slippage_grey = np.array([0.84230211, 0.76257781, 0.85836667])
        self.std_for_slippage_grey = np.array([0.39570568, 0.33581775, 0.40823333])

        "predicting real speeds"

        self.motor_velocity_first = models.load_model('velocity_first')
        self.motor_velocity_second = models.load_model('velocity_second')
        self.motor_velocity_third = models.load_model('velocity_third')

        "prediction of the currents power consumption"

        self.current_first_motor_green = models.load_model('current_first_green')
        self.current_second_motor_green = models.load_model('current_second_green')
        self.current_third_motor_green = models.load_model('current_third_green')

        self.current_first_motor_grey = models.load_model('current_first_grey')
        self.current_second_motor_grey = models.load_model('current_second_grey')
        self.current_third_motor_grey = models.load_model('current_third_grey')

        self.current_first_motor_table = models.load_model('current_first_table')
        self.current_second_motor_table = models.load_model('current_second_table')
        self.current_third_motor_table = models.load_model('current_third_table')

        "prediction of the slippage coefficient"

        self.wheel_first_slippage_green = models.load_model('slippage_wheel_1_green')
        self.wheel_second_slippage_green = models.load_model('slippage_wheel_2_green')
        self.wheel_third_slippage_green = models.load_model('slippage_wheel_3_green')

        self.wheel_first_slippage_grey = models.load_model('slippage_wheel_1')
        self.wheel_second_slippage_grey = models.load_model('slippage_wheel_2')
        self.wheel_third_slippage_grey = models.load_model('slippage_wheel_3')

    def predict(self, velocity_1, velocity_2, velocity_3, type_surface_1, type_surface_2, type_surface_3, time,
                current_angle):
        current_angle = current_angle
        time = time
        input_data1 = np.array([velocity_1 * 30 / math.pi, velocity_2 * 30 / math.pi, velocity_3 * 30 / math.pi,
                                type_surface_1])
        input_data1 = input_data1.reshape(1, -1)
        input_data2 = np.array([velocity_1 * 30 / math.pi, velocity_2 * 30 / math.pi, velocity_3 * 30 / math.pi,
                                type_surface_2])
        input_data2 = input_data2.reshape(1, -1)
        input_data3 = np.array([velocity_1 * 30 / math.pi, velocity_2 * 30 / math.pi, velocity_3 * 30 / math.pi,
                                type_surface_3])
        input_data3 = input_data3.reshape(1, -1)

        input_data1 -= self.mean
        input_data1 /= self.std
        input_data2 -= self.mean
        input_data2 /= self.std
        input_data3 -= self.mean
        input_data3 /= self.std

        speed_motor_1 = self.motor_velocity_first.predict(input_data1, verbose = 0)
        speed_motor_2 = self.motor_velocity_second.predict(input_data2, verbose = 0)
        speed_motor_3 = self.motor_velocity_third.predict(input_data3, verbose = 0)
        motors_speed = np.c_[speed_motor_1, speed_motor_2]
        motors_speed = np.c_[motors_speed, speed_motor_3]

        if type_surface_1 == 2:
            location_for_current1 = np.array(motors_speed[0])

            location_for_current1 -= self.mean_velocity_current_grey
            location_for_current1 /= self.std_velocity_current_grey

            location_for_current1 = location_for_current1.reshape(1, -1)

            current_first_motor = self.current_first_motor_grey.predict(location_for_current1, verbose = 0)

        elif type_surface_1 == 3:
            location_for_current1 = np.array(motors_speed[0])

            location_for_current1 -= self.mean_velocity_current_green
            location_for_current1 /= self.std_velocity_current_green

            location_for_current1 = location_for_current1.reshape(1, -1)

            current_first_motor = self.current_first_motor_green.predict(location_for_current1, verbose = 0)

        if type_surface_2 == 2:
            location_for_current2 = np.array(motors_speed[0])

            location_for_current2 -= self.mean_velocity_current_grey
            location_for_current2 /= self.std_velocity_current_grey

            location_for_current2 = location_for_current2.reshape(1, -1)

            current_second_motor = self.current_second_motor_grey.predict(location_for_current2, verbose = 0)

        elif type_surface_2 == 3:
            location_for_current2 = np.array(motors_speed[0])

            location_for_current2 -= self.mean_velocity_current_green
            location_for_current2 /= self.std_velocity_current_green

            location_for_current2 = location_for_current2.reshape(1, -1)

            current_second_motor = self.current_second_motor_green.predict(location_for_current2, verbose = 0)

        if type_surface_3 == 2:
            location_for_current3 = np.array(motors_speed[0])

            location_for_current3 -= self.mean_velocity_current_grey
            location_for_current3 /= self.std_velocity_current_grey

            location_for_current3 = location_for_current3.reshape(1, -1)

            current_third_motor = self.current_third_motor_grey.predict(location_for_current3, verbose = 0)

        elif type_surface_3 == 3:
            location_for_current3 = np.array(motors_speed[0])

            location_for_current3 -= self.mean_velocity_current_green
            location_for_current3 /= self.std_velocity_current_green

            location_for_current3 = location_for_current3.reshape(1, -1)

            current_third_motor = self.current_third_motor_green.predict(location_for_current3, verbose = 0)

        current_motors = np.c_[current_first_motor, current_second_motor]
        current_motors = np.c_[current_motors, current_third_motor]

        if type_surface_1 == 2:
            current_motors_1 = current_motors.copy()
            current_motors_1 -= self.mean_for_slippage_grey
            current_motors_1 /= self.std_for_slippage_grey
            slippage_first = self.wheel_first_slippage_grey.predict(current_motors_1, verbose = 0)

        elif type_surface_1 == 3:
            current_motors_1 = current_motors.copy()
            current_motors_1 -= self.mean_for_slippage_green
            current_motors_1 /= self.std_for_slippage_green
            slippage_first = self.wheel_first_slippage_green.predict(current_motors_1, verbose = 0)

        if type_surface_2 == 2:
            current_motors_2 = current_motors.copy()
            current_motors_2 -= self.mean_for_slippage_grey
            current_motors_2 /= self.std_for_slippage_grey
            slippage_second = self.wheel_second_slippage_grey.predict(current_motors_2, verbose = 0)

        elif type_surface_2 == 3:
            current_motors_2 = current_motors.copy()
            current_motors_2 -= self.mean_for_slippage_green
            current_motors_2 /= self.std_for_slippage_green
            slippage_second = self.wheel_second_slippage_green.predict(current_motors_2, verbose = 0)

        if type_surface_3 == 2:
            current_motors_3 = current_motors.copy()
            current_motors_3 -= self.mean_for_slippage_grey
            current_motors_3 /= self.std_for_slippage_grey
            slippage_third = self.wheel_third_slippage_grey.predict(current_motors_3, verbose = 0)

        elif type_surface_3 == 3:
            current_motors_3 = current_motors.copy()
            current_motors_3 -= self.mean_for_slippage_green
            current_motors_3 /= self.std_for_slippage_green
            slippage_third = self.wheel_third_slippage_green.predict(current_motors_3, verbose = 0)

        if speed_motor_1[0] < 0:
            slippage_first[0] = -slippage_first[0]

        if speed_motor_2[0] < 0:
            slippage_second[0] = -slippage_second[0]

        if speed_motor_3[0] < 0:
            slippage_third[0] = -slippage_third[0]

        slippage = np.c_[slippage_first, slippage_second]
        slippage = np.c_[slippage, slippage_third]
        real_velocity = motors_speed - slippage
        real_velocity = real_velocity * math.pi / 30

        "Radius of omni-wheel"

        r = 0.04

        "Direct kinematics"
        matrix = np.array([[-2 * math.sin(math.pi / 3), -2 * math.sin(math.pi), 2 * math.sin(math.pi / 3)],
                           [2 * math.cos(math.pi / 3), 2 * math.cos(math.pi), 2 * math.cos(math.pi / 3)],
                           [1 / 0.13, 1 / 0.13, 1 / 0.13]])
        m1 = np.dot(1 / 3, matrix)
        mat = np.matmul(m1, real_velocity[0] / 16)
        vxvywz = np.dot(r, mat)
        dxdyda_local = np.array([vxvywz[0] * time, vxvywz[1] * time, vxvywz[2] * time])

        angle = dxdyda_local[2]

        if -3 <= velocity_1 <= 3 and -3 <= velocity_2 <= 3 and -3 <= velocity_3 <= 3:
            angle = 0

        global_coord_delta = np.array([-vxvywz[0] * time * math.sin(angle + current_angle)
                                       - vxvywz[1] * time * math.cos(angle + current_angle),
                                       -vxvywz[0] * time * math.cos(angle + current_angle)
                                       + vxvywz[1] * time * math.sin(angle + current_angle)])

        return [global_coord_delta[0], global_coord_delta[1], angle, current_motors, slippage]
