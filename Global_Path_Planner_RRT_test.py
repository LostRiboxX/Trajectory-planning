from CV_Module import Global_Planner_CV_Module
from Trans_Module import Global_Planner_Trans_Module
# from NeurophysicalModel_Module import Global_Planner_NeurophysicalModel_Module
import math
import numpy as np
import cv2
import pickle
import random


class Node:
    def __init__(self, position, movement=None, parent=None):
        self.position = position
        self.movement = movement
        self.parent = parent


class Global_Path_Planner_RRT_Module():
    def __init__(self, imgname):
        self.CV_Module = Global_Planner_CV_Module(imgname)
        self.Trans_Module = Global_Planner_Trans_Module()
        # self.NeurophysicalModel_Module = Global_Planner_NeurophysicalModel_Module()
        self.nodes = []
        self.routes = []
        self.routes_data = []
        self.routes_metrics = []
        self.countours = None
        self.field_size = (2.1, 2.1)
        self.best_routes = []
        self.best_routes_data = []
        self.best_routes_metrics = []
        self.contmass = self.CV_Module.contmass_ret(13, 1, 9)

    def is_close(self, pos1, pos2, epsilon=0.03):
        # print(abs(pos1[0] - pos2[0]), abs(pos1[1] - pos2[1]))
        return abs(pos1[0] - pos2[0]) < epsilon and abs(pos1[1] - pos2[1]) < epsilon

    def euclidean_distance(self, pos1, pos2):
        return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

    def choose_random_movement(self, current_surface, current_position, goal_position):
        if tuple(current_surface) in self.combs_dict:
            possible_moves = self.combs_dict[tuple(current_surface)]
            # Фильтруем перемещения, выбирая только те, которые уменьшают расстояние до цели
            filtered_moves = [move for move in possible_moves if self.euclidean_distance(
                (current_position[0] + move[0], current_position[1] + move[1]),
                goal_position) < self.euclidean_distance(current_position, goal_position)]
            if filtered_moves:
                return random.choice(filtered_moves)
        return None

    def add_node(self, parent_node, movement):
        dx, dy, _ = movement[:3]  # Extract displacement and angle change
        new_position = (parent_node.position[0] + dx, parent_node.position[1] + dy)
        # Проверяем, что новая позиция находится в пределах поля
        if 0 <= new_position[0] <= self.field_size[0] and 0 <= new_position[1] <= self.field_size[1]:
            new_node = Node(new_position, movement, parent_node)
            return new_node
        return None

    def backtrack_route(self, node):
        route = []
        route_data = []
        while node.parent is not None:
            route.append(node.position)
            route_data.append(node.movement)
            node = node.parent
        route.append(node.position)
        route.reverse()
        return [route, route_data]

    def run_rrt(self, start, goal, combs_dict, max_routes=5):
        i = 0
        self.combs_dict = combs_dict

        root = Node(start, None)
        self.nodes = [root]
        while len(self.routes) < max_routes:
            rand_movement = self.choose_random_movement(self.get_current_surface_type(self.nodes[-1].position),
                                                        self.nodes[-1].position, goal)
            if rand_movement:
                new_node = self.add_node(self.nodes[-1], rand_movement)
                if new_node:  # Убедимся, что новый узел валиден
                    self.nodes.append(new_node)
                    if self.is_close(new_node.position, goal):
                        data = self.backtrack_route(new_node)
                        self.routes.append(data[0])
                        self.routes_data.append(data[1])
                        print(i)
                        self.nodes = [root]  # Restart to find a new route
                        i += 1
                        # print("-----Found smthng-----")
                        # print(self.backtrack_route(new_node))
                        # print("----------END---------")

    def get_current_surface_type(self, position):
        point_p = self.Trans_Module.coords_from_m_2_p_mass(position)
        return self.CV_Module.find_wheels_pos(point_p, self.contmass, 0)

    def save_routes(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.routes, f)

    # def draw_routes(self, img):
    #     for route in self.routes:
    #         if route:
    #             # print(route)
    #             # Перебираем все точки маршрута, чтобы нарисовать линии между ними
    #             for i in range(len(route) - 1):
    #                 start_point = self.Trans_Module.coords_from_m_2_p_mass([route[i][0], route[i][1]])
    #                 end_point = self.Trans_Module.coords_from_m_2_p_mass([route[i + 1][0], route[i + 1][1]])
    #                 cv2.line(img, start_point, end_point, (0, 255, 0), 2)  # Зеленая линия толщиной 2
    #
    #             # Рисуем круги в начальной и конечной точках маршрута
    #             cv2.circle(img, self.Trans_Module.coords_from_m_2_p_mass([route[0][0], route[0][1]]), 5, (0, 0, 255),
    #                        -1)  # Красный
    #             cv2.circle(img, self.Trans_Module.coords_from_m_2_p_mass([route[-1][0], route[-1][1]]), 5, (255, 0, 0),
    #                        -1)  # Синий
    #
    #     cv2.imshow("Routes", img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()

    def calculate_metrics(self, end_point):
        for route_data, route_poses in zip(self.routes_data, self.routes):
            global_watt = 0
            global_efficient = 0
            route_length = 0
            for movement_data, pos, q in zip(route_data, route_poses, range(len(route_poses))):
                # print(movement_data)
                # print(pos)
                global_watt += sum([abs(movement_data[-1][0][i] * movement_data[3][0][i]) for i in range(3)])
                if q != 0:
                    dist = self.euclidean_distance(route_poses[q], end_point)
                    dist_prev = self.euclidean_distance(route_poses[q - 1], end_point)
                    global_efficient += (sum([abs(movement_data[-1][0][i] * movement_data[3][0][i]) for i in range(3)])) / (
                            (dist_prev - dist) * 1000)
                    route_length += self.euclidean_distance(route_poses[q], route_poses[q-1])
            self.routes_metrics.append({'watt': global_watt, 'length': route_length, 'efficient' : 1 / global_efficient * 10000})

    def sort_by_power(self):
        indecies = np.argsort([item['watt'] for item in self.routes_metrics])[:3]
        for index in indecies:
            self.best_routes.append(self.routes[index])
            self.best_routes_data.append(self.routes_data[index])
            self.best_routes_metrics.append(self.routes_metrics[index])

    def draw_routes_best(self, img):
        for route in self.routes:
            if route:
                # print(route)
                # Перебираем все точки маршрута, чтобы нарисовать линии между ними
                for i in range(len(route) - 1):
                    start_point = self.Trans_Module.coords_from_m_2_p_mass([route[i][0], route[i][1]])
                    end_point = self.Trans_Module.coords_from_m_2_p_mass([route[i + 1][0], route[i + 1][1]])
                    cv2.line(img, start_point, end_point, (0, 255, 0), 2)  # Зеленая линия толщиной 2

                # Рисуем круги в начальной и конечной точках маршрута
                cv2.circle(img, self.Trans_Module.coords_from_m_2_p_mass([route[0][0], route[0][1]]), 5, (0, 0, 255),
                           -1)  # Красный
                cv2.circle(img, self.Trans_Module.coords_from_m_2_p_mass([route[-1][0], route[-1][1]]), 5, (255, 0, 0),
                           -1)  # Синий
        for route in self.best_routes:
            if route:
                # print(route)
                # Перебираем все точки маршрута, чтобы нарисовать линии между ними
                for i in range(len(route) - 1):
                    start_point = self.Trans_Module.coords_from_m_2_p_mass([route[i][0], route[i][1]])
                    end_point = self.Trans_Module.coords_from_m_2_p_mass([route[i + 1][0], route[i + 1][1]])
                    cv2.line(img, start_point, end_point, (255, 0, 0), 2)  # Зеленая линия толщиной 2

                # Рисуем круги в начальной и конечной точках маршрута
                cv2.circle(img, self.Trans_Module.coords_from_m_2_p_mass([route[0][0], route[0][1]]), 5, (255, 0, 255),
                           -1)  # Красный
                cv2.circle(img, self.Trans_Module.coords_from_m_2_p_mass([route[-1][0], route[-1][1]]), 5, (255, 0, 255),
                           -1)  # Синий
        cv2.imshow("Routes", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

Path_Planner = Global_Path_Planner_RRT_Module("../1.jpg")
filename = "combs_dict.pkl"
with open(filename, 'rb') as file:
    combs_dict = pickle.load(file)
Path_Planner.run_rrt((0.9346341463414635, 0.7456310679611651), (1.3639024390243903, 1.836893203883495), combs_dict,
                     max_routes=5000)
Path_Planner.save_routes("150000_RRT_routes.pkl")
# img = cv2.imread("../1.jpg_croped.png_done.png")
img = cv2.imread("../1.jpg_croped.png_done.png")
Path_Planner.calculate_metrics((1.3639024390243903, 1.836893203883495))
Path_Planner.sort_by_power()
Path_Planner.draw_routes_best(img)
print(Path_Planner.best_routes_metrics)
