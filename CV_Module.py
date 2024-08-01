import cv2
import numpy as np
from PIL import Image, ImageOps
import math
# from shapely.geometry import LineString, Point
from skimage.draw import line


class Global_Planner_CV_Module:
    def __init__(self, imgname):
        self.seqeunce_of_colors = ['gray', 'blue', 'brown', 'lightgreen', 'lightgreen', 'red', 'gray', 'brown', 'red',
                                   'darkgray',
                                   'gray',
                                   'lightgreen', 'brown', 'red', 'green', 'darkgray', 'gray', 'blue', 'gray', 'pink',
                                   'lightgreen',
                                   'green',
                                   'lightbrown']
        self.dict_w_color_types = {'gray': 2, 'blue': 2, 'brown': 3, 'lightgreen': 2, 'red': 3, 'darkgray': 2,
                                   'green': 3, 'pink': 2,
                                   'lightbrown': 2}

        self.dict_w_color_score = {'red': 10, 'blue': 4, 'gray': 1, 'darkgray': 5, 'lightgreen': 2, 'green': 3,
                                   'pink': 2,
                                   'lightbrown': 4,
                                   'brown': 5}
        self.prime_img_name = imgname

    def __crop(self, imgname):
        img = cv2.imread(imgname)
        cropped_image = img[0:1025, 370:1400]
        return cropped_image

    def __filter(self, img):
        blur = cv2.medianBlur(img, 31)

        image = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)

        pixel_vals = image.reshape((-1, 3))

        pixel_vals = np.float32(pixel_vals)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100000, 0)
        k = 18
        retval, labels, centers = cv2.kmeans(pixel_vals, k, None, criteria, 18, cv2.KMEANS_RANDOM_CENTERS)
        centers = np.uint8(centers)
        segmented_data = centers[labels.flatten()]
        segmented_image = segmented_data.reshape((image.shape))
        img = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
        return img

    def __get_rawconts(self, imgfiltered):
        img = imgfiltered
        for i in range(img.shape[0]):
            for j in range(img.shape[1] - 1):
                if ((img[i, j][0] + img[i, j][1] + img[i, j][2]) - (
                        img[i, j + 1][0] + img[i, j + 1][1] + img[i, j + 1][2])) >= 10:
                    img[i, j] = (0, 0, 0)
                else:
                    img[i, j] = (255, 255, 255)

        eroded = cv2.erode(img, None, iterations=15)
        cv2.imwrite('differosed.jpg', eroded)
        ImageOps.expand(Image.open('differosed.jpg'), border=3, fill='black').save('differosed.jpg')
        img = cv2.imread('differosed.jpg')

        _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)
        return thresh

    def __contersmass(self, rawconts_img, croped_img):
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (10, 500)
        fontScale = 2
        fontColor = (255, 255, 255)
        thickness = 2
        lineType = 2

        img = rawconts_img
        image = croped_img

        median_blur = cv2.medianBlur(img, 15)
        gray = cv2.cvtColor(median_blur, cv2.COLOR_BGR2GRAY)

        ret, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
        j = 0
        contours, hierarchies = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        massbrown = []
        contsmass = []
        for i in contours:
            area = cv2.contourArea(i)
            if 4700 < area < 100000:
                M = cv2.moments(i)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    cv2.drawContours(image, [i[0] - 200, i[1] - 200], -1, (0, 255, 0), 2)
                    cv2.circle(image, (cx + 5, cy - 5), 10, (0, 0, 255), -1)
                    cv2.putText(image, str(j), (cx - 10, cy - 10), font, fontScale, fontColor, thickness, lineType)
                    massx = ["cx brown", cx, "cy brown: ", cy]
                    mass1 = [cx - 5, cy - 5]
                    massbrown.append(mass1)
                    contsmass.append(i)
                    j += 1
        return image, massbrown, contsmass

    def __dijkstra(self, graph, start, end):
        heap = [(0, start)]
        visited = set()
        paths = {start: [start]}  # Словарь для хранения путей

        while heap:
            (cost, current) = heapq.heappop(heap)
            if current in visited:
                continue
            visited.add(current)
            if current == end:
                return cost, paths[current]

            for neighbor, edge_cost in graph[current].items():
                if neighbor not in visited:
                    heapq.heappush(heap, (cost + edge_cost, neighbor))
                    if neighbor not in paths or len(paths[current]) + 1 < len(paths[neighbor]):
                        paths[neighbor] = paths[current] + [neighbor]
        return -1, []

    def __find_top_paths_weighted(self, graph, start, end, top_k=10):
        paths = []

        def dfs(current_path, current_weight):
            nonlocal paths

            current_node = current_path[-1]

            if current_node == end:
                paths.append((current_path.copy(), current_weight))
                return

            for neighbor, weight in graph[current_node].items():
                if neighbor not in current_path:
                    dfs(current_path + [neighbor], current_weight + weight)

        dfs([start], 0)
        paths.sort(key=lambda x: x[1])  # Сортируем по общему весу
        return paths[:top_k]

    def __calculate_line_equation(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        k = (y2 - y1) / (x2 - x1)

        b = y1 - k * x1

        return k, b

    def __calculate_distance(self, point1, point2):
        x1, y1 = point1
        x2, y2 = point2

        diff_x = x2 - x1
        diff_y = y2 - y1

        distance = math.sqrt(diff_x ** 2 + diff_y ** 2)

        return distance

    def __draw_path(self, img, centermass: list, path: list):
        p1_p2_k_b_len_mass = []
        for i in range(len(path)):
            try:
                cv2.line(img, centermass[path[i]], centermass[path[i + 1]], (255, 0, 0), 3)
                k, b = self.__calculate_line_equation(centermass[path[i]], centermass[path[i + 1]])
                length = self.__calculate_distance(centermass[path[i]], centermass[path[i + 1]])
                p1_p2_k_b_len_mass.append(
                    [centermass[path[i]], centermass[path[i + 1]], k, b, length, math.atan(k) * 180 / math.pi])
            except:
                break

        return img, p1_p2_k_b_len_mass

    def _find_connection(self, contours, number, q):
        cont = contours[number]
        connections = []
        for i in range(len(cont)):
            for j in range(len(contours)):
                for b in range(q // 10):
                    if ((number != j) and (
                            cv2.pointPolygonTest(contours[j], (int(cont[i][0][0] - (b * 10)), int(cont[i][0][1])),
                                                 False) == 1)):
                        connections.append(j)
                    if ((number != j) and (
                            cv2.pointPolygonTest(contours[j], (int(cont[i][0][0]), int(cont[i][0][1] - (b * 10))),
                                                 False) == 1)):
                        connections.append(j)
                    if ((number != j) and (
                            cv2.pointPolygonTest(contours[j], (int(cont[i][0][0] + (b * 10)), int(cont[i][0][1])),
                                                 False) == 1)):
                        connections.append(j)
                    if ((number != j) and (
                            cv2.pointPolygonTest(contours[j], (int(cont[i][0][0]), int(cont[i][0][1] + (b * 10))),
                                                 False) == 1)):
                        connections.append(j)

        seen = set()
        seen_add = seen.add
        return [x for x in connections if not (x in seen or seen_add(x))]

    def __find_connection(self, croped1, centersmass, contours):
        connections = []
        for i in range(len(contours)):
            connections.append(self._find_connection(contours, i, 100))

        for i in range(len(connections)):
            for j in (connections[i]):
                cv2.line(croped1, (centersmass[i][0], centersmass[i][1]), (centersmass[j][0], centersmass[j][1]),
                         (255, 255, 255), 2)
        return connections

    # def __get_raw_lines_data(self, start, end, number):
    #     croped = self.__crop(self.prime_img_name)
    #     filtered = self.__filter(croped)
    #     raw_conts_img = self.__get_rawconts(filtered)
    #     detimage, massbrown, contsmass = self.__contersmass(raw_conts_img, croped)
    #     image = detimage.copy()
    #     connections = self.__find_connection(croped, massbrown, contsmass)
    #     dict_graph = {}
    #     for i in range(len(connections)):
    #         tmp = {}
    #         for j in range(len(connections[i])):
    #             tmp[connections[i][j]] = self.dict_w_color_score.copy()[
    #                 self.seqeunce_of_colors.copy()[connections[i][j]]]
    #         dict_graph[i] = tmp
    #     paths = self.__find_top_paths_weighted(dict_graph, start, end)
    #     img_w_path_name, lines_data = self.__draw_path(image, massbrown, paths[number][0])
    #     return lines_data, contsmass, paths[number][0], paths[number][1], img_w_path_name, massbrown

    def __new_dijkstra_weight(self, n1, s1, n2, s2, massbrown, contmass):

        p1 = massbrown[n1]
        p2 = massbrown[n2]
        cont = contmass[n1]

        for pt in zip(*line(*p1, *p2)):
            point = tuple([int(round(pt[0])), int(round(pt[1]))])
            if cv2.pointPolygonTest(cont, point, False) == 0:
                intersection_point = point
                break

        try:
            length1 = math.sqrt((p1[0] - intersection_point[0]) ** 2 + (p1[1] - intersection_point[1]) ** 2)
            length2 = math.sqrt((p2[0] - intersection_point[0]) ** 2 + (p2[1] - intersection_point[1]) ** 2)

        except UnboundLocalError:
            length1 = math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) / 2
            length2 = length1

        return length1 * s1 + length2 * s2

    def __new_dijkstra_w_weight(self, i, j, massbrown):
        return math.sqrt((massbrown[i][0] - massbrown[j][0]) ** 2 + (massbrown[i][1] - massbrown[j][1]) ** 2)

    def __get_raw_lines_data(self, start, end, number):
        croped = self.__crop(self.prime_img_name)
        filtered = self.__filter(croped)
        raw_conts_img = self.__get_rawconts(filtered)
        detimage, massbrown, contsmass = self.__contersmass(raw_conts_img, croped)
        image = detimage.copy()
        connections = self.__find_connection(croped, massbrown, contsmass)
        dict_graph = {}
        for i in range(len(connections)):
            tmp = {}
            for j in range(len(connections[i])):
                # print(i, self.dict_w_color_score.copy()[
                #     self.seqeunce_of_colors.copy()[i]], connections[i][j], self.dict_w_color_score.copy()[
                #           self.seqeunce_of_colors.copy()[
                #               connections[i][j]]])
                #     tmp[connections[i][j]] = self.dict_w_color_score.copy()[
                #         self.seqeunce_of_colors.copy()[connections[i][j]]]
                tmp[connections[i][j]] = self.__new_dijkstra_w_weight(i, connections[i][j], massbrown)
                # tmp[connections[i][j]] = self.__new_dijkstra_weight(i, self.dict_w_color_score.copy()[
                #     self.seqeunce_of_colors.copy()[i]], connections[i][j], self.dict_w_color_score.copy()[
                #                                                         self.seqeunce_of_colors.copy()[
                #                                                             connections[i][j]]], massbrown, contsmass)
            dict_graph[i] = tmp
        paths = self.__find_top_paths_weighted(dict_graph, start, end)
        # print("DICT GRAPHSSSSSS", dict_graph)
        img_w_path_name, lines_data = self.__draw_path(image, massbrown, paths[number][0])
        cv2.imwrite(f"{number}_w_weight.png", img_w_path_name)
        return lines_data, contsmass, paths[number][0], paths[number][1], img_w_path_name, massbrown

    # def intersection_method_test(self):
    #
    #     img = cv2.imread("E:/Py/DiplomaWork/1.jpg_croped.pngauf.png")
    #     croped = self.__crop(self.prime_img_name)
    #     filtered = self.__filter(croped)
    #     raw_conts_img = self.__get_rawconts(filtered)
    #     detimage, massbrown, contsmass = self.__contersmass(raw_conts_img, croped)
    #
    #     # 9 and 4 test
    #
    #     p1 = massbrown[8]
    #     p2 = massbrown[3]
    #     cont = contsmass[3]
    #
    #
    #     cv2.circle(img, p1, 2, (0, 0, 255), 4)
    #     cv2.circle(img, p2, 2, (0, 0, 255), 4)
    #
    #     for pt in zip(*line(*p1, *p2)):
    #         point = tuple([int(round(pt[0])), int(round(pt[1]))])
    #         if cv2.pointPolygonTest(cont, point, False) == 0:
    #             cv2.circle(img, point, 2, (255, 255, 0), 4)
    #
    #     cv2.imshow('0', img)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()
    #     return

    # def test_dijkstra(self, start, end, number):
    #     croped = self.__crop(self.prime_img_name)
    #     filtered = self.__filter(croped)
    #     raw_conts_img = self.__get_rawconts(filtered)
    #     detimage, massbrown, contsmass = self.__contersmass(raw_conts_img, croped)
    #     image = detimage.copy()
    #     connections = self.__find_connection(croped, massbrown, contsmass)
    #     dict_graph = {}
    #     for i in range(len(connections)):
    #         tmp = {}
    #         for j in range(len(connections[i])):
    #             print(i, self.dict_w_color_score.copy()[
    #                 self.seqeunce_of_colors.copy()[i]], connections[i][j], self.dict_w_color_score.copy()[
    #                       self.seqeunce_of_colors.copy()[
    #                           connections[i][j]]])
    #             tmp[connections[i][j]] = self.__new_dijkstra_weight(i, self.dict_w_color_score.copy()[
    #                 self.seqeunce_of_colors.copy()[i]], connections[i][j], self.dict_w_color_score.copy()[
    #                                                                     self.seqeunce_of_colors.copy()[
    #                                                                         connections[i][j]]], massbrown, contsmass)
    #         dict_graph[i] = tmp
    #     print(dict_graph)
    #     paths = self.__find_top_paths_weighted(dict_graph, start, end)
    #     img_w_path_name, lines_data = self.__draw_path(image, massbrown, paths[number][0])
    #     return lines_data, contsmass, paths[number][0], paths[number][1], img_w_path_name, massbrown

    def __wheels_x_y(self, point, angle0):
        angle0 = - angle0
        angle0 = angle0 * 180 / math.pi
        angle = angle0 + 210

        center_x, center_y = point[0], point[1]

        radius1 = 90

        angle1 = angle * math.pi / 180
        angle2 = (2 * math.pi / 3) + (angle * math.pi / 180)
        angle3 = (4 * math.pi / 3) + (angle * math.pi / 180)

        center1_x = int(center_x + radius1 * math.cos(angle1))
        center1_y = int(center_y + radius1 * math.sin(angle1))
        center2_x = int(center_x + radius1 * math.cos(angle2))
        center2_y = int(center_y + radius1 * math.sin(angle2))
        center3_x = int(center_x + radius1 * math.cos(angle3))
        center3_y = int(center_y + radius1 * math.sin(angle3))
        return ((center1_x, center1_y), (center2_x, center2_y), (center3_x, center3_y))

    def __determine_surface_types(self, wheels, contours, color_sequence, color_type_mapping):
        result = []

        for wheel_set in wheels:
            surface_types = []
            for wheel_coords in wheel_set:
                x, y = wheel_coords
                contour_index = None

                for i, contour in enumerate(contours):
                    if any((x, y) in point for point in contour):
                        contour_index = i
                        break

                if contour_index is not None:
                    contour_color = color_sequence[contour_index]

                    surface_type = color_type_mapping.get(contour_color, 'unknown')
                    surface_types.append(surface_type)
                else:
                    print(f"Контур с координатами {wheel_coords} не найден.")
                    surface_types.append(2)

            result.append(tuple(surface_types))

        return tuple(result)

    def find_wheels_pos(self, point, contours, angle):
        wheels_coords = self.__wheels_x_y(point, angle)
        surface_types = []
        for wheel_coords in wheels_coords:
            x, y = wheel_coords

            contour_index = None

            for i, contour in enumerate(contours):
                if any((x, y) in point for point in contour):
                    contour_index = i
                    break

            if contour_index is not None:
                contour_color = self.seqeunce_of_colors[contour_index]

                surface_type = self.dict_w_color_types.get(contour_color, 'unknown')
                surface_types.append(surface_type)
            else:
                # print(f"Контур с координатами {wheel_coords} не найден.")
                surface_types.append(2)

        return surface_types

    def find_wheels_pos_edited(self, point, contours):
        wheels_coords = self.__wheels_x_y(point, 0)
        surface_types = []
        for wheel_coords in wheels_coords:
            x, y = wheel_coords
            contour_index = None

            for i, contour in enumerate(contours):
                if any((x, y) in point for point in contour):
                    contour_index = i
                    break
            tmp = 0
            if contour_index is not None:
                contour_color = self.seqeunce_of_colors[contour_index]

                surface_type = self.dict_w_color_types.get(contour_color, 'unknown')
                tmp = surface_type
            else:
                # print(f"Контур с координатами {wheel_coords} не найден.")
                tmp = 2

            if (contour_index == 8 or contour_index == 6 or contour_index == 14):
                tmp = 4

            surface_types.append(tmp)

        return surface_types

    def contmass_ret(self, start, end, number):
        lines_data, contsmass, path, path_value, img, contscenters = self.__get_raw_lines_data(start, end, number)
        return contsmass

    def points_and_vels(self, v_v_l, disc_time, start, end, number):
        lines_data, contsmass, path, path_value, img, contscenters = self.__get_raw_lines_data(start, end, number)
        result_points = []
        result_vels = []
        for line in lines_data:
            start_point, end_point, _, _, length, angle = line

            v_x = v_v_l * math.cos(math.radians(angle))
            v_y = v_v_l * math.sin(math.radians(angle))

            time = length / v_v_l

            for t in range(int(time / disc_time) + 1):
                x = start_point[0] + v_x * t * disc_time
                y = start_point[1] + v_y * t * disc_time
                result_points.append([round(x), round(y)])
                if end_point[0] - start_point[0] > 0:
                    result_vels.append([v_x, v_y, 0])
                else:
                    result_vels.append([-v_x, -v_y, 0])
            if end_point[0] - start_point[0] > 0:
                result_vels.append([v_x, v_y, 0])
            else:
                result_vels.append([-v_x, -v_y, 0])
            result_points.append([end_point[0], end_point[1]])

        # wheels_coords = []
        # for point in result_points:
        #     wheels_coords.append(self.__wheels_x_y(point, 0))
        #
        # surface_types_per_wheels = self.__determine_surface_types(wheels_coords, contsmass, self.seqeunce_of_colors,
        #                                                           self.dict_w_color_types)

        return result_points, result_vels, _, img, path, contsmass, contscenters
#
# CV_Module = Global_Planner_CV_Module("E:/Py/DiplomaWork/1.jpg")
# CV_Module._get_raw_lines_data(13, 1, 1)
