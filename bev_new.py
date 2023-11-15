import json
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from math import sin, cos

def show_one_image(img1, i):
    plt.imshow(img1)
    plt.title("Frame" + str(i))
    plt.draw()
    while True:
        if plt.waitforbuttonpress(0):
            plt.close()
            break

class PositionEstimation:
    def check_aspect_ratio_for_high_bb(self):
        box = self.min_rect_points
        sort_order_x = box[:, 0].argsort()[::-1]
        sort_order_y = box[:, 1].argsort()[::-1]

        x_min = box[sort_order_x[-1]]
        x_max = box[sort_order_x[0]]

        y_min = box[sort_order_y[-1]]
        y_max = box[sort_order_y[0]]

        x = x_max[0] - x_min[0]
        y = y_max[1] - y_min[1]

        if x > y:
            return False
        if y >= x:
            return True
        else:
            raise Exception("Aspect Ratio Checker Failed! Please check Constraints and start Debugging here!")

    def find_bottom_top_edge(self):
        """
        Use the rotated bounding box to select two edges, the top and bottom edge of the vehicle
        In case of an squared bbox, this will be used as base plate
        """
        box = self.min_rect_points
        bottom_edge = np.zeros((2, 2))
        top_edge = np.zeros((2, 2))
        unique_x, _ = np.unique(box[:, 0], return_counts=True)
        unique_y, _ = np.unique(box[:, 1], return_counts=True)
        is_square = False

        if self.obj_class != "person" or self.obj_class != "bicycle":
            self.high_bb_flag = self.check_aspect_ratio_for_high_bb()



            #if len(box) == 4 and len(unique_x) > 2 and len(unique_y) > 2:
            # Sort points by y coordinate
            sort_order = box[:, 1].argsort()[::-1]
            # Select highest y coordinate
            index_highest_y_coordinate = sort_order[0]
            bottom_vertice_candidate_0 = box[(index_highest_y_coordinate - 1) % 4]
            bottom_vertice_candidate_1 = box[(index_highest_y_coordinate + 1) % 4]
            top_edge[0] = box[(index_highest_y_coordinate + 2) % 4]

            sort_by_y_desc = box[sort_order]
            #  This point is guranted part of the bottom edge of a object
            bottom_vertice_0 = sort_by_y_desc[0]
            #print(bottom_vertice_0)
            bottom_edge[0] = bottom_vertice_0
            candidate_length_0 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_0)
            candidate_length_1 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_1)
            if not self.high_bb_flag:
                if candidate_length_0 > candidate_length_1:
                    bottom_edge[1] = bottom_vertice_candidate_0
                    top_edge[1] = bottom_vertice_candidate_1
                else:
                    bottom_edge[1] = bottom_vertice_candidate_1
                    top_edge[1] = bottom_vertice_candidate_0
            else:
                if candidate_length_0 > candidate_length_1:
                    bottom_edge[1] = bottom_vertice_candidate_1
                    top_edge[1] = bottom_vertice_candidate_0
                else:
                    bottom_edge[1] = bottom_vertice_candidate_0
                    top_edge[1] = bottom_vertice_candidate_1
        else:
            sort_order = box[:, 1].argsort()[::-1]
            # Select highest y coordinate
            index_highest_y_coordinate = sort_order[0]
            sort_by_y_desc = box[sort_order]
            #  This point is guranted part of the bottom edge of a object
            bottom_vertice_0 = sort_by_y_desc[0]
            bottom_edge[0] = bottom_vertice_0
            bottom_edge[1] = sort_by_y_desc[0]

        """
        else:
            
            ### My Case ###
            is_square = True
            # Sort points by y coordinate
            sort_order = box[:, 1].argsort()[::-1]
            # Select highest y coordinate
            index_highest_y_coordinate = sort_order[0]
            bottom_vertice_candidate_0 = box[(index_highest_y_coordinate - 1) % 4]
            bottom_vertice_candidate_1 = box[(index_highest_y_coordinate + 1) % 4]
            top_edge[0] = box[(index_highest_y_coordinate + 2) % 4]

            sort_by_y_desc = box[sort_order]
            #  This point is guranted part of the bottom edge of a object
            bottom_vertice_0 = sort_by_y_desc[0]
            print(bottom_vertice_0)
            bottom_edge[0] = bottom_vertice_0
            candidate_length_0 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_0)
            candidate_length_1 = np.linalg.norm(bottom_vertice_0 - bottom_vertice_candidate_1)
            if candidate_length_0 > candidate_length_1:
                bottom_edge[1] = bottom_vertice_candidate_0
                top_edge[1] = bottom_vertice_candidate_1
            else:
                bottom_edge[1] = bottom_vertice_candidate_1
                top_edge[1] = bottom_vertice_candidate_0
            
            
            if len(unique_y) == 2:
                # Sort by y-coordinates
                sort_order = box[:, 1].argsort()[::-1]
                # select first
                index_highest_y_coordinate = sort_order[0]
                sort_by_y_desc = box[sort_order]
                #  This point is guaranteed part of the bottom edge of a object
                bottom_vertice_0 = sort_by_y_desc[0]
                vertice1 = sort_by_y_desc[1]
                vertice2 = sort_by_y_desc[2]
                vertice3 = sort_by_y_desc[3]
                if bottom_vertice_0[0] == vertice2[0]:
                    length_vector_1 = np.linalg.norm(vertice2 - bottom_vertice_0)
                else:
                    length_vector_1 = np.linalg.norm(vertice3 - bottom_vertice_0)

                length_y_vector = np.linalg.norm(vertice1 - bottom_vertice_0)

                if length_y_vector > length_vector_1:
                    if bottom_vertice_0[0] < vertice1[0]:
                        bottom_edge[0] = vertice1
                        bottom_edge[1] = bottom_vertice_0
                    else:
                        bottom_edge[0] = bottom_vertice_0
                        bottom_edge[1] = vertice1
                    top_edge[0] = vertice2
                    top_edge[1] = vertice3
                    is_square = False
            """
        return np.asarray(bottom_edge, dtype=np.float32), np.asarray(top_edge, dtype=np.float32), is_square
    
    def get_min_rect_points(self):
        point_array = np.array(self.instance_pts)
        min_rect = cv.minAreaRect(np.float32(point_array))
        min_rect_points = cv.boxPoints(min_rect)
        angle_bb = min_rect[2]
        min_rect_points = np.int32(min_rect_points)
        return min_rect_points

    def find_ground_contact_line(self):
        bottom_points, top_points, is_square = self.find_bottom_top_edge()
        return bottom_points

    def move_ground_contact_points_by_bb_coordinates(self):
        for point in self.ground_contact_points_image:
            #print(point)
            point[0] = point[0] + self.bb_coordinates[1] - self.bb_coordinates[3]
            point[1] = point[1] + self.bb_coordinates[0]

    def transform_ground_contact_points_from_image_to_world(self):
        warped_points_list = list()
        for point in self.ground_contact_points_image:
            point_new = [point[0], point[1], 1]
            warped_point = np.matmul(self.Homography_Matrix, point_new)
            scaling = 1 / warped_point[2]

            warped_point[1] = warped_point[1] * scaling
            warped_point[2] = warped_point[2] * scaling
            warped_point[0] = warped_point[0] * scaling
            warped_points_list.append(warped_point)
        np.asarray(warped_points_list, dtype=np.float32)
        return warped_points_list

    def find_and_rotate_rvec_of_bottom_straight_by_degree(self, rotation_angle):
        point1_straight = self.ground_contact_points_world[0]
        point2_straigth = self.ground_contact_points_world[1]
        rvec_straight = np.array([[point2_straigth[0] - point1_straight[0]],
                                            [point2_straigth[1] - point1_straight[1]]], dtype=np.float32)
        rvec_straight = np.array([[(point2_straigth[0] - point1_straight[0]) / np.linalg.norm(rvec_straight)],
                                            [(point2_straigth[1] - point1_straight[1]) / np.linalg.norm(rvec_straight)]], dtype=np.float32)
        
        #print("Rvec" + str(rvec_straight))
        
        if rvec_straight[0] < 0:
            rvec_straight[0] = rvec_straight[0] * -1
            rvec_straight[1] = rvec_straight[1] * -1
        
        #print("Rvec" + str(rvec_straight))
        rotation_angle = np.deg2rad(rotation_angle)
        rot_mat = np.array([[cos(rotation_angle), sin(rotation_angle)],
                           [-sin(rotation_angle), cos(rotation_angle)]], dtype=np.float32)
        
        rvec_straight_rotated = np.matmul(rot_mat, rvec_straight)
        self.rvec_base = rvec_straight
        return rvec_straight_rotated

    def calc_midpoint_from_two_points(self):
        pt1 = self.ground_contact_points_world[0]
        pt2 = self.ground_contact_points_world[1]
        new_x = (pt1[0] + pt2[0]) / 2
        new_y = (pt1[1] + pt2[1]) / 2
        point = [new_x, new_y]
        return point

    def shift_point_by_rvec_and_object_class(self):
        point = self.ground_contact_point_world
        rvec = self.rotated_rvec
        if self.obj_class == "person":
            length = 0
        elif self.obj_class == "bicycle":
            length = 0.3
        elif self.obj_class == "car" and self.high_bb_flag:
            length = 2.0
        elif self.obj_class == "car" and not self.high_bb_flag:
            length = 1.0
        elif self.obj_class == "van":
            length = 1.1
        elif self.obj_class == "truck":
            length = 1.3
        elif self.obj_class == "bus":
            length = 1.5
        else:
            raise Exception("Error. Object Class is not known by Module. Check Message from Object Detector!!!")
        length = self.scale_factor * length
        shifted_point_x = point[0] + rvec[0] * length
        shifted_point_y = point[1] + rvec[1] * length
        shifted_point = [shifted_point_x, shifted_point_y]
        return shifted_point
    
    def transform_point_from_world_to_image(self, point):
        point = [point[0], point[1], np.array([[1]], dtype=np.float32)]
        #print(point)
        warped_point = np.matmul(self.inv_Homography_Matrix, np.array([point[0].item(), point[1].item(), 1]))
        scaling = 1 / warped_point[2]

        warped_point[1] = warped_point[1] * scaling
        warped_point[2] = warped_point[2] * scaling
        warped_point[0] = warped_point[0] * scaling
        np.asarray(warped_point, dtype=np.float32)
        return warped_point

    def read_convex_hull_message_for_one_instance(self, message_dict):
        self.obj_id = message_dict["Object_ID"]
        self.probability = message_dict["Probability"]
        self.bb_coordinates = message_dict["Position_BBox"]
        self.obj_class = message_dict["Class"]
        self.entitiy_mask_img = message_dict["Instance_Mask"]

        if not self.obj_id:
            raise Exception("Message for Instance was not read correctrly or is corrupt! Check Input. Calculation not possible!")
        

    def calculate_ground_contact_point_for_one_instance(self):
        # With Convex Hull
        if self.obj_class != "person":
            self.instance_pts = self.entitiy_mask_img
            self.min_rect_points = self.get_min_rect_points()
            self.ground_contact_points_image = self.find_ground_contact_line()
            self.ground_contact_points_world = self.transform_ground_contact_points_from_image_to_world()
            self.rotated_rvec = self.find_and_rotate_rvec_of_bottom_straight_by_degree(90)
            self.ground_contact_point_world = self.calc_midpoint_from_two_points()
            self.shifted_candidate_1 = self.shift_point_by_rvec_and_object_class()
            self.rotated_rvec = self.find_and_rotate_rvec_of_bottom_straight_by_degree(-90)
            self.ground_contact_point_world = self.calc_midpoint_from_two_points()
            self.shifted_candidate_2 = self.shift_point_by_rvec_and_object_class()
            self.shifted_candidate_1_image = self.transform_point_from_world_to_image(self.shifted_candidate_1)
            self.shifted_candidate_2_image = self.transform_point_from_world_to_image(self.shifted_candidate_2)
            if self.shifted_candidate_1_image[1] < self.shifted_candidate_2_image[1]:
                self.shifted_ground_contact_point_world = self.shifted_candidate_1
            else:
                self.shifted_ground_contact_point_world = self.shifted_candidate_2
        else:
            self.ground_contact_points_image = [[self.bb_coordinates[0], self.bb_coordinates[1] + self.bb_coordinates[3], self.bb_coordinates[0]+0.5*self.bb_coordinates[2]]]
            self.ground_contact_points_world = self.transform_ground_contact_points_from_image_to_world()
            self.shifted_ground_contact_point_world = self.ground_contact_points_world[0]

    def generate_output_message_for_one_instance(self):
        output_dict = dict()
        output_dict["Object_ID"] = self.obj_id
        output_dict["Class"] = self.obj_class
        output_dict["Probability"] = self.probability
        output_dict["Position_World"] = self.shifted_ground_contact_point_world
        output_dict["Color_Hist"] = self.color_hist
        return output_dict

    def __init__(self, h_fname, scale_factor):
            H_file = open(h_fname)
            H = json.load(H_file)
            H = np.array(H)
            self.Homography_Matrix = H
            self.Homography_Matrix = H
            self.inv_Homography_Matrix = np.linalg.inv(self.Homography_Matrix)
            self.scale_factor = scale_factor
            self.obj_id = None
            self.obj_class = None
            self.probability = None
            self.bb_coordinates = None
            self.color_hist = None
            self.entitiy_mask_img = None
            self.bb_coordinates = None
            self.instance_pts = None
            self.min_rect_points = None
            self.ground_contact_points_image = None
            self.ground_contact_points_world = None
            self.rvec_base = None
            self.rotated_rvec = None
            self.ground_contact_point_world = None
            self.shifted_ground_contact_point_world = None
            self.high_bb_flag = None

    def map_entity_and_return_relevant_points(self, track, mask):
        self.obj_id = track.id
        #self.probability = track.score
        #self.bb_coordinates = track.xywh()
        self.obj_class = track.label()
        self.entitiy_mask_img = mask
        self.calculate_ground_contact_point_for_one_instance()

        pt = self.shifted_ground_contact_point_world
        pt_new = (pt[0].item(), pt[1].item())

        #return pt_new, self.rotated_rvec, self.ground_contact_points_image, self.shifted_candidate_1_image, self.shifted_candidate_2_image, self.ground_contact_points_world
        return pt_new, self.rotated_rvec, self.ground_contact_points_image

    def invert_homography(self, point: tuple[float, float]) -> tuple[float, float]:
        point_padded = [point[0], point[1], 1]
        warped_point = np.matmul(self.inv_Homography_Matrix, point_padded)
        scaling = 1 / warped_point[2]
        warped_point *= scaling
        return (warped_point[1], warped_point[0]) # swap coords to line up with data



def map_entity_and_return_relevant_points(message, H_camera):

    pos_est = PositionEstimation(H_camera, scale_factor=632.26/13.5)   # scale vup: 81/5   scale_tum: 632.26/13.5
    pos_est.read_message_for_one_instance(message)
    pos_est.calculate_ground_contact_point_for_one_instance()
   
    return pos_est.shifted_ground_contact_point_world, pos_est.rotated_rvec


if __name__ == "__main__":
    #Load Homography Martix (Matrix is Camera1 from VUP)
    H_camera = np.array([[-0.49445957900116777, -5.336843794679957, 2379.3416459890864], [-0.16617068412336772, -5.121305806291713, 2611.6303323652387], [-5.200759353906408e-05, -0.002518799679750125, 1.0]])
    
    # Messages
    message_dict = dict()

    for message in message_dict:
        gp_world = map_entity_and_return_relevant_points(message, H_camera)



    