import os
import copyreg


import cv2 as cv
import numpy as np
from multiprocessing import Pool
import time


resaved_filename = os.getcwd()+'/media/tempresaved.jpg'

nprocs = 12  ## process sayisi

def _pickle_keypoints(point):
    return cv.KeyPoint, (*point.pt, point.size, point.angle,
                          point.response, point.octave, point.class_id)
copyreg.pickle(cv.KeyPoint().__class__, _pickle_keypoints)


class CopyMoveSIFT: 
    resize_percentage = 100
    max_dist = 150

    def __init__(self,path):

        img_gray = cv.imread(path, cv.IMREAD_GRAYSCALE)
        img_rgb = cv.imread(path, cv.IMREAD_COLOR)

    
        self.img_gray = cv.resize(img_gray, (
            int(img_gray.shape[1] * self.resize_percentage / 100),
            int(img_gray.shape[0] * self.resize_percentage / 100)))
        self.img_rgb = cv.resize(img_rgb, (
            int(img_rgb.shape[1] * self.resize_percentage / 100), int(img_rgb.shape[0] * self.resize_percentage / 100)))

        sift = cv.xfeatures2d.SIFT_create()
        self.keypoints_sift, self.descriptors = sift.detectAndCompute(img_gray, None)
        
        pool = Pool(processes=nprocs)
        matched_pts = pool.map(self.apply_sift, np.array_split(range(len(self.descriptors)), nprocs))
        pool.close()
        self.draw(matched_pts)
        

    def compare_keypoint(self,descriptor1, descriptor2):
        return np.linalg.norm(descriptor1 - descriptor2)


    def apply_sift(self,in_vector):
        out_point_list = []
        for index_dis in in_vector:  # dist = numpy.linalg.norm(a-b)
            for index_ic in range(index_dis + 1, len(self.keypoints_sift)):
                point1_x = int(round(self.keypoints_sift[index_dis].pt[0]))
                point1_y = int(round(self.keypoints_sift[index_dis].pt[1]))
                point2_x = int(round(self.keypoints_sift[index_ic].pt[0]))
                point2_y = int(round(self.keypoints_sift[index_ic].pt[1]))
                if point1_x == point2_x & point1_y == point2_y:
                    # print("benzer keypoints")
                    continue

                dist = self.compare_keypoint(self.descriptors[index_dis], self.descriptors[index_ic])

                if dist < self.max_dist:
                    out_point_list.append([round(self.keypoints_sift[index_dis].pt[0]), round(self.keypoints_sift[index_dis].pt[1]),
                                        round(self.keypoints_sift[index_ic].pt[0]), round(self.keypoints_sift[index_ic].pt[1])])

        if out_point_list:
            return out_point_list


    def draw(self,matched_pts):
        for points in matched_pts:
            if points == None:
                continue
            for in_points in points:
                cv.circle(self.img_rgb, (in_points[0], in_points[1]),
                        4,
                        (0, 0, 255),
                        -1)  # eslesen objeyi isaretlemek icin

                cv.circle(self.img_rgb, (in_points[2], in_points[3]),
                        4,
                        (255, 0, 0),
                        -1)  # eslesen objeyi isaretlemek icin

                img_line = cv.line(self.img_rgb,
                                (in_points[0], in_points[1]),
                                (in_points[2], in_points[3]),
                                (0, 255, 0), 1)
    
        #self.img_rgb.save(resaved_filename, 'JPEG')
        cv.imwrite(resaved_filename,self.img_rgb)


