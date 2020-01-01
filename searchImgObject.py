import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import matplotlib.pyplot as plt

import imutils
import os


path = 'searchImg/'
searchImage = 'images/left2_small.jpg'


sc = ShapeContext()

def get_contour_bounding_rectangles(cannyImg):
    """
      Getting all 2nd level bounding boxes based on contour detection algorithm.
    """

    im2, contours, hierarchy = cv2.findContours(cannyImg, 
                        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    try: hierarchy = hierarchy[0]
    except: hierarchy = []

    height, width = cannyImg.shape
    min_x, min_y = width, height
    max_x = max_y = 0

    # computes the bounding box for the contour, and draws it on the frame,
    for contour, hier in zip(contours, hierarchy):
        (x,y,w,h) = cv2.boundingRect(contour)
        min_x, max_x = min(x, min_x), max(x+w, max_x)
        min_y, max_y = min(y, min_y), max(y+h, max_y)


    if max_x - min_x > 0 and max_y - min_y > 0:
        return min_x, min_y, max_x, max_y
    else:
        return 0, 0, width-1, height-1



def getShapePoints(sc, path):
    """
    Get 'n' random points which describes the shape inside image.
    """
    descs = []

    img = cv2.imread(path, 0)
    edges = cv2.Canny(img,100,200)
    
    # getting our numbers one by one
    min_x, min_y, max_x, max_y = get_contour_bounding_rectangles(edges)
    r = (min_x, min_y, max_x, max_y)

    
    points = sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 1000)
    return np.array(points)


def match(base, current):
    """
      Here we are using correlation diff because we are searching objects.
    """
    res = cdist(base, current, metric="correlation")

    # to consider possible nan values as 0
    return np.nansum(res)



def main():

    queryImgPoints = getShapePoints(sc, searchImage)

    img_files = [f for f in os.listdir(path) if f.endswith('.jpg')]

    matchValues = []

    # Take each file in directory and compare the shape points to compute match value.
    for f in img_files:
        fullpath = os.path.join(path, f)
        imgPoints = getShapePoints(sc, fullpath)
        matchVal = match(queryImgPoints, imgPoints)
        matchValues.append(matchVal)

    print('\nMatch Values')
    print('============')
    
    for i in range(len(matchValues)):
        print(img_files[i] + " \t- " + str(matchValues[i]))
    
    print('\nHere is the object you search for: ' + str(img_files[np.argmin(matchValues)]) + '\n')

    # print(matchValues)
    # print(img_files)


    img = cv2.imread(os.path.join(path, img_files[np.argmin(matchValues)]))
    plt.imshow(img)
    plt.show()


if __name__== "__main__":
  main()