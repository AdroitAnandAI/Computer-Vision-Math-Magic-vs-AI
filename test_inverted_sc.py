import numpy as np
import cv2
import sys
from scipy.spatial.distance import cdist, cosine
from shape_context import ShapeContext
import matplotlib.pyplot as plt

import imutils

numberImage = 'images/numbers.png'
uprightImage = 'images/numbers_test4.png'
invertedImage = 'images/numbers_test4_inverted.png'

sc = ShapeContext()

def get_contour_bounding_rectangles(gray):
    """
      Getting all 2nd level bouding boxes based on contour detection algorithm.
    """

    print(gray.shape)
    cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    res = []
    for cnt in cnts[1]:
        (x, y, w, h) = cv2.boundingRect(cnt)
        res.append((x, y, x + w, y + h))

    return res

def parse_nums(sc, path):
    img = cv2.imread(path, 0)
    # invert image colors
    img = cv2.bitwise_not(img)
    _, img = cv2.threshold(img, 254, 255, cv2.THRESH_BINARY)
    # making numbers fat for better contour detectiion
    kernel = np.ones((2, 2), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    
    print('After thresholding and dilation...')
    plt.imshow(img)
    plt.show()
    # getting our numbers one by one
    rois = get_contour_bounding_rectangles(img)
    grayd = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
    nums = []
    for r in rois:
        grayd = cv2.rectangle(grayd, (r[0], r[1]), (r[2], r[3]), (0, 255, 0), 1)
        nums.append((r[0], r[1], r[2], r[3]))

    print('After greying and bounding...')
    plt.imshow(grayd)
    plt.show()
    # we are getting contours in different order so we need to sort them by x1
    nums = sorted(nums, key=lambda x: x[0])
    print('bounding box x coords')
    print(nums)
    descs = []
    for i, r in enumerate(nums):

        points = sc.get_points_from_img(img[r[1]:r[3], r[0]:r[2]], 15)
        descriptor = sc.compute(points).flatten()
        descs.append(descriptor)
    # print(descs)
    return np.array(descs)

def match(base, current):
    """
      Here we are using cosine diff instead of "by paper" diff, cause it's faster
    """
    res = cdist(base, current.reshape((1, current.shape[0])), metric="cosine")
    # print("min = " + np.argmin(res.reshape(11)))
    char = str(np.argmin(res.reshape(11)))
    if char == '10':
        char = "/"
    # print(char)
    print(np.min(res.reshape(11)))
    return char, np.min(res.reshape(11))



def findUpright(baseImage, firstImg, secondImg):
    base_0123456789 = parse_nums(sc, baseImage)
    # base_0123456789 = parse_nums(sc, '../resources/sc/4.jpg')
    recognize = parse_nums(sc, firstImg)
    recognize_inverted = parse_nums(sc, secondImg)
    # print(base_0123456789)
    txt = ""
    matchFactor = 0
    val = 0
    for r in recognize:
        c, val = match(base_0123456789, r)
        txt += c
        matchFactor += val

    txtInverted = ""
    matchFactorInv = 0
    val = 0
    for r in recognize_inverted:
        c, val = match(base_0123456789, r)
        txtInverted += c
        matchFactorInv += val

    print("\nUpright Text Match Value = " + str(matchFactor))
    print("Flip Text Match Value = " + str(matchFactorInv))    

    if (matchFactor > matchFactorInv):
        return secondImg, firstImg
    else:
        return firstImg, secondImg


def main():

    upImg, invImg = findUpright(numberImage, uprightImage, invertedImage)

    print("\n\nThis is the upright Image: " + upImg)
    print("This is the inverted Image: " + invImg)

    img = cv2.imread(upImg)
    plt.imshow(img)
    plt.show()

if __name__== "__main__":
  main()