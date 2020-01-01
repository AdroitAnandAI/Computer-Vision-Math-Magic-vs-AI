import numpy as np
import cv2
import math
from scipy import ndimage
from os import listdir
from os.path import isfile, join

mypath = 'images/'

# Show result
def showImage(image):
	cv2.imshow("Result Image", image)
	cv2.waitKey(3000)


def rotateImage(imgPath, image):

	imgFullPath = imgPath + image
	img_before = cv2.imread(imgFullPath)

	img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
	img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
	lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 100, minLineLength=100, maxLineGap=5)

	angles = []

	if (lines is not None):
		for x1, y1, x2, y2 in lines[0]:
		    # cv2.line(img_before, (x1, y1), (x2, y2), (255, 0, 0), 3)
		    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
		    angles.append(angle)

		print(angles)
		median_angle = np.median(angles)

		if (median_angle != 0):
			img_rotated = ndimage.rotate(img_before, median_angle)
		else:
			img_rotated = img_before

		# if (median_angle >= 0):
		# 	img_rotated = ndimage.rotate(img_before, median_angle)
		# else:
		# 	img_rotated = ndimage.rotate(img_before, 180+median_angle)

		print "Angle is {}".format(median_angle)

		print(imgPath+'deskewed/'+image + '_rotated.jpg')
		cv2.imwrite(imgPath+'deskewed/'+image + '_rotated.jpg', img_rotated)


def main():

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	for file in onlyfiles:
		rotateImage(mypath, file)

    # print("{} files has been rotated.".format(len(f)))

if __name__== "__main__":
  main()