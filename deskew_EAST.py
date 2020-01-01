import numpy as np
import cv2
import math
import time
from scipy import ndimage
from os import listdir
from os.path import isfile, join
from imutils.object_detection import non_max_suppression

mypath = 'images/'

# Show result
def showImage(image):
	cv2.imshow("Result Image", image)
	cv2.waitKey(3000)



def getBoundingBox(fullImage):

	(H, W) = fullImage.shape[:2]

	rW = W / 320
	rH = H / 320

	# resize the image and grab the new image dimensions
	image = cv2.resize(fullImage, (320, 320))
	(H, W) = image.shape[:2]

	# define the two output layer names for the EAST detector model that
	# we are interested -- the first is the output probabilities and the
	# second can be used to derive the bounding box coordinates of text
	layerNames = [
		"feature_fusion/Conv_7/Sigmoid",
		"feature_fusion/concat_3"]

	# load the pre-trained EAST text detector
	print("[INFO] loading EAST text detector...")
	net = cv2.dnn.readNet('frozen_east_text_detection.pb')

	# construct a blob from the image and then perform a forward pass of
	# the model to obtain the two output layer sets
	blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
		(123.68, 116.78, 103.94), swapRB=True, crop=False)
	start = time.time()
	net.setInput(blob)
	(scores, geometry) = net.forward(layerNames)
	end = time.time()

	# show timing information on text prediction
	print("[INFO] text detection took {:.6f} seconds".format(end - start))

	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the geometrical
		# data used to derive potential bounding box coordinates that
		# surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability, ignore it
			if scoresData[x] < 0.5:
				continue

			# compute the offset factor as our resulting feature maps will
			# be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and then
			# compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height of
			# the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates for
			# the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score to
			# our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# apply non-maxima suppression to suppress weak, overlapping bounding
	# boxes
	boxes = non_max_suppression(np.array(rects), probs=confidences)

	print("boxes = " + str(len(boxes)))
	print(type(boxes))
	# print(boxes[0])
	print(boxes)

	if (len(boxes) > 0):

		# if any location index is -ve then make it 0 (to correspond to image edge)
		boxes[boxes < 0] = 0
		# loop over the bounding boxes
		for (startX, startY, endX, endY) in boxes:
			# scale the bounding box coordinates based on the respective
			# ratios
			print(startX, startY, endX, endY)
			startX = int(startX * rW)
			startY = int(startY * rH)
			endX = int(endX * rW)
			endY = int(endY * rH)
			# cv2.rectangles
			fullImage = cv2.rectangle(fullImage, (startX, startY), (endX, endY), (255, 0, 0), 2)

		showImage(fullImage)
		return fullImage[startY:endY, startX:endX]
	else:
		return False


def isImgInverted(imgPath, image):

	print(imgPath + image)
	imgFullPath = imgPath + image
	img = cv2.imread(imgFullPath)

	imgCrop = getBoundingBox(img)
	print("inversion check")
	print(imgCrop.size)

	if (imgCrop is False):
		return False

	showImage(imgCrop)



	# im_bw = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	# ret,thresh = cv2.threshold(im_bw, 127,255,0)
	# image, contours, hierarchy = cv2.findContours(
	# 					thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

	# for c in contours:
	# 	rect = cv2.boundingRect(c)
 #    	# rect = cv2.boundingRect(c)
 #    	print(rect)
 #    	print(cv2.contourArea(c))


def rotateImage(imgPath, image):

	imgFullPath = imgPath + image
	img_before = cv2.imread(imgFullPath)

	# print(image)
	# print(img_before)
	# cv2.imshow("Before", img_before)    
	# key = cv2.waitKey(0)

	img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
	img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)
	lines = cv2.HoughLinesP(img_edges, 1, math.pi / 180.0, 
									100, minLineLength=100, maxLineGap=5)

	# print(lines)

	angles = []

	# print(lines)
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

		# showImage(img_rotated)
		print(imgPath+'deskewed/'+image + '_rotated.jpg')
		cv2.imwrite(imgPath+'deskewed/'+image + '_rotated.jpg', img_rotated)


def main():

	onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

	for file in onlyfiles:
		# rotateImage(mypath, file)
		isImgInverted(mypath, file)
    # print("{} files has been rotated.".format(len(f)))

if __name__== "__main__":
  main()