# Base stub obtained from below and modified as required.
# https://stackoverflow.com/questions/55654142/detect-if-a-text-image-is-upside-down
import cv2
import numpy as np
import matplotlib.pyplot as plt

# load image, convert to grayscale, threshold it at 127 and invert.
# page = cv2.imread('images/upright.jpg')
page = cv2.imread('images/inverted.jpg')

page = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
page = cv2.threshold(page, 127, 255, cv2.THRESH_BINARY_INV)[1]

# project the page to the side and smooth it with a gaussian
projection = np.sum(page, 1)
gaussian_filter = np.exp(-(np.arange(-3, 3, 0.1)**2))
gaussian_filter /= np.sum(gaussian_filter)
smooth = np.convolve(projection, gaussian_filter)

x = range(len(projection))
plt.figure()
plt.plot(x, projection)
plt.show()


x = range(len(smooth))
plt.figure()
plt.plot(x, smooth)
plt.show()

# find the pixel values where we expect lines to start and end
mask = smooth > np.average(smooth)
edges = np.convolve(mask, [1, -1])
line_starts = np.where(edges == 1)[0]
line_endings = np.where(edges == -1)[0]

# count lines with peaks on the lower side
lower_peaks = 0
for start, end in zip(line_starts, line_endings):
    line = smooth[start:end]
    if np.argmax(line) < len(line)/2:
        lower_peaks += 1

print(lower_peaks)
print(line_starts)
flip_confidence = float(lower_peaks) / len(line_starts)*100

# if more peaks are on the lower side then orientation is correct (not inverted)

if (flip_confidence < 50):
	print('Flipped Image Confidence: ' + str(100-flip_confidence))
else:
	print('Upright Image Confidence: ' + str(flip_confidence))