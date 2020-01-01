import numpy as np
import cv2
import matplotlib.pyplot as plt


# img1 = cv2.imread('images/IMG_8295.JPG',0)
# img2 = cv2.imread('images/IMG_8292.JPG',0)

template = cv2.imread('images/book.jpg',0)
img = cv2.imread('images/box.png',0)

orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(template,None)
kp2, des2 = orb.detectAndCompute(img,None)

# Brute-Force Matching with SIFT Descriptors and Ratio Test
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1,des2, k=2)

# Apply ratio test explained by D.Lowe in his paper.
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)


matches = sorted(good, key = lambda x:x.distance)

# cv2.drawMatchesKnn expects list of lists as matches.
img3 = cv2.drawMatchesKnn(template,kp1,img,kp2,[matches[:3]],None,flags=2)

plt.imshow(img3)
plt.show()

## extract the matched keypoints
src_pts  = np.float32([kp1[m.queryIdx].pt for m in matches[:5]]).reshape(-1,1,2)
dst_pts  = np.float32([kp2[m.trainIdx].pt for m in matches[:5]]).reshape(-1,1,2)

## find homography matrix and do perspective transform
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
h,w = template.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv2.perspectiveTransform(pts,M)

## draw found regions
img2 = cv2.polylines(img, [np.int32(dst)], True, (0,255,255), 3, cv2.LINE_AA)

plt.imshow(img2)
plt.show()