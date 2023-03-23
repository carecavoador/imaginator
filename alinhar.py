"""
Practical Image Registration and Alignment with OpenCV and Python
https://thinkinfi.com/image-alignment-and-registration-with-opencv/
"""
import cv2
import numpy as np


# Imagem de referência
img_referencia = cv2.imread("Entrada/original.jpg")

# Imagem distorcida
# img_distorcida = cv2.imread("Entrada/distorcido.jpg")
img_distorcida = cv2.imread("Entrada/distorcido_menor.jpg")
# img_distorcida = cv2.imread("Entrada/original_extra-01.jpg")


# Converter para tons de cinza
img_ref_cinza = cv2.cvtColor(img_referencia, cv2.COLOR_BGR2GRAY)
img_dist_cinza = cv2.cvtColor(img_distorcida, cv2.COLOR_BGR2GRAY)
altura, largura = img_ref_cinza.shape

# Configure ORB feature detector Algorithm with 1000 features.
orb_detector = cv2.ORB_create(1000)

# Extract key points and descriptors for both images
keyPoint1, des1 = orb_detector.detectAndCompute(img_dist_cinza, None)
keyPoint2, des2 = orb_detector.detectAndCompute(img_ref_cinza, None)

# Display keypoints for reference image in green color
imgKp_Ref = cv2.drawKeypoints(img_referencia, keyPoint2, 0, (0,222,0), None)
# imgKp_Ref = cv2.resize(imgKp_Ref, (largura//2, altura//2))

# cv2.imshow('Key Points', imgKp_Ref)
# cv2.waitKey(0)


# Match features between two images using Brute Force matcher with Hamming distance
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Match the two sets of descriptors.
matches = matcher.match(des1, des2)

# Sort matches on the basis of their Hamming distance.
# matches.sort(key=lambda x: x.distance)
# matches = sorted(matches, key=lambda x: x.distance)

# Take the top 90 % matches forward.
matches = matches[:int(len(matches) * 0.9)]
no_of_matches = len(matches)

# Display only 100 best matches {good[:100}
imgMatch = cv2.drawMatches(img_dist_cinza, keyPoint2, img_ref_cinza, keyPoint1, matches[:100], None, flags = 2)
# imgMatch = cv2.resize(imgMatch, (largura//3, altura//3))

# cv2.imshow('Image Match', imgMatch)
# cv2.waitKey(0)

# Define 2x2 empty matrices
p1 = np.zeros((no_of_matches, 2))
p2 = np.zeros((no_of_matches, 2))

# Storing values to the matrices
for i, match in enumerate(matches):
    p1[i, :] = keyPoint1[match.queryIdx].pt
    p2[i, :] = keyPoint2[match.trainIdx].pt

# Find the homography matrix.
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC)

# Use homography matrix to transform the unaligned image wrt the reference image.
aligned_img = cv2.warpPerspective(img_distorcida, homography, (largura, altura))
# Resizing the image to display in our screen (optional)
# aligned_img = cv2.resize(aligned_img, (largura//3, altura//3))
aligned_img = cv2.resize(aligned_img, (largura, altura))

# Copy of input image
imgTest_cp = img_dist_cinza.copy()
imgTest_cp = cv2.resize(imgTest_cp, (largura//3, altura//3))
# Save the align image output.
cv2.imwrite('Saida/saida.jpg', aligned_img)

# cv2.imshow('Input Image', imgTest_cp)
# cv2.imshow('Output Image', aligned_img)
# cv2.waitKey(0)
