"""
https://pyimagesearch.com/2017/06/19/image-difference-with-opencv-and-python/
Image Difference with OpenCV and Python
"""
from skimage.metrics import structural_similarity as compare_ssim
import imutils
import cv2


img_referencia = "Entrada/original.jpg"
img_teste = "Saida/alterado.jpg"
img_teste = "Saida/saida.jpg"

# load the two input images
imageA = cv2.imread(img_referencia)
imageB = cv2.imread(img_teste)
# convert the images to grayscale
grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

ALTURA, LARGURA = grayA.shape

# compute the Structural Similarity Index (SSIM) between the two
# images, ensuring that the difference image is returned
(score, diff) = compare_ssim(grayA, grayB, full=True)
diff = (diff * 255).astype("uint8")
print("SSIM: {}".format(score))

# threshold the difference image, followed by finding contours to
# obtain the regions of the two input images that differ
thresh = cv2.threshold(diff, 0, 255,
	cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

# loop over the contours
for c in cnts:
    # compute the bounding box of the contour and then draw the
    # bounding box on both input images to represent where the two
    # images differ
    (x, y, w, h) = cv2.boundingRect(c)
    cv2.rectangle(imageA, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(imageB, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Resize images to display
# imageA = cv2.resize(imageA, (ALTURA//3, LARGURA//3))

# show the output images
# cv2.imshow("Original", imageA)
# cv2.imshow("Modified", imageB)
# cv2.imshow("Diff", diff)
# cv2.imshow("Thresh", thresh)
# cv2.waitKey(0)

cv2.imwrite("Diferencas/diff.jpg", diff)
cv2.imwrite("Diferencas/threshold.jpg", thresh)
cv2.imwrite("Diferencas/original.jpg", imageA)
cv2.imwrite("Diferencas/modificado.jpg", imageB)
