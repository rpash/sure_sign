import numpy as np
import cv2
import pywt

img = cv2.imread(
    "dataset/asl_alphabet_test/asl_alphabet_test/space_test.jpg", 0)

thresh = 0.5

A, lv3, lv2, lv1 = pywt.wavedec2(img, "haar", level=3)
recon = np.array(A / np.max(A) * 255, dtype=np.uint8)

for H, V, D in [lv1, lv2, lv3]:
    H = np.array(H)
    V = np.array(V)
    D = np.array(D)
    H[H < thresh * np.max(H)] = 0
    V[V < thresh * np.max(V)] = 0
    D[D < thresh * np.max(D)] = 0

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.imshow("Image", recon)
cv2.waitKey(0)
print(recon.shape)