import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def full_path(path):
    """
    Append the given path to this file's full path
    """
    this_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_path, path)


# flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS

def run():
    label = "C"
    sift_pred = "C"
    surf_pred = "C"
    orb_pred = "C"
    test_path = full_path(
        "dataset/asl_alphabet_test/asl_alphabet_test/{}_test.jpg".format(label))
    img = cv2.imread(test_path, cv2.IMREAD_GRAYSCALE)
    """
    X = []
    y = []
    for path, _, filenames in os.walk(test_path):
        for f in filenames:
            X.append(cv2.imread(os.path.join(
                path, f), cv2.IMREAD_GRAYSCALE))
            y.append(f.split('_')[0])
    """
    sift = cv2.xfeatures2d.SIFT_create()
    surf = cv2.xfeatures2d.SURF_create()
    orb = cv2.ORB_create()

    #for img, label in zip(X, y):
    plt.suptitle("Keypoint Featurizers with K-Means")
    plt.subplot(2, 2, 1)
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    plt.title('Original {}'.format(label))

    plt.subplot(2, 2, 2)
    kp = sift.detect(img, None)
    plt.imshow(cv2.drawKeypoints(img, kp, None))
    plt.axis("off")
    plt.title("SIFT Keypoints {}".format(sift_pred))

    plt.subplot(2, 2, 3)
    kp = surf.detect(img, None)
    plt.imshow(cv2.drawKeypoints(img, kp, None))
    plt.axis("off")
    plt.title("SURF Keypoints {}".format(surf_pred))

    plt.subplot(2, 2, 4)
    kp = orb.detect(img, None)
    plt.imshow(cv2.drawKeypoints(img, kp, None))
    plt.axis("off")
    plt.title("ORB Keypoints {}".format(orb_pred))

    plt.show()


if __name__ == '__main__':
    run()
