"""
ASL decoder app
"""

import os
import yaml
import cv2
import numpy as np
from src.features.featurizer import Featurizer
from src.models.ensemble.classifier import ASLClassifier
import src.utils as utils


def full_path(path):
    """
    Append the given path to this file's full path
    """
    this_path = os.path.dirname(os.path.realpath(__file__))
    return os.path.join(this_path, path)


def get_img_num(path):
    img_num = 0
    if not os.path.isdir(path):
        os.makedirs(path)
        return img_num

    for _, _, filenames in os.walk(path):
        for filename in filenames:
            file_parts = filename.split("_")
            try:
                cur_num = int(file_parts[1])
                if cur_num > img_num:
                    img_num = cur_num
            except ValueError:
                continue
    return img_num + 1


def save_file(save_path, img_num):
    filename = os.path.join(save_path, "save_{}.jpg".format(img_num))
    return full_path(filename)


def choose_featurizer(featurizer, ft_name):
    if ft_name == "fft":
        return featurizer.fft
    elif ft_name == "rgb":
        return featurizer.rgb
    elif ft_name == "dwt":
        return featurizer.dwt
    elif ft_name == "sift":
        return featurizer.sift
    elif ft_name == "surf":
        return featurizer.surf
    elif ft_name == "orb":
        return featurizer.orb


def run():
    yaml_path = full_path(r"config.yaml")
    with open(yaml_path) as yaml_file:
        config = yaml.load(yaml_file, Loader=yaml.FullLoader)

    if config is None:
        print("Could not load config file.")
        return

    ft_name = config["featurizers"]["featurizer"]
    ft_config = config["featurizers"][ft_name]
    clf_config = config["classification"]
    dt = 1000.0 / config["app"]["fps"]
    save_path = full_path(config["app"]["save_path"])
    img_num = get_img_num(save_path)

    featurizer = Featurizer(interactive=False)
    featurize = choose_featurizer(featurizer, ft_name)
    classifier = ASLClassifier(clf_config)
    cap = cv2.VideoCapture(0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2.5
    font_color = (0, 0, 255)
    font_thickness = 2
    font_line_type = cv2.LINE_AA

    if not cap.isOpened():
        print("Cannot open camera")
        return

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    xmin = int((height / 2) - 100)
    xmax = xmin + 200
    ymin = int((width / 2) - 100)
    ymax = ymin + 200

    font_y = ymin
    font_x = int(0.1 * height)

    ret, frame = cap.read()
    print(xmin, xmax)
    print(ymin, ymax)
    print(frame.shape)

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = img[xmin:xmax, ymin:ymax]
        feature = featurize(img[np.newaxis, ...], ft_config)
        pred = utils.number_to_label(classifier.predict(feature)[0])

        frame = cv2.putText(frame, pred, (font_y, font_x), font, font_scale,
                            font_color, font_thickness, font_line_type)
        frame = cv2.rectangle(frame, (ymin, xmin),
                              (ymax, xmax), font_color, font_thickness)

        print(pred)
        cv2.imshow("ASL Classifier", frame)
        key = cv2.waitKey(int(dt)) & 0xff
        if key == ord('y'):
            cv2.imwrite(save_file(save_path, img_num), frame)
            img_num += 1
        elif key == ord('q'):
            cv2.destroyAllWindows()
            return


if __name__ == '__main__':
    run()
