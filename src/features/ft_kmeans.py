"""
Featurization method collection for image descriptors quantized by k-means
"""
import os
from multiprocessing import Pool
import itertools
import numpy as np
from sklearn.cluster import MiniBatchKMeans
import cv2
import src.utils as utils


def _get_featurizer(featurizer_name):
    """
    Creates a image keypoint featurizer from its name. Supports:
        - SIFT
        - SURF
        - ORB
    Input:
        featurizer_name: name of the featurizer
    Output:
        featurizer: An instance of the keypoint featurizer
    """
    if featurizer_name == 'sift':
        return cv2.xfeatures2d.SIFT_create()
    elif featurizer_name == 'surf':
        return cv2.xfeatures2d.SURF_create()
    elif featurizer_name == 'orb':
        return cv2.ORB_create()
    return None


def _compute_features(args):
    """
    Compute keypoint features for a batch of images
    Input:
        data: (N,H,W[,C]) matrix of N training images
    Output:
        features: (N, M) matrix of M-length feature vectors of N images
    """
    featurizer = _get_featurizer(args[1])
    descriptors = []
    no_det = []  # need to keep track of no detections to fill them with zeros
    for i, img in enumerate(args[0]):
        _, desc = featurizer.detectAndCompute(img, None)
        descriptors.append(desc)
        if desc is None:
            no_det.append(i)

    # feature length varies so we need to find the correct length
    # if all images had no detections we are screwed anyway so don't worry
    fill_in = None
    for desc in descriptors:
        if desc is not None:
            fill_in = np.zeros(desc.shape)
            break

    for i in no_det:
        descriptors[i] = fill_in

    return descriptors


class KMeansFeaturizer:
    def __init__(self, vocab_size, featurizer="sift"):
        self.__vocab_size = vocab_size
        self.__kmeans = None
        self.__name = featurizer
        self.__featurizer = _get_featurizer(featurizer)

    def train(self, data, pickle_path=None):
        """
        Train this featurizer on the training set using the following procedure
        1. Compute the keypoint descriptors for each image. keypoint descriptors
           are M-length vectors for each detected keypoint in an image. There
           can be an arbitrary number of keypoints per image
        2. Perform k-means clustering on the set of all descriptors gathered
           from all images. The descriptors will be divided into K groups (K is
           specified in the constructor). The distribution of an image's
           keypoints among these K groups (wich was computed using each
           keypoint's descriptor) determines the feature vector of that image.
        3. For each image, go through the keypoints of that image and increment
           the element in the zero-initialized feature vector which corresponds
           to the label of that image. In essence, each keypoint of an image
           votes on which of K groups it thinks the image belongs to. These
           votes become the feature vector of that image.
        Input:
            data: (N,H,W[,C]) matrix of N training images
            pickle_path: location of pickle file to save/load model
        Output:
            features: (N, K) matrix of K-length feature vectors of N images
        """
        if pickle_path is not None:
            self.__kmeans = utils.load_model(pickle_path)
            if self.__kmeans is not None:
                print("Skipping training")
                return self.test(data)

        # Use multiple processes to calculate descriptors
        # Use all but one thread if possible to prevent crashes when
        # using all threads. Pool doesn't like it when there is not much data
        # so if there is only need for 1 processor we don't pool. Otherwise we
        # will block indefinitely for some reason.
        n_images = data.shape[0]
        features = np.zeros((n_images, self.__vocab_size))
        nprocs = (os.cpu_count() - 1) if os.cpu_count() > 1 else 1
        nprocs = nprocs if n_images > (nprocs * 20) else 1
        print("Computing {} descriptors using {} processes".format(
            self.__name.upper(), nprocs))
        if nprocs > 1:
            subsets = np.array_split(data, nprocs, axis=0)
            pool = Pool(processes=nprocs)
            pool_args = zip(subsets, itertools.repeat(self.__name))
            batch_descriptors = pool.map(_compute_features, pool_args)
            img_descriptors = np.concatenate([d for d in batch_descriptors])
        else:
            img_descriptors = _compute_features((data, self.__name))

        # Reshape data so that we can give k-means a list of descriptors
        # while maintaining a descriptor->image mapping which we will need
        # to generate features later
        img_ids, descriptors = [], []
        for i, img_desc in enumerate(img_descriptors):
            img_ids.extend([i] * len(img_desc))
            descriptors.extend([desc for desc in img_desc])

        # k-means to determine a feature vector for each image
        # Use MiniBatchKmeans for speed
        self.__kmeans = MiniBatchKMeans(
            n_clusters=self.__vocab_size).fit(descriptors)

        # We can assume that the image IDs calculated before correspond
        # to the correct k-means label because MiniBatchKMeans preserves order
        # Otherwise we would have to use preddict(X) on each image (slow)
        for img_id, cluster_id in zip(img_ids, self.__kmeans.labels_):
            features[img_id, cluster_id] += 1

        if pickle_path is not None:
            utils.save_model(self.__kmeans, pickle_path)

        return features

    def test(self, data):
        """
        Compute the feature vector of each image in `data` using the k-means
        clustering computed using `train`.
        NOTE: THE FIST AXIS MUST BE THE IMAGE NUMBER, EVEN FOR SINGLE IMAGE
        Input:
            data: (N,H,W[,C]) matrix of N training images
        Output:
            features: (N, K) matrix of K-length feature vectors of N images
        """
        if self.__kmeans is None:
            return None

        n_images = data.shape[0]
        features = np.zeros((n_images, self.__vocab_size))
        for i, img in enumerate(data):
            features[i] = self.featurize(img)

        return features

    def featurize(self, img):
        """
        Compute the feature vector of a single image using the k-means
        clustering computed using `train`
        Input:
            img: (H,W[,C]) image with (optional) C channels
        Output:
            features: K-length feature vector of the input image
        """
        if self.__kmeans is None:
            return None

        features = np.zeros(self.__vocab_size)
        _, desc = self.__featurizer.detectAndCompute(img, None)

        # Occasionally no keypoints are detected so use empty vector
        if desc is None:
            return features
        pred = self.__kmeans.predict(desc)
        for vote in pred:
            features[vote] += 1

        return features
