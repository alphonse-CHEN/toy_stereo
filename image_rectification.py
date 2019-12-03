import cv2
from camera_calibration import read_img, write_img
import os
import pickle as pkl
import numpy as np


def image_sequence_feature_matching(fn1_list, fn2_list,
                                    process_folder,
                                    out_matching_pair_fn,
                                    feature,
                                    matching_method,
                                    matching_threshold=0.75,
                                    ):
    """
    using OpenCV libraries to process two list of filenames matching. the processed results are stored in process_folder
    i.e. pts, matching drawings
    :param fn1_list: the filename list of left image(s)
    :param fn2_list: the filename list of right image(s)
    :param process_folder: intermediate output folder e.g. temp \ set_xx \ rectification
    :param out_matching_pair_fn: the pkl file stores the matching pts
    :param feature: which feature to use, support sift, surf, brief
    :param matching_method: bf (brutal force) or flann
    :param matching_threshold: matching threshold to select out the matches, default to be 0.75
    :return:
    """

    pts_1 = []
    pts_2 = []

    if len(fn1_list) == 1:
        fn1_list *= len(fn2_list)
    if len(fn2_list) == 1:
        fn2_list *= len(fn1_list)
    if len(fn1_list) != len(fn2_list):
        raise ValueError("Length of Left image list is %i, while Right image is %i . They are Not equal." % (
            len(fn1_list), len(fn2_list)))

    for i, (fn1, fn2) in enumerate(zip(fn1_list, fn2_list)):

        img1 = read_img(fn1)
        img2 = read_img(fn2)
        out_matching_img_fn = "matching_img-fn1-%s-fn2-%s.jpg" % (os.path.split(fn1)[-1].rsplit('.')[0],
                                                                  os.path.split(fn2)[-1].rsplit('.')[0])
        good_pts1, good_pts2 = feature_matching(img1=img1, img2=img2, feature=feature,
                                                matching_method=matching_method,
                                                matching_threshold=matching_threshold,
                                                out_matching_fn=os.path.join(process_folder, out_matching_img_fn)
                                                )
        pts_1.extend(good_pts1)
        pts_2.extend(good_pts2)

    data = [pts_1, pts_2]
    with open(out_matching_pair_fn, 'wb') as out_file:
        pkl.dump(data, out_file)
    return data


def compute_rectification_parameters(
        ref_left_fn_list,
        ref_right_fn_list,
        set_rectification_folder,
        feature,
        matching_method='flann',
        matching_threshold=0.75):
    """
    :param feature: 'sift', 'surf' from OpenCV etc.
    :param matching_method: 'bf' or 'flann', OpenCV related parameters
    :param matching_threshold: distance threshold for OpenCV
    :return: homographies and fundamental matrix
    """
    out_matching_pair_fn = os.path.join(set_rectification_folder, 'sift_matching_pairs')

    if os.path.isfile(out_matching_pair_fn):
        with open(out_matching_pair_fn, 'rb') as f:
            data = pkl.load(f)
    else:
        data = image_sequence_feature_matching(
            fn1_list=ref_left_fn_list,
            fn2_list=ref_right_fn_list,
            process_folder=set_rectification_folder,
            out_matching_pair_fn=out_matching_pair_fn,
            feature=feature,
            matching_method=matching_method,
            matching_threshold=matching_threshold,
        )
    pts1, pts2 = data
    H1_left, H2_right, F = OpenCV_rectification_using_points(pts1, pts2,
                                                             outlier_method=cv2.RANSAC
                                                             )

    return F.astype(np.float32), H1_left.astype(np.float32), H2_right.astype(np.float32)


def OpenCV_rectification_using_points(pts1, pts2, outlier_method=cv2.RANSAC, img_size=(2048, 2592)):
    """
    with the matching points pts1 and pts2_np, previously found by OpenCV_sequence_matching or OpenCV_feature_matching,
    calculate rectification information and fundamental matrix.
    :param pts1: matching points in left image(s)
    :param pts2: matching points in right image(s)
    :param outlier_method: cv2.RANSAC or cv2.FM_LMEDS
    :param img_size: the image size used, assume to be (2592, 2048)
    :param hist_out_fn: default None, if given, write histogram of Hx errors to given fn
    :return: H1, H2, F
    """
    if len(pts1) == 0:
        raise ValueError("No Matching Points Found. Quit.")
    if len(pts2) == 0:
        raise ValueError("No Matching Points Found. Quit.")

    pts1_np = np.array(pts1)
    pts2_np = np.array(pts2)

    F, mask = cv2.findFundamentalMat(pts1_np, pts2_np, outlier_method)
    pts1_selected = pts1_np[mask.ravel() == 1]
    pts2_selected = pts2_np[mask.ravel() == 1]
    flag, H1, H2 = cv2.stereoRectifyUncalibrated(points1=pts1_selected, points2=pts2_selected, F=F,
                                                 imgSize=img_size)
    print('Results are calculated based on %i matching points.' % np.sum(mask.ravel() == 1))
    print('The Fundamental Matrix is: ')
    print(F)

    return H1, H2, F


def feature_matching(img1, img2,
                     feature='sift',
                     matching_method='bf',
                     matching_threshold=0.3,
                     kNN_number=2,
                     out_matching_fn=None
                     ):
    """
    The function process image 1 and image 2, and establish large amount of correspondence.
    :param img1:
    :param img2:
    :param feature:
    :param matching_method:
    :return:
    OpenCV - SIFT: https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html
    OpenCV - feature matching: https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_feature2d/py_matcher/py_matcher.html
    """

    if feature == 'sift':
        feature_detector = cv2.xfeatures2d.SIFT_create()
    elif feature == 'surf':
        feature_detector = cv2.xfeatures2d.SURF_create()
    elif feature == 'brief':
        feature_detector = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    else:
        raise NameError('The extraction method:[ %s ] is not supported. Terminating...' % feature)

    kp1, des1 = feature_detector.detectAndCompute(img1, None)
    kp2, des2 = feature_detector.detectAndCompute(img2, None)

    if matching_method == 'bf':
        matcher = cv2.BFMatcher()
        matches = matcher.match(des1, des2)
    elif matching_method == 'flann':
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = matcher.knnMatch(des1, des2, k=kNN_number)
    else:
        raise NameError('The matching method:[ %s ] is not supported' % matching_method)

    # --- perform matching and apply threshold ---
    good = []
    pts1 = []
    pts2 = []
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    for i, (m, n) in enumerate(matches):
        if m.distance < matching_threshold * n.distance:
            matchesMask[i] = [1, 0]
            good.append(m)
            pts2.append(kp2[m.trainIdx].pt)
            pts1.append(kp1[m.queryIdx].pt)
    print("In total %i Good Matching Found. " % (len(good)))

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=0)

    matching_img = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    write_img(fn=out_matching_fn, img=matching_img)

    return pts1, pts2
