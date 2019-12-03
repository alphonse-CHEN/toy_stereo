import cv2
import numpy as np
import pickle as pkl
import os
from PIL import Image


def dir_exist_create(folder, warning=False):
    if folder is None:
        print("Path is None, Do NOTHING !")
        return
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        if warning:
            print("Folder: %s ALREADY exists. Not creating. Be aware of over-writing !" % folder)


def read_img(fn, grey=False):
    """
    read image using PIL, and return as numpy array.
    :param fn: filename of the target image
    :param grey: whether convert into gray scale
    :return: image as numpy array in uint8
    """
    if grey:
        file = Image.open(fn).convert('L')  # if original image is gray scale, then directly read without conversion.
        img = np.asarray(file).astype(np.uint8)
    else:
        file = Image.open(fn)
        img = np.asarray(file)

    return img


def write_img(fn, img):
    """
    output image(np array format) into a file.
    :param fn: filename of output image
    :param img: image in np.array format
    :return: a flag
    """
    img_PIL = Image.fromarray(img)
    img_PIL.save(fn)


def undistort_image(camera_matrix, dist_coeffs, img_fn, out_fn):
    img = read_img(img_fn, grey=True)
    # ---- convert to OpenCV BGR ----
    undistort_img = cv2.undistort(src=img,
                                  cameraMatrix=camera_matrix,
                                  distCoeffs=dist_coeffs,
                                  newCameraMatrix=None)
    write_img(out_fn, undistort_img)

def get_candidate_imgs_fn_ff(folder, candidate_ext=['.png', '.bmp']):
    """
    This function scan the files in a folder and return the image filename list.
    :param folder: where the image folder located
    :param candidate_ext: the extension of the candidate images
    :return: the abspath of images (explicit by candidate_ext) in a list
    """
    imgs_fn = [os.path.abspath(os.path.join(folder, f)) for f in os.listdir(folder)
               if os.path.isfile(os.path.join(folder, f)) and
               (os.path.splitext(f)[-1] in candidate_ext) and
               (not os.path.split(f)[-1].startswith('.'))
               ]

    return imgs_fn


def camera_calibration(img_folder, out_file_fn,
                       pattern_size=(11, 8), w=2592, h=2048, square_size=12):
    """
    camera calibration function calling OpenCV.
    :param img_folder: the calibration image folder, source images are acquired
    :param out_file_fn: calibration filename
    :param pattern_size: the square pattern size i.e. the number of internal cross from square blocks
    :param w: the width of the image
    :param h: the height of the image
    :param square_size: mm of square e.g. 12 mm
    :return: rms, camera_matrix, dist_coeffs, rvecs, tvecs  as defined in OpenCV::cv2.calibrateCamera
    """

    if os.path.isfile(out_file_fn):
        print("File Found. Loading...")
        with open(out_file_fn, 'rb') as f:
            [rms, camera_matrix, dist_coeffs, rvecs, tvecs] = pkl.load(f)

    else:
        print("Processing... %s " % out_file_fn)
        # ---------------- Pattern Points ----------------------
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size
        # ---------------- Refer Folders -----------------
        ref_cap_fns = get_candidate_imgs_fn_ff(img_folder)
        # print(ref_cap_fns)
        ref_cap_fns.sort()
        output_calibration_file_folder = os.path.split(out_file_fn)[0]
        debug_folder = os.path.join(output_calibration_file_folder, 'debug')
        if not os.path.isdir(debug_folder):
            os.mkdir(debug_folder)

        pts = []
        obj_points = []

        print("Processing to %s: " % output_calibration_file_folder)
        for fn in ref_cap_fns:

            now_img = read_img(fn, grey=True)
            print("Calculating ==> %s" % os.path.basename(fn))

            found, corners = cv2.findChessboardCorners(now_img, pattern_size)
            if found:
                term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                cv2.cornerSubPix(now_img, corners, (5, 5), (-1, -1), term)
            if os.path.isdir(debug_folder):
                vis = cv2.cvtColor(now_img, cv2.COLOR_GRAY2BGR)
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                cv2.imwrite(os.path.join(debug_folder, 'cali_%s_cb.png' % fn), vis)
            if not found:
                print('chessboard not found')
                continue
            pts.append(corners.reshape(-1, 2))
            obj_points.append(pattern_points)
        rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points,
                                                                            pts,
                                                                            (w, h),
                                                                            None,
                                                                            None,
                                                                            )
        data = [rms, camera_matrix, dist_coeffs, rvecs, tvecs]
        with open(out_file_fn, 'wb') as f:
            pkl.dump(data, f)
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs