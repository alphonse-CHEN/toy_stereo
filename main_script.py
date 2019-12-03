import cv2
import os
from camera_calibration import camera_calibration, dir_exist_create, get_candidate_imgs_fn_ff, undistort_image, read_img, write_img
from image_rectification import compute_rectification_parameters

__author__ = 'Xuanli CHEN'
"""
Xuanli Chen
version: 4.0.0 beta
PhD student at PSI-ESAT, KU Leuven
Supervisor: Prof. Luc Van Gool
Research Domain: Computer Vision, Machine Learning

Address:
Kasteelpark Arenberg 10 - bus 2441
B-3001 Heverlee
Belgium

Group website: http://www.esat.kuleuven.be/psi/visics
LinkedIn: https://be.linkedin.com/in/xuanlichen
"""

calibration_data_folder = './data/'
capture_data_folder = './data/capture_for_reconstruction'
left_camera_folder = os.path.join(calibration_data_folder, 'left')
left_calib_file = os.path.join(left_camera_folder, 'calib.pkl')
right_camera_folder = os.path.join(calibration_data_folder, 'right')
right_calib_file = os.path.join(right_camera_folder, 'calib.pkl')


# ------ camera calibration -------
rms_left, camera_matrix_left, dist_coeffs_left, rvecs_left, tvecs_left = camera_calibration(
    img_folder=left_camera_folder, out_file_fn=left_calib_file,
    pattern_size=(11, 8),
    w=2592, h=2048, square_size=12
)

rms_right, camera_matrix_right, dist_coeffs_right, rvecs_right, tvecs_right = camera_calibration(
    img_folder=right_camera_folder, out_file_fn=right_calib_file,
    pattern_size=(11, 8),
    w=2592, h=2048, square_size=12
)

# ------- undistort the images --------
left_undistort_camera_folder = os.path.join(calibration_data_folder, 'left_undistorted')
dir_exist_create(left_undistort_camera_folder)
right_undistort_camera_folder = os.path.join(calibration_data_folder, 'right_undistorted')
dir_exist_create(right_undistort_camera_folder)

left_capture_folder = os.path.join(capture_data_folder, 'left')
left_capture_undistort_folder = os.path.join(capture_data_folder, 'undistort_left')
dir_exist_create(left_capture_undistort_folder)
right_capture_folder = os.path.join(capture_data_folder, 'right')
right_capture_undistort_folder = os.path.join(capture_data_folder, 'undistort_right')
dir_exist_create(right_capture_undistort_folder)

left_fns = get_candidate_imgs_fn_ff(left_camera_folder)
for fn in left_fns:
    out_fn = os.path.join(left_undistort_camera_folder, os.path.basename(fn))
    undistort_image(camera_matrix=camera_matrix_left, dist_coeffs=dist_coeffs_left,
                    img_fn=fn, out_fn=out_fn
                    )

right_fns = get_candidate_imgs_fn_ff(right_camera_folder)
for fn in right_fns:
    out_fn = os.path.join(right_undistort_camera_folder, os.path.basename(fn))
    undistort_image(camera_matrix=camera_matrix_left, dist_coeffs=dist_coeffs_left,
                    img_fn=fn, out_fn=out_fn
                    )

left_capture_fns = get_candidate_imgs_fn_ff(left_capture_folder)
for fn in left_capture_fns:
    out_fn = os.path.join(left_capture_undistort_folder, os.path.basename(fn))
    undistort_image(camera_matrix=camera_matrix_left, dist_coeffs=dist_coeffs_left,
                    img_fn=fn, out_fn=out_fn
                    )

right_capture_fns = get_candidate_imgs_fn_ff(right_capture_folder)
for fn in right_capture_fns:
    out_fn = os.path.join(right_capture_undistort_folder, os.path.basename(fn))
    undistort_image(camera_matrix=camera_matrix_right, dist_coeffs=dist_coeffs_right,
                    img_fn=fn, out_fn=out_fn
                    )

# ------- image rectification -------

left_undistort_fns = get_candidate_imgs_fn_ff(left_undistort_camera_folder)
right_undistort_fns = get_candidate_imgs_fn_ff(right_undistort_camera_folder)
F, H_left, H_right = compute_rectification_parameters(
        ref_left_fn_list=left_undistort_fns,
        ref_right_fn_list=right_undistort_fns,
        set_rectification_folder=calibration_data_folder,
        feature='sift',
        matching_method='flann',
        matching_threshold=0.75
)
print('H_left:')
print(H_left)
print('H_right')
print(H_right)

# ------- warp image to rectify --------
left_image = read_img(os.path.join(left_capture_undistort_folder, '1.png'))
right_image = read_img(os.path.join(right_capture_undistort_folder, '1.png'))

left_warped = cv2.warpPerspective(src=left_image, M=H_left, dsize=(2592, 2048))
write_img(fn=os.path.join(calibration_data_folder, 'rectified_left.png'), img=left_warped)
left_warped_res = cv2.resize(left_warped, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
right_warped = cv2.warpPerspective(src=right_image, M=H_right, dsize=(2592, 2048))
write_img(fn=os.path.join(calibration_data_folder, 'rectified_right.png'), img=right_warped)
right_warped_res = cv2.resize(right_warped, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)

# ------ calculate disparity ------
stereo = cv2.StereoSGBM_create(minDisparity=-64, numDisparities=128, blockSize=15, P1=8, P2=256)
disparity = stereo.compute(left=left_warped_res, right=right_warped_res)
from matplotlib import pyplot as plt

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(left_warped, cmap='gray')
axes[0, 1].imshow(right_warped, cmap='gray')
disp_plot = axes[1, 0].imshow(disparity, cmap='jet')
fig.colorbar(disp_plot, ax=axes[0, 1])

# ------ output -------
save_to_fig = True
if save_to_fig:
    plt.savefig(os.path.join(capture_data_folder, 'overview.png'))
else:
    plt.show()
plt.close()
plt.imshow(disparity)
plt.colorbar()
plt.savefig(os.path.join(capture_data_folder, 'disparity.png'))
plt.close()

