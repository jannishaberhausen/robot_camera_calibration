import argparse
import cv2
import json
import numpy as np
import os


CHESSBOARD_DIMS = (9, 6)
IMAGE_FOLDER = 'images/'
ROBOT_TRANSFORMS = 'robot_transforms.json'
PHYSICAL_SQUARE_SIZE = 0.01 # 1cm


### --- CHESSBOARD DETECTION --- ###
def chessboard_detection_from_memory(directory, verbose, CHESSBOARD_SIZE=CHESSBOARD_DIMS, PHYSICAL_SQUARE_SIZE=PHYSICAL_SQUARE_SIZE, scale_factor=1.0):
    """"Using opencv-python functionality to detect chessboard corners in images. The images are loaded from the folder given by 'directory'.
    The images themselves must be named as '0.png', '1.png', '2.png', etc. If the images are high resolution, it is recommended to use a 
    scale_factor < 1.0 to speed up the process. The function returns the 2d image points, 3d object points and the images themselves."""
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp = np.zeros((CHESSBOARD_SIZE[0]*CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= PHYSICAL_SQUARE_SIZE
    
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    images = []
    n_images = 0    
    for filename in os.scandir(directory):
        if filename.is_file():
            n_images += 1
    for i in range(n_images):
        img = cv2.imread(f'{directory}/{i}.png')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        scaled_img = cv2.resize(gray, (0, 0), fx=scale_factor, fy=scale_factor) 
        ret, corners = cv2.findChessboardCorners(scaled_img, CHESSBOARD_SIZE, None)
        if ret:
            corners_refined = cv2.cornerSubPix(scaled_img, corners, (11, 11), (-1, -1), criteria)    
            imgpoints.append(corners_refined / scale_factor)
            objpoints.append(objp)
            images.append(gray)
            if verbose:
                scaled_img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
                cv2.drawChessboardCorners(scaled_img, CHESSBOARD_SIZE, corners_refined, ret)
                cv2.imshow('Detected Chessboard', scaled_img)
                cv2.waitKey(500)
    return imgpoints, objpoints, images


def load_robot_transforms(file):
    R_list = []
    t_list = []
    with open(file, 'r') as f:
        data = json.load(f)
    confs = data['robot_transforms']
    for conf in confs:
        R_list.append(np.array(conf['R_gripper2base']))       
        t_list.append(np.array(conf['t_gripper2base']))
    return R_list, t_list


### --- EYE IN HAND CALIBRATION --- ###
def eye_in_hand_calibration(R_target2cam, t_target2cam, robot_conf_file=ROBOT_TRANSFORMS):
    R_gripper2base, t_gripper2base = load_robot_transforms(robot_conf_file)
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam)
    return R_cam2gripper, t_cam2gripper


def save_camera_calibration(intrinsic_matrix, dist_coeffs, R_cam2gripper, t_cam2gripper, file='camera_params.yaml'):
    fs = cv2.FileStorage(file, cv2.FILE_STORAGE_WRITE)
    fs.write('camera_matrix', intrinsic_matrix)
    fs.write('distortion_coefficients', dist_coeffs)
    fs.write('R_cam2gripper', R_cam2gripper)
    fs.write('t_cam2gripper', t_cam2gripper)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--verbose',
                    default=False,
                    help='show images, chessboard detections, etc.')
    return vars(ap.parse_args())


def main():
    args = parse_args()
    verbose = args['verbose']
    
    # CHESSBOARD DETECTION
    imgpoints, objpoints, images = chessboard_detection_from_memory(IMAGE_FOLDER, verbose, scale_factor=0.5)
    
    # CAMERA CALIBRATION
    ret, intrinsic_matrix, dist_coeffs, rvecs, t_target2cam = cv2.calibrateCamera(
        objpoints, imgpoints, images[0].shape[::-1], None, None)
    R_target2cam = [cv2.Rodrigues(rvec)[0] for rvec in rvecs]

    # EYE IN HAND CALIBRATION
    R_cam2gripper, t_cam2gripper = eye_in_hand_calibration(R_target2cam, t_target2cam)
    
    # SAVE CAMERA CALIBRATION
    save_camera_calibration(intrinsic_matrix, dist_coeffs, R_cam2gripper, t_cam2gripper)
    
    if verbose:
        print('### Camera Calibration Results ### \n')
        print(f'Camera Matrix: \n {intrinsic_matrix} \n')
        print(f'Distortion Coefficients: \n {dist_coeffs} \n')
        print(f'R_cam2gripper: \n {R_cam2gripper} \n')
        print(f't_cam2gripper: \n {t_cam2gripper} \n')


if __name__ == '__main__':
    main()
    