import argparse
import cv2
import json
import numpy as np
from time import sleep


CHESSBOARD_DIMS = (9, 6)
ROBOT_IP = ""   # TODO: Fill in the IP address of your robot


def init_camera():
    # TODO: Initialize your camera here
    raise NotImplementedError("Write this function for your camera. The function should return the camera object.")
    return camera


def construct_homogeneous_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def verify_pose_by_inspection(R_gripper2base, t_gripper2base):
    print(f'R_gripper2base: \n {R_gripper2base}')
    print(f't_gripper2base: \n {t_gripper2base}')
    H_gripper2base = construct_homogeneous_matrix(R_gripper2base, t_gripper2base)
    p_InGripper = np.zeros(4)
    p_InGripper[3] = 1 # homogenous point
    p_InBase = H_gripper2base @ p_InGripper
    print(f'p_inBase: \n {p_InBase}')
    v_InGripper = np.zeros(4)
    v_InGripper[0] = 1 # x-axis
    v_InBase = H_gripper2base @ v_InGripper
    print(f'v_InBase (x-axis): \n {v_InBase}')


def save_robot_pose(id, ROBOT_IP, filename='robot_transforms.json'):
    element = {}
    element['id'] = id
    
    # TODO: Get the transforms of the robot end-effector to the robot base
    
    raise NotImplementedError(
        "Complete this function for your robot. You will need to get the pose of " + \
        "the robot end-effector and then calculate R_gripper2base and t_gripper2base.")
    
    element['R_gripper2base'] = R_gripper2base.tolist()
    element['t_gripper2base'] = t_gripper2base.tolist()
    
    with open(filename, 'r') as f:
        data = json.load(f)
        
    # Append the new transform to the list (assuming the list is under a key 'robot_transforms')
    if 'robot_transforms' in data and isinstance(data['robot_transforms'], list):
        data['robot_transforms'].append(element)
    else:
        data['robot_transforms'] = [element]
        
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)
    
    
def chessboard_detactable(img, verbose, scale_factor=1.0):
    """Check if the chessboard is detectable in the image. Return True if detected, False otherwise."""
    # Downscale image for faster processing
    if scale_factor != 1.0:
        img = cv2.resize(img, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_DIMS, None)
    if ret:
        # Upscale the corner positions back to the original image size
        corners = corners / scale_factor
        
        if verbose:
            cv2.drawChessboardCorners(img, CHESSBOARD_DIMS, corners * scale_factor, ret)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        return True
    else:
        return False
        

def capture_image(camera, verbose, idx):
    # TODO: Capture an image from the camera
    raise NotImplementedError("Write this function for your camera. The function should make your camera capture an image.")

    if verbose:    
        cv2.imshow('img', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    if chessboard_detactable(image):
        print("Checkerboard detected")
        cv2.imwrite(f"images/{idx}.png", image)
        return True
    else:
        print("Checkerboard not detected. Skipping image.")
        return False
    
    
def clean_up(camera):
    # TODO: Release the camera
    raise NotImplementedError("Write this function for your camera. The function should release the camera.")
    
    
def init_robot(Robot_IP):#
    # TODO: Initialize your robot
    raise NotImplementedError("Write this function for your robot. The function should initialize the robot.")
    
    
def move_robot(goal, ROBOT_IP):
    # TODO: Move the robot to the goal configuration
    raise NotImplementedError("Write this function for your robot. The function should move the robot to the goal configuration.")
   
    
def load_configurations(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    return data['robot_goal_configurations']
        

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('-v', '--verbose',
                    default=False,
                    help='show images, chessboard detections, etc.') 
    return vars(ap.parse_args())


def main():
    args = parse_args()
    verbose = args['verbose']
    
    configurations = load_configurations('robot_goal_configurations.json')
    init_robot(ROBOT_IP)
    camera = init_camera()
    
    for conf in configurations:
        move_robot(np.array(conf['conf']), ROBOT_IP)
        sleep(2) # wait for residual motion to stop
        
        valid = capture_image(camera, verbose, conf['id'])
        if valid:
            save_robot_pose(conf['id'], ROBOT_IP)
        else:
            print(f"Skipping configuration {conf['id']}")

    clean_up(camera)
    
    
if __name__ == "__main__":
    main()
    