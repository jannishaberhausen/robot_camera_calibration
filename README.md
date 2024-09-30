# Robot Camera Calibration

This repo contains a script to perform a robot-camera-calibration that calculates both the camera intrinsics (camera matrix and distortion coefficients) as well as camera extrinsics (transformation from the camera frame of reference to the robots frame of reference). Here it is assumed that the camera is mounted rigidly to the robot's endeffector. This procedure is called "eye-in-hand" calibration.

The ```robot_camera_calibration.py``` script takes the transformations from the ```robot_transforms.json``` file together with the matching images from the ```images/``` directory as input and produces the ```camera_params.yaml``` file as output. For your reference, both the file and the folder are populated with example data you can use to run the script and check its workings. However, to calibrate your own camera and robot you will have to create your own data and store it in the same format as the example data.

The repo further contains the ```aquire_calibration_data.py``` script, which outlines the necessary functions required to produce the data from your camera and robot. Write all the functions marked with ```TODO``` (fixing the **NotImplementedErrors**). The data will then be saved in the correct format for the calibration script.

## Getting Started

Clone this repo, then install its dependencies (best in a virtual environment) like so:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

You can immediately test the ```robot_camera_calibration.py``` script with the example data provided (for instructions how to run the script see **Robot Camera Calibration** section below).

## Aquire Calibration Data

You can use this script as a stepping stone for your own camera and robot. Write the necessary functions marked with ```TODO```. It can also help to save convenient robot configurations in which the camera can "see" the calibration board if you think you will have to calibrate your camera multiple times in the future. For this purpose, you can save these robot configurations in the provided ```robot_goal_configurations.json``` (here provided as a reference for a 6D robot arm). Afterwards, you can run the script like so:

```bash
python aquire_calibration_data.py -v true
```

## Robot Camera Calibration

The ```robot_camera_calibration.py``` script is the main script of the repo. It will produce the ```camera_params.yaml``` file you are after. Assuming you aquired the data from your camera and robot, you can run the script like so:

```bash
python robot_camera_calibration.py -v true
```
