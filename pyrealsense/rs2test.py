# code by https://github.com/antoinelame/GazeTracking

import pyrealsense2 as rs
import numpy as np
# import cv2

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# try:
#     while cv2.waitKey(1) < 0:
#         # Wait for a coherent pair of frames: depth and color
#         frames = pipeline.wait_for_frames()
#         depth_frame = frames.get_depth_frame()
#         color_frame = frames.get_color_frame()
#         if not depth_frame or not color_frame:
#             continue

#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())

#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

#         depth_colormap_dim = depth_colormap.shape
#         color_colormap_dim = color_image.shape

#         # If depth and color resolutions are different, resize color image to match depth image for display
#         if depth_colormap_dim != color_colormap_dim:
#             resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
#             images = np.hstack((resized_color_image, depth_colormap))
#         else:
#             images = np.hstack((color_image, depth_colormap))

#         # Show images
#         cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
#         cv2.imshow('RealSense', images)

# finally:
#     # Stop streaming
#     pipeline.stop()

import cv2
from gaze_tracking import GazeTracking

gaze = GazeTracking()
# webcam = cv2.VideoCapture(1)

pre_eye_l = [0, 0, 0]
pre_eye_r = [0, 0, 0]
eye_l = [0, 0, 0]
eye_r = [0, 0, 0]
eye_c = [0, 0, 0]

while True:
    # # We get a new frame from the webcam
    # _, frame = webcam.read()

    # Wait for a coherent pair of frames: depth and color
    frames = pipeline.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        continue

    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

    depth_colormap_dim = depth_colormap.shape
    color_colormap_dim = color_image.shape

    # If depth and color resolutions are different, resize color image to match depth image for display
    if depth_colormap_dim != color_colormap_dim:
        resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]), interpolation=cv2.INTER_AREA)
        # frame = np.hstack((resized_color_image, depth_colormap))
    else:
        resized_color_image = color_image
        # frame = np.hstack((color_image, depth_colormap))

    # We send this frame to GazeTracking to analyze it
    gaze.refresh(resized_color_image)

    frame = gaze.annotated_frame()
    text = ""

    # if gaze.is_blinking():
    #     text = "Blinking"
    # elif gaze.is_right():
    #     text = "Looking right"
    # elif gaze.is_left():
    #     text = "Looking left"
    # elif gaze.is_center():
    #     text = "Looking center"

    # cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    if left_pupil is not None:
        eye_l[0], eye_l[1] = left_pupil
    else:
        eye_l = pre_eye_l
    eye_l[2] = round(depth_frame.get_distance(eye_l[0], eye_l[1]), 4)

    if right_pupil is not None:
        eye_r[0], eye_r[1] = right_pupil
    else:
        eye_r = pre_eye_r
    eye_r[2] = round(depth_frame.get_distance(eye_r[0], eye_r[1]), 4)

    eye_c[0] = int((eye_l[0] + eye_r[0]) * 0.5)
    eye_c[1] = int((eye_l[1] + eye_r[1]) * 0.5)
    eye_c[2] = round(depth_frame.get_distance(eye_c[0], eye_c[1]), 4)

    pre_eye_l = eye_l
    pre_eye_r = eye_r

    cv2.putText(frame, "Left  :  " + str(eye_l), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)
    cv2.putText(frame, "Right : " + str(eye_r), (20, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)
    cv2.putText(frame, "Center: " + str(eye_c), (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
# webcam.release()
cv2.destroyAllWindows()
