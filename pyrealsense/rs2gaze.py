# code by https://github.com/vardanagarwal/Proctoring-AI

import pyrealsense2 as rs
import numpy as np
import cv2
import dlib
import statistics

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

img_width = 640
img_height = 480

config.enable_stream(rs.stream.depth, img_width, img_height, rs.format.z16, 30)

if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, img_width, img_height, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

intr = rs.video_stream_profile(pipeline_profile.get_stream(rs.stream.color)).get_intrinsics()
# print("color cam intrinsic")
# print(intr)

ratio_w = img_width / intr.width
ratio_h = img_height / intr.height
intr_resize = [intr.fx*ratio_w, intr.fy*ratio_h, intr.ppx*ratio_w, intr.ppy*ratio_h]

###
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	# return the list of (x, y)-coordinates
	return coords

def eye_on_mask(mask, side):
    points = [shape[i] for i in side]
    points = np.array(points, dtype=np.int32)
    mask = cv2.fillConvexPoly(mask, points, 255)
    return mask

def contouring(thresh, mid, img, right=False):
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    try:
        cnt = max(cnts, key = cv2.contourArea)
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if right:
            cx += mid
        cv2.circle(img, (cx, cy), 4, (0, 255, 0), 1)
    except:
        pass

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

left = [36, 37, 38, 39, 40, 41]
right = [42, 43, 44, 45, 46, 47]

kernel = np.ones((9, 9), np.uint8)

pre_eye_l = [0, 0, 0]
pre_eye_r = [0, 0, 0]
# pre_eye_c = [0, 0, 0]
eye_l = [0, 0, 0]
eye_r = [0, 0, 0]
# eye_c = [0, 0, 0]

eye_lw = [0, 0, 0]
eye_rw = [0, 0, 0]
# eye_cw = [0, 0, 0]

min_dist = 0.1 # 10cm
max_dist = 3.0 # 3m

while True:
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
    else:
        resized_color_image = color_image

    #
    img = resized_color_image.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    
    pupil_l = [0, 0, 100]
    pupil_r = [0, 0, 100]
    # pupil_l_d = []
    # pupil_r_d = []
    # pupil_c_d = []
    for rect in rects: # for all detected face
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)

        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # mask = eye_on_mask(mask, left)
        # mask = eye_on_mask(mask, right)
        # mask = cv2.dilate(mask, kernel, 5)
        
        # eyes = cv2.bitwise_and(img, img, mask=mask)
        # mask = (eyes == [0, 0, 0]).all(axis=2)
        # eyes[mask] = [255, 255, 255]
        # mid = (shape[42][0] + shape[39][0]) // 2
        # eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)

        # threshold = 200 #
        # _, thresh = cv2.threshold(eyes_gray, threshold, 255, cv2.THRESH_BINARY)
        # thresh = cv2.erode(thresh, None, iterations=2) #1
        # thresh = cv2.dilate(thresh, None, iterations=4) #2
        # thresh = cv2.medianBlur(thresh, 3) #3
        # thresh = cv2.bitwise_not(thresh)
        
        # contouring(thresh[:, 0:mid], mid, img)
        # contouring(thresh[:, mid:], mid, img, True)

        # face landmark index
        # pupil_l[0] = (shape[36][0] + shape[39][0]) // 2
        # pupil_l[1] = (shape[36][1] + shape[39][1]) // 2
        # pupil_r[0] = (shape[42][0] + shape[45][0]) // 2
        # pupil_r[1] = (shape[42][1] + shape[45][1]) // 2

        # left eye detection
        eye_width = [img_width, 0]
        eye_height = [img_height, 0]
        tmp_l = [0, 0, 0]
        tmp_l_d = []
        for i in left:
            tmp_l[0] += shape[i][0]
            tmp_l[1] += shape[i][1]
            # tm_d = depth_frame.get_distance(shape[i][0], shape[i][1])
            # if tm_d > min_dist and tm_d < max_dist:
            #     pupil_l_d.append(tm_d)

            # cv2.line(img, (shape[i][0], shape[i][1]), (shape[i][0], shape[i][1]), (0, 255, 0), 2)

            eye_width[0] = min(eye_width[0], shape[i][0])
            eye_width[1] = max(eye_width[1], shape[i][0])
            eye_height[0] = min(eye_height[0], shape[i][1])
            eye_height[1] = max(eye_height[1], shape[i][1])

        # print("left eye range : " + str(eye_width) + str(eye_height))
        for i in range(eye_width[0], eye_width[1]+1, 1):
            for j in range(eye_height[0], eye_height[1]+1, 1):
                tm_d = depth_frame.get_distance(i, j)
                if tm_d > min_dist and tm_d < max_dist:
                    tmp_l_d.append(tm_d)

        # right eye detection
        eye_width = [img_width, 0]
        eye_height = [img_height, 0]
        tmp_r = [0, 0, 0]
        tmp_r_d = []
        for i in right:
            tmp_r[0] += shape[i][0]
            tmp_r[1] += shape[i][1]
            # tm_d = depth_frame.get_distance(shape[i][0], shape[i][1])
            # if tm_d > min_dist and tm_d < max_dist:
            #     pupil_r_d.append(tm_d)

            # cv2.line(img, (shape[i][0], shape[i][1]), (shape[i][0], shape[i][1]), (0, 255, 0), 2)

            eye_width[0] = min(eye_width[0], shape[i][0])
            eye_width[1] = max(eye_width[1], shape[i][0])
            eye_height[0] = min(eye_height[0], shape[i][1])
            eye_height[1] = max(eye_height[1], shape[i][1])
        
        # print("right eye range : " + str(eye_width) + str(eye_height))
        for i in range(eye_width[0], eye_width[1]+1, 1):
            for j in range(eye_height[0], eye_height[1]+1, 1):
                tm_d = depth_frame.get_distance(i, j)
                if tm_d > min_dist and tm_d < max_dist:
                    tmp_r_d.append(tm_d)

        tmp_l[0] //= 6
        tmp_l[1] //= 6
        tmp_r[0] //= 6
        tmp_r[1] //= 6

        if len(tmp_l_d) != 0:
            tmp_l[2] = round(statistics.median(tmp_l_d), 4)
        else:
            tmp_l[2] = max_dist * 2
        if len(tmp_r_d) != 0:
            tmp_r[2] = round(statistics.median(tmp_r_d), 4)
        else:
            tmp_r[2] = max_dist * 2

        if tmp_l[2] < pupil_l[2] or tmp_r[2] < pupil_r[2]:
            pupil_l = tmp_l.copy()
            pupil_r = tmp_r.copy()

        # cv2.line(img, (pupil_l[0], pupil_l[1]), (pupil_l[0], pupil_l[1]), (0, 255, 255), 2)
        # cv2.line(img, (pupil_r[0], pupil_r[1]), (pupil_r[0], pupil_r[1]), (0, 255, 255), 2)

    # print("eye ", str(pupil_l), str(pupil_r))
    if pupil_l[2] > max_dist:
        pupil_l[2] = pre_eye_l[2]
    eye_l = pupil_l.copy()
    # eye_l[2] = round(depth_frame.get_distance(eye_l[0], eye_l[1]), 4)

    if pupil_r[2] > max_dist:
        pupil_r[2] = pre_eye_r[2]
    eye_r = pupil_r.copy()
    # eye_r[2] = round(depth_frame.get_distance(eye_r[0], eye_r[1]), 4)
    
    # eye_c[0] = (eye_l[0] + eye_r[0]) // 2
    # eye_c[1] = (eye_l[1] + eye_r[1]) // 2
    # eye_c[2] = round(depth_frame.get_distance(eye_c[0], eye_c[1]), 4)

    # for i in range(-2, 2, 1):
    #     for j in range(-2, 2, 1):
    #         if eye_l[0] > 2 and eye_l[0] < img_width-3 and eye_l[1] > 2 and eye_l[1] < img_height-3:
    #             tm_d = depth_frame.get_distance(eye_l[0]+i, eye_l[1]+j)
    #             if tm_d > min_dist and tm_d < max_dist:
    #                 pupil_l_d.append(tm_d)

    #         if eye_r[0] > 2 and eye_r[0] < img_width-3 and eye_r[1] > 2 and eye_r[1] < img_height-3:
    #             tm_d = depth_frame.get_distance(eye_r[0]+i, eye_r[1]+j)
    #             if tm_d > min_dist and tm_d < max_dist:
    #                 pupil_r_d.append(tm_d)

            # if eye_c[0] > 2 and eye_c[0] < img_width-3 and eye_c[1] > 2 and eye_c[1] < img_height-3:
            #     tm_d = depth_frame.get_distance(eye_c[0]+i, eye_c[1]+j)
            #     if tm_d > min_dist and tm_d < max_dist:
            #         pupil_c_d.append(tm_d)
    
    # if len(pupil_l_d) != 0:
    #     eye_l[2] = round(statistics.median(pupil_l_d), 4)
    # else:
    #     eye_l[2] = pre_eye_l[2]
    # if len(pupil_r_d) != 0:
    #     eye_r[2] = round(statistics.median(pupil_r_d), 4)
    # else:
    #     eye_r[2] = pre_eye_r[2]
    # if len(pupil_c_d) != 0:
    #     eye_c[2] = round(statistics.median(pupil_c_d), 4)
    # else:
    #     eye_c[2] = pre_eye_c[2]

    eye_lw = eye_l.copy()
    eye_rw = eye_r.copy()
    # eye_cw = eye_c.copy()

    eye_lw[0] = round((eye_lw[0] - intr_resize[2]) / intr_resize[0] * eye_lw[2], 4)
    eye_lw[1] = round((-eye_lw[1] + intr_resize[3]) / intr_resize[1] * eye_lw[2], 4)

    eye_rw[0] = round((eye_rw[0] - intr_resize[2]) / intr_resize[0] * eye_rw[2], 4)
    eye_rw[1] = round((-eye_rw[1] + intr_resize[3]) / intr_resize[1] * eye_rw[2], 4)

    # eye_cw[0] = round((eye_cw[0] - intr_resize[2]) / intr_resize[0] * eye_cw[2], 4)
    # eye_cw[1] = round((-eye_cw[1] + intr_resize[3]) / intr_resize[1] * eye_cw[2], 4)

    cv2.putText(img, "Left   " + str(eye_l) + " " + str(eye_lw), (20, 20), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(img, "Right  " + str(eye_r) + " " + str(eye_rw), (20, 55), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
    # cv2.putText(img, "Center " + str(eye_c) + " " + str(eye_cw), (20, 90), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)

    cv2.line(img, (eye_l[0], eye_l[1]), (eye_l[0], eye_l[1]), (0, 255, 0), 3)
    cv2.line(img, (eye_r[0], eye_r[1]), (eye_r[0], eye_r[1]), (0, 255, 0), 3)

    pre_eye_l = eye_l.copy()
    pre_eye_r = eye_r.copy()
    # pre_eye_c = eye_c.copy()

    # images = np.hstack((img, depth_colormap))
    cv2.imshow("gaze tracking", img)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
pipeline.stop()
