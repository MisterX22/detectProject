# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

# import the necessary packages
import argparse
import datetime
import imutils
import time
import cv2
import logging
import requests
import json
import base64

import sys
import dlib
from skimage import io
detector = dlib.get_frontal_face_detector()


# Settings logs information
logger = logging.getLogger('peopleDetector')
hdlr = logging.FileHandler('/tmp/peopleDetector.log')
formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
hdlr.setFormatter(formatter)
logger.addHandler(hdlr)
logger.setLevel(logging.INFO)

# Sending Data to my Garage Cloud 9 projets
headers = {'content-type': 'application/json','charset': 'utf-8'}
url = 'https://garageairquality-misterx22.c9users.io/callback/aws'
def send2Dashboard(image, cpt_body_detected, cpt_face_detected):
    image = open(image, 'rb')
    image_read = image.read()
    image_64_encode = base64.encodestring(image_read)
    data = {'NumOfBody': cpt_body_detected,'NumOfFace': cpt_face_detected,'Image': image_64_encode}
    response = requests.post(url, data=json.dumps(data), headers=headers)
    logger.info('send2Dashboard response : ' + str(response)) 
    return response

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
	camera = cv2.VideoCapture(0)
	time.sleep(0.5)

# otherwise, we are reading from a video file
else:
	camera = cv2.VideoCapture(args["video"])

width = camera.get(3)
height = camera.get(4)

# initialize the first frame in the video stream
firstFrame = None
previousFrame = None

# How many detection ?
cpt_body_detected = 0
cpt_face_detected = 0

SKIP_FRAMES = 2

count = 0
# loop over the frames of the video
while True:
	count = count + 1
	motion = 0
	#logger.info('Boucle : ' + str(count)) 
	
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	if ( (count % SKIP_FRAMES) == 0 ) :
		#logger.info('skip, boucle : ' + str(count)) 
		continue 
	#else: 
		#logger.info('go, boucle : ' + str(count)) 

	frame = imutils.resize(frame, width=320)
	#frame = cv2.flip(frame,flipCode=-1)
	#logger.info('flip done')

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		print("Quit here")
		break

	# resize the frame, convert it to grayscale, and blur it
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#frame_gray = frame
	gray = cv2.GaussianBlur(frame_gray, (21, 21), 0)

	# if the first frame is None, initialize it
	if firstFrame is None:
		firstFrame = gray
		continue
	else:
		if previousFrame is None:
			previousFrame = gray
			firstFrame = gray
			continue
		else:
			firstFrame = previousFrame

	# compute the absolute difference between the current frame and
	# first frame
	frameDelta = cv2.absdiff(firstFrame, gray)
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

	previousFrame = gray

	# dilate the thresholded image to fill in holes, then find contours
	# on thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	(_,cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# loop over the contours
	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < args["min_area"]:
			continue
		motion = motion + 1

	if ( motion > 0) :
		#logger.info('detection started, width:' + str(frame.shape[0]) + ' height:' + str(frame.shape[1]))
		#dets = detector(frame_gray, 0)
		dets = detector(frame, 0)
		#logger.info('detection done')
		nbobjs_face = len(dets)
		if ( nbobjs_face > 0):
			logger.info('face detected : ' + str(nbobjs_face)) 
			#Draw a rectangle around every found obj
			for i, d in enumerate(dets):
				cpt_face_detected = cpt_face_detected + 1
				cv2.rectangle(frame,(d.left(),d.top()),(d.right(),d.bottom()),(0,255,0),1)
				face = "/tmp/face_" + str(cpt_face_detected) + ".jpg"
				cv2.imwrite(face, frame, [int(cv2.IMWRITE_JPEG_QUALITY),80])
			#logger.info('Sending to dash board ')
			#send2Dashboard(face,cpt_body_detected,cpt_face_detected)


	# show the frame and record if the user presses a key
	cv2.imshow("Frame", frame)
	#cv2.imshow("Gray", gray)
	#cv2.imshow("Frame Gray", frame_gray)
	#cv2.imshow("Thresh", thresh)
	#cv2.imshow("Frame Delta", frameDelta)
	key = cv2.waitKey(1) & 0xFF

	# if the `q` key is pressed, break from the lop
	if key == ord("q"):
		break

# cleanup the camera and close any open windows
camera.release()
cv2.destroyAllWindows()
