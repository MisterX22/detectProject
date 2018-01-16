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

#Load a cascade file for detecting objs
obj_cascade_face = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')
obj_cascade = cv2.CascadeClassifier('/usr/local/share/OpenCV/haarcascades/haarcascade_fullbody.xml')

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

# initialize the first frame in the video stream
firstFrame = None
previousFrame = None

scaleFactor = 1.1

# How many detection ?
cpt_body_detected = 0
cpt_face_detected = 0
face_detection = 1
body_detection = 1

# loop over the frames of the video
while True:
	motion = 0
	# grab the current frame and initialize the occupied/unoccupied
	# text
	(grabbed, frame) = camera.read()
	frame = cv2.flip(frame,flipCode=-1)
	text = "Unoccupied"

	# if the frame could not be grabbed, then we have reached the end
	# of the video
	if not grabbed:
		print("Quit here")
		break

	# resize the frame, convert it to grayscale, and blur it
	#frame = imutils.resize(frame, width=500)
	frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		#(x1, y1, w1, h1) = cv2.boundingRect(c)
		#cv2.rectangle(frame, (x1, y1), (x1 + w1, y1 + h1), (255, 0, 0), 4)
		motion = motion + 1

	if ( motion > 0) :
		#logger.info('motion detected : ' + str(motion)) 
		text = "Occupied"
		# draw the text and timestamp on the frame
		cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
		cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
			(10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
		if ( body_detection > 0 ):
			objs = obj_cascade.detectMultiScale(frame_gray,scaleFactor,5)
			nbobjs = len(objs)
			if ( nbobjs > 0 ):
				logger.info('People detected : ' + str(nbobjs)) 
				#Draw a rectangle around every found obj
				for (x,y,w,h) in objs:
					cpt_body_detected = cpt_body_detected + 1 
					cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

				filename_body = "/tmp/body_" + str(cpt_body_detected) +".jpg"
				cv2.imwrite(filename_body, frame, [int(cv2.IMWRITE_JPEG_QUALITY),80])

				#for (x,y,w,h) in objs:
				#frame_cropped = frame[y:y+h, x:x+w]
				#frame_cropped = imutils.resize(frame_cropped, width=500)
				#gray_cropped = cv2.cvtColor(frame_cropped, cv2.COLOR_BGR2GRAY)


		if ( face_detection > 0 ):
			objs_face = obj_cascade_face.detectMultiScale(gray,scaleFactor,5)
			nbobjs_face = len(objs_face)
			if ( nbobjs_face > 0):
				logger.info('face detected : ' + str(nbobjs_face)) 
				#Draw a rectangle around every found obj
				for (x2,y2,w2,h2) in objs_face:
					cpt_face_detected = cpt_face_detected + 1
					cv2.rectangle(frame,(x2,y2),(x2+w2,y2+h2),(0,255,0),1)
					face = "/tmp/face_" + str(cpt_face_detected) + ".jpg"
					cv2.imwrite(face, frame, [int(cv2.IMWRITE_JPEG_QUALITY),80])
				logger.info('Sending to dash board ')
				send2Dashboard(face,cpt_body_detected,cpt_face_detected)


	# show the frame and record if the user presses a key
	#cv2.imshow("Security Feed", frame)
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
