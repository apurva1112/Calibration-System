# USAGE
# python faster_facial_landmarks.py --shape-predictor shape_predictor_5_face_landmarks.dat

# import the necessary packages
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and then create the
# facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# initialize the video stream and sleep for a bit, allowing the
# camera sensor to warm up
print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
REC_X1=250
REC_Y1=100
REC_X2=450
REC_Y2=400
REC_RED=0
REC_GREEN=255
REC_BLUE=0
REC_WID=5
TEXT_RED=0
TEXT_GREEN=255
TEXT_BLUE=0
TEXT_X=10
TEXT_Y=30
TEXT_SIZE=0.7
TEXT_WIDTH=2
REC_FACE_BLUE=255
REC_FACE_GREEN=0
REC_FACE_RED=0
REC_FACE_WIDTH=2
REC_CENTER_X=350
REC_CENTER_Y=250
# vs = VideoStream(usePiCamera=True).start() # Raspberry Pi
time.sleep(2.0)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream, resize it to
    # have a maximum width of 400 pixels, and convert it to
    # grayscale
    frame = vs.read()
    #frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # detect faces in the grayscale frame
    rects = detector(gray, 0)

    # check to see if a face was detected, and if so, draw the total
    # number of faces on the frame
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)

    # loop over the face detections
    for rect in rects:
    # compute the bounding box of the face and draw it on the
        # frame
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(0, 255, 0), 1)
                
        CENTER_X=(bX+(bW/2))
        CENTER_Y=(bY+(bH/2))


        if abs(CENTER_X - REC_CENTER_X)>=abs(CENTER_Y - REC_CENTER_Y):     
            if CENTER_X<REC_CENTER_X and bX<REC_X1:
                cv2.putText(frame, "move left", (TEXT_X,TEXT_Y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)
            elif CENTER_X>REC_CENTER_X and bX+bW>REC_X2:
                cv2.putText(frame, "move right", (TEXT_X,TEXT_Y), cv2.FONT_HERSHEY_SIMPLEX,TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)
        else:        
            if CENTER_Y<REC_CENTER_Y and bY<REC_Y1:
                cv2.putText(frame, "move down", (TEXT_X,TEXT_Y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)
            elif CENTER_Y>REC_CENTER_Y and bY+bH>REC_Y2:
                cv2.putText(frame, "move up", (TEXT_X,TEXT_Y), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)
        
        area = (min(REC_X2,bX+bW)-max(REC_X1,bX))*((min(REC_Y2,bY+bH)-max(REC_Y1,bY)))
        A = bW*bH
        per = 100*area/A
        cv2.putText(frame, str(per), (TEXT_X,TEXT_Y+30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)

            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw each of them
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

            # show the frame
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
            # if the `q` key was pressed, break from the loo p
    if key == ord("q"):
        break

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
