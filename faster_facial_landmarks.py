
from imutils.video import VideoStream
from imutils import face_utils
import argparse
import imutils
import time
import dlib
import cv2


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
args = vars(ap.parse_args())


print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])


print("[INFO] camera sensor warming up...")
vs = VideoStream(src=0).start()
REC_X1=250
REC_Y1=100
REC_X2=450
REC_Y2=350
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
REC_FACE_WIDTH=1
REC_CENTER_X=350
REC_CENTER_Y=250
FACE_TEXT_SIZE_X=450
FACE_TEXT_SIZE_Y=20
CIRCLE_BLUE=0
CIRCLE_GREEN=0
CIRCLE_RED=255
CIRCLE_WIDTH=1
RADIUS=0.35

time.sleep(2.0)

while True:
    
    frame = vs.read()
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    
    rects = detector(gray, 0)
    cv2.rectangle(frame, (REC_X1, REC_Y1), (REC_X2, REC_Y2), (REC_BLUE, REC_GREEN, REC_RED), REC_WID)

    
    if len(rects) > 0:
        text = "{} face(s) found".format(len(rects))
        cv2.putText(frame, text, (FACE_TEXT_SIZE_X, FACE_TEXT_SIZE_Y), cv2.FONT_HERSHEY_SIMPLEX,TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)

    
    for rect in rects:
   
        (bX, bY, bW, bH) = face_utils.rect_to_bb(rect)
        cv2.rectangle(frame, (bX, bY), (bX + bW, bY + bH),(REC_FACE_BLUE, REC_FACE_GREEN, REC_FACE_RED), REC_FACE_WIDTH)
                
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
        if per>=80:
            cv2.putText(frame, "calibrated", (TEXT_X,TEXT_Y+30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)
        else:
            cv2.putText(frame, str(per), (TEXT_X,TEXT_Y+30), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SIZE, (TEXT_BLUE, TEXT_GREEN, TEXT_RED), TEXT_WIDTH)

            
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
            
        for (i, (x, y)) in enumerate(shape):
            cv2.circle(frame, (x, y), 1, (CIRCLE_BLUE, CIRCLE_GREEN, CIRCLE_RED), -1)
            cv2.putText(frame, str(i + 1), (x - 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, RADIUS, (CIRCLE_BLUE, CIRCLE_GREEN, CIRCLE_RED), CIRCLE_WIDTH)

            
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
           
    if key == ord("q"):
        break


cv2.destroyAllWindows()
vs.stop()
