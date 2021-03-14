import cv2
import numpy as np

#Load YOLO Algorithm
net=cv2.dnn.readNet("yolov3-custom_best.weights","yolov3-custom.cfg")

#To load all objects that have to be detected
classes=[]
with open("obj.names","r") as f:
    read=f.readlines()
for i in range(len(read)):
    classes.append(read[i].strip("\n"))

#Defining layer names
layer_names=net.getLayerNames()
output_layers=[]
for i in net.getUnconnectedOutLayers():
    output_layers.append(layer_names[i[0]-1])

################################################################
frameWidth= 640                     # DISPLAY WIDTH
frameHeight = 480                  # DISPLAY HEIGHT
color= (255,0,255)
#################################################################

cap = cv2.VideoCapture(0)
cap.set(3, frameWidth)
cap.set(4, frameHeight)

def empty(a):
    pass
 
# CREATE TRACKBAR
cv2.namedWindow("Result")
cv2.resizeWindow("Result",frameWidth,frameHeight+100)
cv2.createTrackbar("Min Area","Result",0,100000,empty)
cv2.createTrackbar("Brightness","Result",180,255,empty)

import threading
import time
import logging



import random

def Mark():
    while True:
        print("Marking...")
        
        cameraBrightness = cv2.getTrackbarPos("Brightness", "Result")
        cap.set(10, cameraBrightness)
        # GET CAMERA IMAGE AND CONVERT TO GRAYSCALE
        success, img = cap.read()
        #height,width,channels=img.shape
        #Extracting features to detect objects            
        
        height,width,channels=img.shape
        
        blob=cv2.dnn.blobFromImage(img,0.00392,(416,416),(0,0,0),True,crop=False)
                                                            #Inverting blue with red
                                                            #bgr->rgb

        #We need to pass the img_blob to the algorithm
        net.setInput(blob)
        outs=net.forward(output_layers)
        #print(outs)

        #Displaying informations on the screen
        class_ids=[]
        confidences=[]
        boxes=[]

        for output in outs:
            for detection in output:
                #Detecting confidence in 3 steps
                scores=detection[5:]                #1
                class_id=np.argmax(scores)          #2
                confidence =scores[class_id]        #3

                if confidence >0.5: #Means if the object is detected
                    center_x=int(detection[0]*width)
                    center_y=int(detection[1]*height)
                    w=int(detection[2]*width)
                    h=int(detection[3]*height)

                    #Drawing a rectangle
                    x=int(center_x-w/2) # top left value
                    y=int(center_y-h/2) # top left value

                    boxes.append([x,y,w,h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
                    #cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

        #Removing Double Boxes
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,0.3,0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]  # name of the objects
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)

        cv2.imshow("Result", img)
            
        if cv2.waitKey(1) == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__=="__main__":
    Mark()

