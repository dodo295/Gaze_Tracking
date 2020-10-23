#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 21:15:44 2020

@author: Doaa
"""
import numpy as np
import math
import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,480))
# Check if camera opened successfully
if (cap.isOpened()== False):
    print("Error opening video stream or file")
# Read until video is completed
while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()    
  if ret == True:
      gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
      faces = face_cascade.detectMultiScale(gray, 1.1, 6)
      
      print('Number of faces detected:', len(faces))


      frame_with_detections = np.copy(frame)
      for (x,y,w,h) in faces:
          cv2.rectangle(frame_with_detections, (x,y), (x+w,y+h), (0,255,0), 3)
          bgr_crop = frame_with_detections[y:y+h, x:x+w] 
          orig_shape_crop = bgr_crop.shape
          gray_crop = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
          eyes = eye_cascade.detectMultiScale(gray_crop, 1.1, 6)
          if len(eyes) != 2:
              print("no eyes detected")
          resize_gray_crop = cv2.resize(gray_crop, (96, 96)) / 255.0
          """
          model = load_model('model_1_Eye.h5')
          landmarks = model.predict(resize_gray_crop.reshape(1,96,96,1))[0]
          d1= ((landmarks[0::2])* np.float32(orig_shape_crop[0])/96.0) + +np.float32(x)
          d2= ((landmarks[1::2])* np.float32(orig_shape_crop[1])/96.0) + +np.float32(y)
          cv2.circle(frame_with_detections, (d1[0],d1[1]), 1, (255,0,0), 3)
          cv2.circle(frame_with_detections, (d2[0],d2[1]), 1, (255,0,0), 3)
          
          """
      for eye in eyes:
          ex, ey, ew, eh = eye
          cv2.rectangle(bgr_crop, (ex,ey), (ex+ew,ey+eh), (0, 140, 255) , 2)
      
      right_eye = eyes[0]    
      left_eye = eyes[1]
      rex, rey, rew, reh = right_eye
      lex, ley, lew, leh = left_eye 
      right_eye = gray_crop[rey:rey + reh, rex:rex + rew]
      left_eye = gray_crop[ley:ley + leh, lex:lex + lew]
      # increase contrast in the image
      right_eye = cv2.equalizeHist(right_eye)
      left_eye = cv2.equalizeHist(left_eye)
      right_thres = cv2.inRange(right_eye, 0, 20)
      left_thres = cv2.inRange(left_eye, 0, 20)
      kernel = np.ones((3, 3), np.uint8)
      
      """processing to remove small noise:
      dilation increases the white region in the image or size of 
      foreground object increases. Normally, in cases like noise removal, 
      erosion is followed by dilation. Because, erosion removes white noises, 
      but it also shrinks our object. So we dilate it. Since noise is gone, 
      they won’t come back, but our object area increases.
      """
      right_dilation = cv2.dilate(right_thres, kernel, iterations=2)
      left_dilation = cv2.dilate(left_thres, kernel, iterations=2)
      right_erosion = cv2.erode(right_dilation, kernel, iterations=3)
      left_erosion = cv2.erode(left_dilation, kernel, iterations=3)
      
      # find contours
      right_contours, right_hierarchy = cv2.findContours(
              right_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      
      left_contours, left_hierarchy = cv2.findContours(
              left_erosion, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
      # algorithm to find pupil and reduce noise of other contours by selecting the closest to center
      right_closest_cx = None
      right_closest_cy = None
      right_closest_distance = None
      right_center = (rey + reh//2, rex + rew//2)
      left_closest_cx = None
      left_closest_cy = None
      left_closest_distance = None
      left_center = (ley + leh//2, lex + lew//2)
      if len(right_contours) >= 1:
          """cv2.moments() gives a dictionary of all moment values calculated.
          From this moments, you can extract useful data like area, centroid etc. 
          Centroid is given by the relations, 
          C_x = int(M['m10']/M['m00'])
          C_y = int(M['m01']/M['m00'])"""
          right_M = cv2.moments(right_contours[0])
          if right_M['m00'] != 0:
              right_cx = int(right_M['m10']/right_M['m00'])
              right_cy = int(right_M['m01']/right_M['m00'])
              for contour in right_contours:
                  # distance between center and potential pupil
                  right_distance = math.sqrt(
                          (right_cy - right_center[0])**2 + (right_cx - right_center[1])**2)
                  if right_closest_distance is None or right_distance < right_closest_distance:
                      right_closest_cx = right_cx
                      right_closest_cy = right_cy
                      right_closest_distance = right_distance

      if right_closest_cx is not None and right_closest_cy is not None:
          # base size of pupil to size of eye
          cv2.circle(frame_with_detections, (x + rex + right_closest_cx, y + rey + right_closest_cy+9),
                       rew//12, (0, 140, 255), -1)
      if len(left_contours) >= 1:
          left_M = cv2.moments(left_contours[0])
          if left_M['m00'] != 0:
              left_cx = int(left_M['m10']/left_M['m00'])
              left_cy = int(left_M['m01']/left_M['m00'])
              for contour in left_contours:
                  # distance between center and potential pupil
                  left_distance = math.sqrt(
                          (left_cy - left_center[0])**2 + (left_cx - left_center[1])**2)
                  if left_closest_distance is None or left_distance < left_closest_distance:
                      left_closest_cx = left_cx
                      left_closest_cy = left_cy
                      left_closest_distance = left_distance

      if left_closest_cx is not None and left_closest_cy is not None:
          # base size of pupil to size of eye
          cv2.circle(frame_with_detections, (x + lex + left_closest_cx+10, y + ley + left_closest_cy+10),
                       lew//12, (0, 140, 255), -1)
      # Display the resulting frame
      out.write(frame_with_detections)
      frame_with_detections = cv2.resize(frame_with_detections, (960, 540))
      cv2.imshow("Frame",frame_with_detections)

      # Press Q on keyboard to  exit
      if cv2.waitKey(25) & 0xFF == ord('q'):
          break

  # Break the loop
  else: 
    break

# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()

