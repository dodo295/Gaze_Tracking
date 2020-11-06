# Gaze_Tracking
Real time eye tracking using OpenCV and haarcascades

# Objective 

- This project uses OpenCV cascading classifiers and established databases to detect face and eye. 

- It will intentionally track the gaze of only one face. 

- From the bounding box, an algorithm that detects the group of contours closest to the center.
 
- Then, it produces an approximation of the pupil and is marked with an orange point.

# Steps

- [x] The first step is to download the required packages. Installation via pip:

      pip install opencv-python

- [x] The second step is to classify, you need a classifer. There are haarcascades available face ('haarcascade_frontalface_default.xml') and eye ('haarcascade_eye.xml') stickers that come with the OpenCV library, and you can download them from the [official github repository](https://github.com/opencv/opencv/tree/master/data/haarcascades)

3- The third step is to run code from this command:

    python Gaze Tracking.py

# Example
![grab-landing-page](https://github.com/dodo295/Gaze_Tracking/blob/main/Test.gif)
