## Advanced Lane Finding
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

This is the repositor of the codes to detect lanes on roads. The library in [`LanePipeline.py`](./LanePipeline.py) provids two methods in detecting lanes from images and videos respectively. Examples are as follows

```python
from LanePipeline import LanePipeline
import cv2

lanePipeline = LanePipeline()

# Detect the lane on a 720 by 1280 by 3 image
image = cv2.imread('test_images/test1.jpg')
image_output = lanePipeline.lane_in_image(image) 

# Detect lanes from video 
lanePipeline.lane_in_video('project_video.mp4')
```

* Please refer to [`writeup.md`](./writeup.md) for the checking of the project rubric points and solution images
* Please refer to [`output/project_video.mp4`](./output/project_video.mp4) for the result video. 
* Please refer to [`camera_calibration.py`](./camera_calibration.py) for calibrating the camera. The data is saved in [`wide_dist_pickle.p`](./wide_dist_pickle.p) for quick loading. 

The  [`LanePipeline.py`](./LanePipeline.py) library
---

The input image goes through the following pipeline:

* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
 
---
Comments: this project is forked from a Udacity respository. 