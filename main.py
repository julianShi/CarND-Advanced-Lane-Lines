from LanePipelineFast import LanePipeline
import matplotlib.pyplot as plt 
import cv2

lanePipeline=LanePipeline();

# Find lane from video 
lanePipeline.lane_in_video('project_video.mp4')

# # Read in an image
# img = cv2.imread('test_images/test3.jpg')
# # Find lane from image
# image_output = lanePipeline.lane_in_image(img)
