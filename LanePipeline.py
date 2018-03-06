import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import collections 

class LanePipeline():    
    def __init__(self):
        # Read in the saved objpoints and imgpoints
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]
        self.img_size = (1280,720)
        self.number_sample=20
        self.left_I_deque=collections.deque([[]]*10)
        self.left_J_deque=collections.deque([[]]*10)
        self.right_I_deque=collections.deque([[]]*10)
        self.right_J_deque=collections.deque([[]]*10)
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size,None,None)

        offset = 0 # offset for dst points
    #     src = np.float32([[556,463], [756,464], [1117,597],[296,597]])
        src = np.float32([[553,465], [727,465], [1237,700],[43,700]])
        dst = np.float32([[offset, offset], [self.img_size[0]-offset, offset], 
                                     [self.img_size[0]-offset, self.img_size[1]-offset], 
                                     [offset, self.img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        self.perspectiveTransform = cv2.getPerspectiveTransform(src, dst)
        self.perspectiveTransform_inverse = cv2.getPerspectiveTransform(dst, src)

    def undistort(self,img):
        # Use cv2.calibrateCamera() and cv2.undistort()    
        undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
        return undist

    def warp(self,img):
        # Warp the image using OpenCV warpPerspective()
        warped = cv2.warpPerspective(img, self.perspectiveTransform, self.img_size)
        # Return the resulting image and matrix
        return warped

    def unwarp(self,img):
        # Unwarp the image using OpenCV warpPerspective()
        unwarped = cv2.warpPerspective(img, self.perspectiveTransform_inverse, self.img_size)
        # Return the resulting image and matrix
        return unwarped

    def abs_sobel_thresh(self,img, sobel_kernel=5, mag_thresh=(50, 255)):
        # Take both Sobel x and y gradients
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # Calculate the gradient magnitude
        gradmag = np.sqrt(sobelx**2 + sobely**2)
        # Rescale to 8 bit
        scale_factor = np.max(gradmag)/255 
        gradmag = (gradmag/scale_factor).astype(np.uint8) 
        # Create a binary image of ones where threshold is met, zeros otherwise
        binary_output = np.zeros_like(gradmag)
        binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1
        # Return the binary image
        return binary_output

    def _cv_sample(self,image):
        # image is binary
        I,J=[],[]
        for i in range(len(image)):
            for j in range(len(image[0])):
                if image[i,j]>0:
                    I.append(i)
                    J.append(j)
        result =random.sample(zip(I,J),self.number_sample)
        return zip(*result)[0], zip(*result)[1] 

    def _polyfit(self,I,J):
        poly_coeffient = np.polyfit(I,J,2)
        Y = np.array(range(self.img_size[1]))
        # X = poly_coeffient[0]*(Y**2) + poly_coeffient[1]*Y + poly_coeffient[2]
        X = np.int_(np.polyval(poly_coeffient,Y))
        J_eval = np.polyval(poly_coeffient,I) 
        error = ((J_eval-J)**2).mean()
        if (error>10000):
            raise Exception('Warning: the lanes are blurry.')
        return X,Y

    def polyfill_image(self,image_warp):
        left_I,left_J = self._cv_sample(image_warp[:,:self.img_size[0]/2])
        right_I, right_J = self._cv_sample(image_warp[:,self.img_size[0]/2:])
        left_fitx, temp = self._polyfit(left_I,left_J)
        right_fitx, ploty = self._polyfit(right_I,right_J)

        # Fill the road between the two lanes 
        road_binary = np.zeros_like(image_warp)
        for i in range(len(road_binary)):
            for j in range(len(road_binary[0])):
                if (j > left_fitx[i] and j < right_fitx[i]+self.img_size[0]/2 ):
                    road_binary[i,j] = 1
        return road_binary

    def polyfill_video(self,image_warp):
        left_I,left_J = self._cv_sample(image_warp[:,:self.img_size[0]/2])
        right_I, right_J = self._cv_sample(image_warp[:,self.img_size[0]/2:])
        self.left_I_deque.append(left_I); self.left_I_deque.popleft()
        self.left_J_deque.append(left_J); self.left_J_deque.popleft()
        self.right_I_deque.append(right_I); self.right_I_deque.popleft()
        self.right_J_deque.append(right_J); self.right_J_deque.popleft()
        left_fitx, temp = self._polyfit(np.array(self.left_I_deque).reshape(-1), np.array(self.left_J_deque).reshape(-1))
        right_fitx, ploty = self._polyfit(np.array(self.right_I_deque).reshape(-1), np.array(self.right_J_deque).reshape(-1))
        
        # Fill the road between the two lanes 
        road_binary = np.zeros_like(image_warp)
        # for i in range(len(road_binary)):
        #     for j in range(len(road_binary[0])):
        #         if (j > left_fitx[i] and j < right_fitx[i]+self.img_size[0]/2 ):
        #             road_binary[i,j] = 1
        left_line = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        right_line = np.array([np.flipud(np.transpose(np.vstack([right_fitx+self.img_size[0]/2, ploty])))])
        left_line_pts = np.hstack((left_line, right_line))
        cv2.fillPoly(road_binary, np.int_([left_line_pts]),1)
        return road_binary

    def lane_in_image(self,image_original):
        # three-channel image
        # return the found lane on the original image
        image_undistort = self.undistort(image_original)
        # Convert RGB to the saturation channel 
        image = cv2.cvtColor(image_undistort,cv2.COLOR_RGB2HLS)[:,:,2]
        # Find the edges 
        image = self.abs_sobel_thresh(image)
        # Convert to the bird-eye view
        image = self.warp(image)
        plt.imshow(image)
        plt.show()
        road_binary = self.polyfill_image(image)
        # unwarp the retrieved lane
        road_binary_unwarp = self.unwarp(road_binary)
        # Stack the two image layers
        road_unwarp_stack = np.dstack([road_binary_unwarp*100,road_binary_unwarp*130,road_binary_unwarp*30])
        result = cv2.addWeighted(image_undistort, 1, road_unwarp_stack, 0.5, 0)
        return result

    def lane_in_image_delay(self,image_original):
        # three-channel image
        # return the found lane on the original image
        image_undistort = self.undistort(image_original)
        try:
            # Convert RGB to the saturation channel 
            image = cv2.cvtColor(image_undistort,cv2.COLOR_RGB2HLS)[:,:,2]
            # image = cv2.cvtColor(image_undistort,cv2.COLOR_RGB2GRAY)
            # Find the edges 
            image = self.abs_sobel_thresh(image,sobel_kernel=3, mag_thresh=(50, 255)) # binary
            # Convert to the bird-eye view
            image = self.warp(image)
            road_binary = self.polyfill_video(image)
            # unwarp the retrieved lane
            road_binary_unwarp = self.unwarp(road_binary)
            # Stack the two image layers
            road_unwarp_stack = np.dstack([road_binary_unwarp*100,road_binary_unwarp*130,road_binary_unwarp*30])
            result = cv2.addWeighted(image_undistort, 1, road_unwarp_stack, 0.5, 0)
            return result
        except Exception as error_message:
            print(error_message)
            return image_undistort
        else:
            print('-- Invalid frame detected. ')
            return image_undistort

    def lane_in_video(self,video_file):
        from moviepy.editor import VideoFileClip
        self.left_I_deque=collections.deque([range(self.number_sample)]*10)
        self.left_J_deque=collections.deque([[190]*self.number_sample]*10)
        self.right_I_deque=collections.deque([range(self.number_sample)]*10)
        self.right_J_deque=collections.deque([[450]*self.number_sample]*10)
        clip1 = VideoFileClip(video_file)
        white_clip = clip1.fl_image(self.lane_in_image_delay)
        white_clip.write_videofile('output/'+video_file.split('/')[-1], audio=False)



