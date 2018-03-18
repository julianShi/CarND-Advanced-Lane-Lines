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
        src = np.float32([[513,479], [767,479], [1237,700],[43,700]])
        dst = np.float32([[offset, offset], [self.img_size[0]-offset, offset], 
                                     [self.img_size[0]-offset, self.img_size[1]-offset], 
                                     [offset, self.img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        self.perspectiveTransform = cv2.getPerspectiveTransform(src, dst)
        self.perspectiveTransform_inverse = cv2.getPerspectiveTransform(dst, src)

        self.image_previous=None
        JJ,_=np.meshgrid(np.arange(self.img_size[0]),np.arange(self.img_size[1]))
        margin=100
        # self.search_mask_left=numpy.full(self.img_size[1],self.img_size[0],False,dtype=bool)
        # self.search_mask_right=numpy.full(self.img_size[1],self.img_size[0],False,dtype=bool)
        # self.search_mask_left[(JJ>200-margin) & (JJ<200+margin)]=True
        # self.search_mask_right[(JJ>1000-margin) & (JJ<1000+margin)]=True
        self.search_mask_left=(JJ>200-margin) & (JJ<200+margin)
        self.search_mask_right=(JJ>1080-margin) & (JJ<1080+margin)

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
        I,J=np.where(image>0)
        result =random.sample(zip(I,J),self.number_sample)
        return zip(*result)[0], zip(*result)[1] 

    def _polyfit(self,I,J):
        # poly_coeffient = np.polyfit(I,J,2)
        poly_coeffient, res, _, _, _ = np.polyfit(I,J,2,full=True)
        print(res)
        if (res>100):
            raise Exception('Warning: the lanes are blurry.')
        Y = np.array(range(self.img_size[1]))
        # X = poly_coeffient[0]*(Y**2) + poly_coeffient[1]*Y + poly_coeffient[2]
        X = np.polyval(poly_coeffient,Y)
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

    def polyfill_image_merge(self,image_warp):
        ym_per_pix = 60./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/900 # meters per pixel in x dimension
        def _polyfit(I,J):
            # poly_coeffient = np.polyfit(I,J,2)
            poly_coeffient, res, _, _, _ = np.polyfit(I,J,2,full=True)
            res=res/len(I)
            print(res)
            if (res>10000):
                raise Exception('Warning: the lanes are blurry.')
            return poly_coeffient
        image_warp_left=np.array(image_warp)
        image_warp_left[self.search_mask_left==False]=0
        image_warp_right=np.array(image_warp)
        plt.imshow(image_warp_right)
        plt.show()
        image_warp_right[self.search_mask_right==False]=0
        I_left,J_left = self._cv_sample(image_warp_left)
        I_right,J_right = self._cv_sample(image_warp_right)
        poly_coeffient_left = _polyfit(I_left,J_left)
        poly_coeffient_right = _polyfit(I_right,J_right)
        poly_constant_left=np.polyval(poly_coeffient_left,self.img_size[1]-1)
        poly_constant_right=np.polyval(poly_coeffient_right,self.img_size[1]-1)
        poly_coeffient = _polyfit(I_left+I_right,np.hstack([J_left-poly_constant_left,J_right-poly_constant_right]))

        position=np.floor( ( poly_constant_left+poly_constant_left)/2 )* xm_per_pix
        curverad = ((1 + (2*poly_coeffient[0]*self.img_size[1]*ym_per_pix + poly_coeffient[1])**2)**1.5) / np.absolute(2*poly_coeffient[0])
        print('-- poly_constant_left')
        print(poly_constant_left)
        print('-- poly_constant_right')
        print(poly_constant_right)
        print('-- poly_coeffient')
        print(poly_coeffient)
        # Fill the road between the two lanes 
        road_binary = np.ones_like(image_warp)
        JJ,II=np.meshgrid(np.arange(self.img_size[0]),np.arange(self.img_size[1]))
        road_binary[(poly_coeffient[0]*II**2+poly_coeffient[1]*II+poly_coeffient[2]+poly_constant_left>JJ )]=0
        road_binary[(poly_coeffient[0]*II**2+poly_coeffient[1]*II+poly_coeffient[2]+poly_constant_right<JJ )]=0
        return road_binary

    def curvature(self,image_warp):
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 60./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/900 # meters per pixel in x dimension

        left_I,left_J = self._cv_sample(image_warp[:,:self.img_size[0]/2])
        right_I, right_J = self._cv_sample(image_warp[:,self.img_size[0]/2:])
        leftx, ploty = self._polyfit(left_I,left_J)
        rightx, ploty = self._polyfit(right_I,right_J)
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
        # Calculate the new radii of curvature
        y_eval=self.img_size[1]
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        # Now our radius of curvature is in meters
        print('left_curverad: '+str(left_curverad)+' m')
        print('right_curverad: '+str(right_curverad)+' m')
        position = (self.img_size[0]/2 - (rightx[-1]+leftx[-1]+self.img_size[0]/2)/2) * xm_per_pix
        print('car position: '+str(position)+' m')
        # Example values: 632.1 m    626.2 m

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
        self.number_sample=40
        # return the found lane on the original image
        image_undistort = self.undistort(image_original)
        # Convert RGB to the saturation channel 
        image = cv2.cvtColor(image_undistort,cv2.COLOR_RGB2HLS)[:,:,2]
        # Find the edges 
        image = self.abs_sobel_thresh(image)
        # Convert to the bird-eye view
        image_warp = self.warp(image)
        # self.curvature(image_warp)
        road_binary = self.polyfill_image_merge(image_warp)
        # unwarp the retrieved lane
        road_binary_unwarp = self.unwarp(road_binary)
        # Stack the two image layers
        road_unwarp_stack = np.dstack([road_binary_unwarp*100,road_binary_unwarp*130,road_binary_unwarp*30])
        result = cv2.addWeighted(image_undistort, 1, road_unwarp_stack, 0.5, 0)
        return result

    def lane_in_image_delay(self,image_original):
        # three-channel image
        try:
            # return the found lane on the original image
            image_undistort = self.undistort(image_original)
            # Convert RGB to the saturation channel 
            image = cv2.cvtColor(image_undistort,cv2.COLOR_RGB2HLS)[:,:,2]
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
            self.image_previous=result
        except Exception as error_message:
            print(error_message)
        else:
            print('-- Invalid frame detected. ')
        return self.image_previous

    def lane_in_video(self,video_file):
        from moviepy.editor import VideoFileClip
        self.left_I_deque=collections.deque([range(self.number_sample)]*10)
        self.left_J_deque=collections.deque([[190]*self.number_sample]*10)
        self.right_I_deque=collections.deque([range(self.number_sample)]*10)
        self.right_J_deque=collections.deque([[450]*self.number_sample]*10)
        clip1 = VideoFileClip(video_file)
        self.image_previous=clip1.make_frame(0)
        self.img_size=(self.image_previous.shape[1],self.image_previous.shape[0])
        white_clip = clip1.fl_image(self.lane_in_image)
        white_clip.write_videofile('output/'+video_file.split('/')[-1], audio=False)

