import glob
import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
import collections 
import skimage.measure

class LanePipeline():    
    def __init__(self):
        # Read in the saved objpoints and imgpoints
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        objpoints = dist_pickle["objpoints"]
        imgpoints = dist_pickle["imgpoints"]
        self.img_size = (1280,720)
        self.number_sample=200
        self.left_I_deque=collections.deque()
        self.left_J_deque=collections.deque()
        self.right_I_deque=collections.deque()
        self.right_J_deque=collections.deque()
        ret, self.mtx, self.dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, self.img_size,None,None)

        offset = 0 # offset for dst points
    #     src = np.float32([[556,463], [756,464], [1117,597],[296,597]])
        # src = np.float32([[553,459], [727,459], [1237,700],[43,700]])
        src = np.float32([[513,479], [767,479], [1237,700],[43,700]])
        dst = np.float32([[offset, offset], [self.img_size[0]-offset, offset], 
                                     [self.img_size[0]-offset, self.img_size[1]-offset], 
                                     [offset, self.img_size[1]-offset]])
        # Given src and dst points, calculate the perspective transform matrix
        self.perspectiveTransform = cv2.getPerspectiveTransform(src, dst)
        self.perspectiveTransform_inverse = cv2.getPerspectiveTransform(dst, src)

        self.previous_image = None
        self.position=0
        self.curverad_left=0
        self.curverad_right=0
        self.coeffient_left=None
        self.coeffient_right=None


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
        result =random.sample(zip(I,J),min(len(I),self.number_sample))
        return zip(*result)[0], zip(*result)[1] 

    def _polyfit(self,I,J):
        poly_coeffient = np.polyfit(I,J,2)
        Y = np.array(range(self.img_size[1]))
        # X = poly_coeffient[0]*(Y**2) + poly_coeffient[1]*Y + poly_coeffient[2]
        X = np.polyval(poly_coeffient,Y)
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

    def update_deque(self,image_warp):
        left_I,left_J = self._cv_sample(image_warp[:,:self.img_size[0]/2])
        right_I, right_J = self._cv_sample(image_warp[:,self.img_size[0]/2:])
        deque_length=10
        if(len(left_I)>10):
            self.left_I_deque.append(left_I); 
            self.left_J_deque.append(left_J); 
            if ( len(self.left_I_deque) > deque_length/2):
                left_J_val = np.polyval(self.coeffient_left,left_I)
                if(((left_J_val-left_J)**2).mean()>10000):
                    # Evaluate the smaples of this frame. Discard largely scattered samples
                    self.left_I_deque.pop()
                    self.left_J_deque.pop()        

        if (len(self.left_I_deque)>deque_length):
            self.left_I_deque.popleft()
            self.left_J_deque.popleft()        

        if(len(right_I)>10):
            self.right_I_deque.append(right_I); 
            self.right_J_deque.append(right_J); 
            if ( len(self.right_I_deque) > deque_length/2):
                right_J_val = np.polyval(self.coeffient_right,right_I)
                if(((right_J_val-right_J)**2).mean()>10000):
                    # Evaluate the smaples of this frame. Discard largely scattered samples
                    self.right_I_deque.pop()
                    self.right_J_deque.pop()        

        if (len(self.right_I_deque)>deque_length):
            self.right_I_deque.popleft()
            self.right_J_deque.popleft()

    def polyfill_video(self,image_warp):
        self.update_deque(image_warp)
        left_I,left_J = np.array(self.left_I_deque).reshape(-1), np.array(self.left_J_deque).reshape(-1)
        right_I,right_J = np.array(self.right_I_deque).reshape(-1), np.array(self.right_J_deque).reshape(-1)        
        coeffient_left, res_left, _, _, _ = np.polyfit(left_I,left_J,2,full=True)
        coeffient_right, res_right, _, _, _ = np.polyfit(right_I,right_J,2,full=True)
        self.coeffient_left=coeffient_left
        self.coeffient_right=coeffient_right
        
        # Fill the road between the two lanes 
        road_binary = np.ones((self.img_size[1],self.img_size[0]),dtype=np.uint8)
        JJ,II=np.meshgrid(np.arange(self.img_size[0]),np.arange(self.img_size[1]))
        road_binary[(coeffient_left[0]*II**2+coeffient_left[1]*II+coeffient_left[2]>JJ )]=0
        road_binary[(coeffient_right[0]*II**2+coeffient_right[1]*II+coeffient_right[2]+self.img_size[0]/2<JJ )]=0

        # Calculate curvature
        ym_per_pix = 60./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/900 # meters per pixel in x dimension

        self.position=np.floor( (np.polyval(coeffient_left,self.img_size[0]-1)+np.polyval(coeffient_right,self.img_size[0]-1))/2 - self.img_size[0]/4)* xm_per_pix
        self.curverad_left = ((1 + (2*coeffient_left[0]*self.img_size[1]*ym_per_pix + coeffient_left[1])**2)**1.5) / np.absolute(2*coeffient_left[0])
        self.curverad_right = ((1 + (2*coeffient_right[0]*self.img_size[1]*ym_per_pix + coeffient_right[1])**2)**1.5) / np.absolute(2*coeffient_right[0])

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
        self.curvature(image_warp)
        road_binary = self.polyfill_image(image_warp)
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
        # Convert RGB to the saturation channel 
        image_saturation = cv2.cvtColor(image_undistort,cv2.COLOR_RGB2HLS)[:,:,1]/4.+cv2.cvtColor(image_undistort,cv2.COLOR_RGB2HLS)[:,:,2]*3./4.
        # Find the edges 
        image_edge = self.abs_sobel_thresh(image_saturation,sobel_kernel=5, mag_thresh=(50, 255)) # binary
        # Convert to the bird-eye view
        image_warp = self.warp(image_edge)
        try:
            road_binary = self.polyfill_video(image_warp)
            self.previous_image=road_binary
        except Exception as error_message:
            print(error_message)
            road_binary=self.previous_image
        # unwarp the retrieved lane
        road_binary_unwarp = self.unwarp(road_binary)
        # Stack the two image layers
        road_unwarp_stack = np.dstack([road_binary_unwarp*100,road_binary_unwarp*130,road_binary_unwarp*30])
        # image_undistort = np.array( image_undistort, dtype=np.float64)
        result = cv2.addWeighted(image_undistort, 1.0, road_unwarp_stack, 0.5, 0)

        # Image Stitching
        cv2.putText(result,'Curvature Left [m]: %.3f '%self.curverad_left,(100,100),1,3,(255,255,255),thickness=5)
        cv2.putText(result,'Curvature Right [m]: %.3f '%self.curverad_right,(100,150),1,3,(255,255,255),thickness=5)
        cv2.putText(result,'Deviation [m]: %.3f '%self.position,(100,200),1,3,(255,255,255),thickness=5)

        image_edge = skimage.measure.block_reduce(image_edge, (2,2), np.max)
        image_edge=np.dstack([image_edge*255,image_edge*255,image_edge*255])
        
        image_bird_eye = skimage.measure.block_reduce(image_warp, (2,2), np.max)
        image_bird_eye = np.dstack([image_bird_eye*200,image_bird_eye*100,image_bird_eye*255])
        
        image_sample = np.zeros((self.img_size[1],self.img_size[0]),dtype=np.uint8)
        image_sample[[np.array(self.left_I_deque).reshape(-1)],[np.array(self.left_J_deque).reshape(-1)]]=1
        image_sample[[np.array(self.right_I_deque).reshape(-1)],[np.array(self.right_J_deque).reshape(-1)+self.img_size[0]/2]]=1
        image_sample = skimage.measure.block_reduce(image_sample, (2,2), np.max)
        image_sample = np.dstack([image_sample*255,image_sample*255,image_sample*255])

        image_polyfit = skimage.measure.block_reduce(road_binary, (2,2), np.max)
        image_polyfit=np.dstack([image_polyfit*155,image_polyfit*155,image_polyfit*10])
        image_bird_eye = cv2.addWeighted(image_sample,1, image_bird_eye,0.5, 0)
        image_bird_eye = cv2.addWeighted(image_polyfit, 0.4, image_bird_eye,1, 0)
        image_hstack = np.hstack([image_edge,image_bird_eye])
        return np.vstack([image_hstack,result])

        return result

    def lane_in_video(self,video_file):
        from moviepy.editor import VideoFileClip
        self.left_I_deque=collections.deque()
        self.left_J_deque=collections.deque()
        self.right_I_deque=collections.deque()
        self.right_J_deque=collections.deque()
        clip1 = VideoFileClip(video_file)
        # Record
        self.image_previous=clip1.make_frame(0)
        self.img_size=(self.image_previous.shape[1],self.image_previous.shape[0])
        # Core function to transfer frames
        white_clip = clip1.fl_image(self.lane_in_image_delay)
        # Save
        white_clip.write_videofile('output/subclip_'+video_file.split('/')[-1], audio=False)



