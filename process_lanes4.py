import numpy as np
import cv2

#The unwarp/straighten triangle
src=np.float32([[578, 460], [700,460], [1036, 678], [261, 678]])

def undistort(img):
    undis = cv2.undistort(img, mtx, dist, None, mtx)
    return undis

def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    
    # Apply the following steps to img
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    dx=0
    dy=0
    if orient == 'x':
        dx=1
    else:
        dy=1
        
    # 3) Take the absolute value of the derivative or gradient
    sobelxy = cv2.Sobel(gray, cv2.CV_64F, dx, dy)
    
    #important to look at only the absolute values (so even -ves become +ves)
    abs_sobel = np.absolute(sobelxy)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude 
            # is > thresh_min and < thresh_max
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    
    return sxbinary

# Define a function that thresholds the S-channel of HLS
# Use exclusive lower bound (>) and inclusive upper (<=)
def hls_select(img, thresh=(0, 255), h_thresh=(15,100)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # 2) Apply a threshold to the S channel
    S=hls[:,:,2]
    binary_output = np.zeros_like(S)
    binary_output[(S > thresh[0]) & (S <= thresh[1])] = 1
    # 3) Return a binary image of threshold result
    return binary_output

def combined_thresh(image):
   
    grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    
    hls_binary = hls_select(image, thresh=(170, 255))
    combined = np.zeros_like(hls_binary)
    combined[(hls_binary == 1) | (grad_binary == 1)] = 1
    return combined


def warp_perspective(img, src):
    offset=10 # to shift the board  up so that only the interested portion is shown
    img_shape = (img.shape[1], img.shape[0])

    lx=src[3][0]
    ly=offset
    rx=src[2][0]
    ry=ly
    brx=rx
    bry=img_shape[1]
    blx=lx
    bly=bry
    dst = np.float32([[lx,ly],[rx,ry],[brx,bry],[blx,bly]])
    #print('dst:', dst)
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, img_shape, flags=cv2.INTER_LINEAR) 
    return warped, M, Minv

ym_per_pix = 30/720 # meters per pixel in y dimension
xm_per_pix = 3.7/700 # meters per pixel in x dimension

#No. of frames to average lines over
N=10

#allowed diff in lanes between current frame and avg
max_lane_diff = 0.5

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        
        
        
#We keep these two global variables
left_line= Line()
right_line = Line()

def avg(l):
    return sum(l, 0.0) / len(l)

#Adds the most recent detection to averages
def compute_lane_avg(ploty, left_fitx, right_fitx):
    if len(left_line.recent_xfitted)==N:
        #remove the first one
        del left_line.recent_xfitted[0]
        del right_line.recent_xfitted[0]
    
    left_line.recent_xfitted.append(left_fitx)
    right_line.recent_xfitted.append(right_fitx)
    
    #get the avg
    left_line.bestx = avg(left_line.recent_xfitted)
    right_line.bestx = avg(right_line.recent_xfitted)
    
    #Now fit the averages
    left_line.best_fit = np.polyfit(ploty, left_line.bestx, 2)
    right_line.best_fit = np.polyfit(ploty, right_line.bestx, 2)
    
    #debug for doing sanity test on averaged lanes (as it uses he current_fit var)
    left_line.current_fit = left_line.best_fit
    right_line.current_fit = right_line.best_fit
    
#A variant in which we compute the weighted avg
# give the best fit a weight of N, and the current fit the weight of 1
def weighted_avg(ploty, left_fitx, right_fitx):
    
    #get the avg
    left_line.bestx = (left_line.bestx * N + left_fitx)/(N+1)
    right_line.bestx = (right_line.bestx * N + right_fitx)/(N+1)
    
    #Now fit the averages
    left_line.best_fit = np.polyfit(ploty, left_line.bestx, 2)
    right_line.best_fit = np.polyfit(ploty, right_line.bestx, 2)
    
#check if the newly detected lane in this frame is close enough 
#to the avg over N frames
def compare_lane_with_avg(left_fitx, right_fitx, ymax):
    #top left point
    if abs(left_fitx[0] - left_line.bestx[0])* xm_per_pix > max_lane_diff:
        return False
    
    #top right point
    if abs(right_fitx[0] - right_line.bestx[0])* xm_per_pix > max_lane_diff:
        return False
    
    #bottom left point
    if abs(left_fitx[ymax] - left_line.bestx[ymax])* xm_per_pix > max_lane_diff:
        return False
    
    #bottom right point
    if abs(right_fitx[ymax] - right_line.bestx[ymax])* xm_per_pix > max_lane_diff:
        return False
    
    return True


#should be called before each video processing, to clear out the pre-existing lane lines data
def reset_lane_lines():
    global left_line, right_line
    left_line=Line()
    right_line=Line()


def curverad(A, B, y):
    curverad = ((1 + (2*A*y + B)**2)**1.5) / np.absolute(2*A)
    return round(curverad, 2)

def radius_curvature(y_eval, nonzerox, nonzeroy, lane_inds):
    # Define conversions in x and y from pixels space to meters
   

    xx = nonzerox[lane_inds]
    yy = nonzeroy[lane_inds] 

    # Fit new polynomials to x,y in world space
    fit_cr = np.polyfit(yy*ym_per_pix, xx*xm_per_pix, 2)

    #Calculate the new radius of curvature
    rr = curverad(fit_cr[0], fit_cr[1], y_eval*ym_per_pix)
    return rr

LANE_MIN_WIDTH=3 # As per US regulations
LANE_MAX_WIDTH=6 #general intiuitive value
def sanity_check(left_l, right_l, yshape):
    
    if left_l.radius_of_curvature < 1000:
        #Radius of curvature diff for curvy lines should be within range
        # Note: We can't do the same for straight lines as they have high values and diff
        #       may not reflect reality
        diff = abs(left_l.radius_of_curvature - right_l.radius_of_curvature)
        if diff > 300:
            #print('radius of curvature diff: ', diff)
            return False
    
    # we check if lines are parallel enough
    xl0 = get_xval_for_quad_fn(left_l.current_fit, 0)
    xr0 = get_xval_for_quad_fn(right_l.current_fit, 0)
    top_diff_x = (xr0-xl0) * xm_per_pix
    xln = get_xval_for_quad_fn(left_l.current_fit, yshape-1)
    xrn = get_xval_for_quad_fn(right_l.current_fit, yshape-1)
    bot_diff_x = (xrn-xln) * xm_per_pix
    #print('top_diff_x: ', top_diff_x)
    #print('bot_diff_x: ', bot_diff_x)
    if abs(top_diff_x - bot_diff_x) > 1:
        #if top lane width and bottom lane width are greater than 1 meter
        # its not a good detection
        #print('diff b/w top lane width and bot lane width: ', abs(top_diff_x - bot_diff_x))
        return False


    #lane width should be withing some bounds
    # greater than 3 meters and less than 5 meters
    # US regulations need a minimum lane width of 3.7 meters
    if bot_diff_x < LANE_MIN_WIDTH or bot_diff_x > LANE_MAX_WIDTH:
        #print('bot_diff_x: ', bot_diff_x)
        return False

    if top_diff_x < LANE_MIN_WIDTH or top_diff_x > LANE_MAX_WIDTH:
        #print('bot_diff_x: ', bot_diff_x)
        return False
    
    
    return True

def poly_fit_lane_lines(binary_warped):
    #print('debug:', binary_warped)
    # Assuming you have created a warped binary image called "binary_warped"
    # Take a histogram of the bottom half of the image
    histogram = np.sum(binary_warped[int(binary_warped.shape[0]/2):,:], axis=0)
    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int(binary_warped.shape[0]/nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    # Set the width of the windows +/- margin
    margin = 100
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    
    ll = Line()
    ll.current_fit=left_fit
    rl = Line()
    rl.current_fit=right_fit
    
    #calculate radius of curvature of each lane
    y_eval = binary_warped.shape[0]-1
    ll.radius_of_curvature = radius_curvature(y_eval, nonzerox, nonzeroy, left_lane_inds)
    rl.radius_of_curvature = radius_curvature(y_eval, nonzerox, nonzeroy, right_lane_inds)
    
    if sanity_check(ll, rl, y_eval):
        ll.detected=True
        rl.detected=True
        
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

    return out_img, ll, rl

# x = f(y) = A * y^2 + B * y + C
def get_xval_for_quad_fn(fn, y):
    x = fn[0]*y**2 + fn[1]*y + fn[2]
    return x


def search_using_prev_frame_lane(binary_warped, left_fit, right_fit):
    # Assume you now have a new warped binary image 
    # from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin))) 
    right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))  

    
    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    
    if len(leftx) == 0 or len(rightx)==0:
        #no lane found return empty 
        return Line(), Line()
    
    
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # Generate x and y values for plotting
    ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0] )
    left_fitx = get_xval_for_quad_fn(left_fit, ploty)
    right_fitx = get_xval_for_quad_fn(right_fit, ploty)
    
    ll = Line()
    ll.current_fit=left_fit
    rl = Line()
    rl.current_fit=right_fit
    
    #calculate radius of curvature of each lane
    y_eval = binary_warped.shape[0]-1
    ll.radius_of_curvature = radius_curvature(y_eval, nonzerox, nonzeroy, left_lane_inds)
    rl.radius_of_curvature = radius_curvature(y_eval, nonzerox, nonzeroy, right_lane_inds)
    
    if sanity_check(ll, rl, y_eval):
        ll.detected=True
        rl.detected=True
   
    return ll, rl

failed_detection_count=0

#in this function we create the pipeline 
#for each image processing
# we try to use previous frame detection 
#only if successive frames have failed we fall back to fresh sliding window detection
# Also: we use radius of curvature to filter out bad frames (for curved lines)
# Or use the distance between lines for straight lanes (radius of curvature falls there)
def process_lanes(image):
    combined_bin = combined_thresh(image)
    binary_warped, _, Minv =warp_perspective(combined_bin, src)

    global left_line, right_line, failed_detection_count
    frame_detected_msg = "No new frame detected"
    
    if left_line.detected & right_line.detected:
        #print('left before passing:', left_line.current_fit)
        #print('right before passing:', right_line.current_fit)
        ll, rl = search_using_prev_frame_lane(binary_warped, left_line.best_fit, right_line.best_fit)
        
        if ll.detected and rl.detected:
            # Generate x and y values for plotting
            yshape = binary_warped.shape[0]
            ploty = np.linspace(0, yshape-1, yshape )
            left_fitx = get_xval_for_quad_fn(ll.current_fit, ploty)
            right_fitx = get_xval_for_quad_fn(rl.current_fit, ploty)
            
            #compare the recent detection lane distance to the global one
            #if compare_lane_with_avg(left_fitx, right_fitx, yshape-1):
               #Add to the global averages
                #weighted_avg(ploty, left_fitx, right_fitx)
            frame_detected_msg = "Frame detected using prev lane lines"
            compute_lane_avg(ploty, left_fitx, right_fitx)
            failed_detection_count=0
            #else:
            #    failed_detection_count+=1
        else:
            failed_detection_count+=1
    
    if failed_detection_count > 30:
                failed_detection_count=0
                left_line.detected = False
                right_line.detected = False
                
    
    if not left_line.detected or not right_line.detected:
        #coming here means that there are too many failures and so we restart detection 
        #from scratch using sliding windows from histogram peaks
        _, ll, rl = poly_fit_lane_lines(binary_warped)
        
        if left_line.bestx == None or (ll.detected and rl.detected):
            # Generate x and y values for plotting
            yshape = binary_warped.shape[0]
            ploty = np.linspace(0, yshape-1, yshape )
            left_fitx = get_xval_for_quad_fn(ll.current_fit, ploty)
            right_fitx = get_xval_for_quad_fn(rl.current_fit, ploty)

            if left_line.bestx == None:
                left_line.bestx = left_fitx
                right_line.bestx = right_fitx
                left_line.best_fit=ll.current_fit
                right_line.best_fit=rl.current_fit
                left_line.radius_of_curvature = ll.radius_of_curvature
                right_line.radius_of_curvature = rl.radius_of_curvature
                left_line.current_fit = ll.current_fit
                right_line.current_fit = rl.current_fit

                left_line.detected=True
                right_line.detected=True
            else:
                frame_detected_msg = "frame detected using sliding windows"
                compute_lane_avg(ploty, left_fitx, right_fitx)
        #else:
        #    print('sanity checked failed for: ', ll, rl)

        

    # Generate x and y values for plotting
    yshape = binary_warped.shape[0]
    xshape = binary_warped.shape[1]
    ploty = np.linspace(0, yshape-1, yshape )
    left_fitx = get_xval_for_quad_fn(left_line.best_fit, ploty)
    right_fitx = get_xval_for_quad_fn(right_line.best_fit, ploty)
    
    #debug 
    #top_leftx = round(left_fitx[0], 2)
    #top_rightx = round(right_fitx[0], 2)
    top_leftx = round(get_xval_for_quad_fn(left_line.best_fit, 0), 2)
    top_rightx = round(get_xval_for_quad_fn(right_line.best_fit, 0), 2)
    sanity_check_res = sanity_check(left_line, right_line, yshape)

    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)
    
    #calculate (and display) the avg. radius of curvature
    left_avg_r = curverad(left_line.best_fit[0], left_line.best_fit[1], (yshape-1)*ym_per_pix)
    right_avg_r = curverad(right_line.best_fit[0], right_line.best_fit[1], (yshape-1)*ym_per_pix)
    avg_r = int((left_avg_r+right_avg_r)/2)
    print_text(result, "Radius of curvature: " + str(avg_r) + " m")
    
    #display the car dist from lane center msg
    disp, disp_msg = cal_car_from_center(left_fitx, right_fitx, yshape, xshape)
    print_text(result, disp_msg, 1)
    
    #debug printing
    print_text(result, "Debug: Top left x: " + str(top_leftx) + " , Top right x: " + str(top_rightx), 2)
    print_text(result, frame_detected_msg, 3)
    #result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return result



#utility fn to print text in an image
def print_text(img, text, line_num=0):
    x = 50 
    y = 50 + line_num*40
    cv2.putText(img,text, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), thickness=2)
    
def cal_car_from_center(left_fitx, right_fitx, yshape, xshape):
    #calculate car shift from lane center
    #assuming camera is in car center
    left_lane_xbottom = left_fitx[yshape-1]
    right_lane_xbottom = right_fitx[yshape-1]
    lane_center_x = int((left_lane_xbottom + right_lane_xbottom)/2)
    car_center_x=int(xshape/2)
    car_disp = round((car_center_x - lane_center_x)*xm_per_pix, 2)
    disp_msg = 'Car is ' + str(abs(car_disp))
    if car_disp <= 0:
        disp_msg += ' m left of lane center'
    else:
        disp_msg += ' m right of lane center'
    return car_disp, disp_msg
