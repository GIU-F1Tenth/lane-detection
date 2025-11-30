#!/usr/bin/env python3
"""
ROS2 Lane Detection Node

Subscribes to camera images and publishes lane detection results as Path messages:
- Left lane boundary path
- Right lane boundary path
- Lane midline/centerline path

Author: Fam Shihata
"""

from datetime import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import matplotlib.pyplot as plt
import numpy as np
from .lane_utils import visualize_boundaries_and_midline, extract_boundary_points, calculate_midline
from .lane_data import LaneDetectionResult
from . import edge_detection as edge

class Lane:
    """
    Represents a lane on a road.
    """

    def __init__(self, orig_frame, config=None):
        """
          Default constructor

        :param orig_frame: Original camera image (i.e. frame)
        :param config: LaneConfig object (optional, uses default if None)
        """
        self.orig_frame = orig_frame
        self.config = config if config is not None else LaneConfig()

        # This will hold an image with the lane lines
        self.lane_line_markings = None

        # This will hold the image after perspective transformation
        self.warped_frame = None
        self.transformation_matrix = None
        self.inv_transformation_matrix = None

        # (Width, Height) of the original video frame (or image)
        self.orig_image_size = self.orig_frame.shape[::-1][1:]

        width = self.orig_image_size[0]
        height = self.orig_image_size[1]
        self.width = width
        self.height = height

        # Four corners of the trapezoid-shaped region of interest
        self.roi_points = self.config.get_roi_points(width, height)

        # The desired corner locations of the region of interest
        # after we perform perspective transformation.
        self.desired_roi_points = self.config.get_desired_roi_points(
            width, height)

        # Histogram that shows the white pixel peaks for lane line detection
        self.histogram = None

        # Sliding window parameters
        self.no_of_windows = self.config.NUM_WINDOWS
        self.margin = int(self.config.WINDOW_MARGIN_RATIO * width)
        self.minpix = int(self.config.MIN_PIXELS_RATIO * width)

        # Best fit polynomial lines for left line and right line of the lane
        self.left_fit = None
        self.right_fit = None
        self.left_lane_inds = None
        self.right_lane_inds = None
        self.ploty = None
        self.left_fitx = None
        self.right_fitx = None
        self.leftx = None
        self.rightx = None
        self.lefty = None
        self.righty = None

        # Pixel parameters for x and y dimensions
        self.YM_PER_PIX = self.config.YM_PER_PIX
        self.XM_PER_PIX = self.config.XM_PER_PIX

        # Radii of curvature and offset
        self.left_curvem = None
        self.right_curvem = None
        self.center_offset = None

    def calculate_car_position(self, print_to_terminal=False):
        """
        Calculate the position of the car relative to the center

        :param: print_to_terminal Display data to console if True       
        :return: Offset from the center of the lane
        """
        # Assume the camera is centered in the image.
        # Get position of car in centimeters
        car_location = self.orig_frame.shape[1] / 2

        # Fine the x coordinate of the lane line bottom
        height = self.orig_frame.shape[0]
        bottom_left = self.left_fit[0]*height**2 + self.left_fit[
            1]*height + self.left_fit[2]
        bottom_right = self.right_fit[0]*height**2 + self.right_fit[
            1]*height + self.right_fit[2]

        center_lane = (bottom_right - bottom_left)/2 + bottom_left
        center_offset = (np.abs(car_location) - np.abs(
            center_lane)) * self.XM_PER_PIX * 100

        if print_to_terminal == True:
            print(str(center_offset) + 'cm')

        self.center_offset = center_offset

        return center_offset

    def calculate_curvature(self, print_to_terminal=False):
        """
        Calculate the road curvature in meters.

        :param: print_to_terminal Display data to console if True
        :return: Radii of curvature
        """
        # Set the y-value where we want to calculate the road curvature.
        # Select the maximum y-value, which is the bottom of the frame.
        y_eval = np.max(self.ploty)

        # Fit polynomial curves to the real world environment
        left_fit_cr = np.polyfit(self.lefty * self.YM_PER_PIX, self.leftx * (
            self.XM_PER_PIX), 2)
        right_fit_cr = np.polyfit(self.righty * self.YM_PER_PIX, self.rightx * (
            self.XM_PER_PIX), 2)

        # Calculate the radii of curvature
        left_curvem = ((1 + (2*left_fit_cr[0]*y_eval*self.YM_PER_PIX + left_fit_cr[
                        1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curvem = ((1 + (2*right_fit_cr[
                        0]*y_eval*self.YM_PER_PIX + right_fit_cr[
            1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

        # Display on terminal window
        if print_to_terminal == True:
            print(left_curvem, 'm', right_curvem, 'm')

        self.left_curvem = left_curvem
        self.right_curvem = right_curvem

        return left_curvem, right_curvem

    def calculate_histogram(self, frame=None, plot=True):
        """
        Calculate the image histogram to find peaks in white pixel count

        :param frame: The warped image
        :param plot: Create a plot if True
        """
        if frame is None:
            frame = self.warped_frame

        # Generate the histogram
        self.histogram = np.sum(frame[int(
            frame.shape[0]/2):, :], axis=0)

        if plot == True:

            # Draw both the image and the histogram
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 row, 1 columns
            figure.set_size_inches(10, 5)
            ax1.imshow(frame, cmap='gray')
            ax1.set_title("Warped Binary Frame")
            ax2.plot(self.histogram)
            ax2.set_title("Histogram Peaks")
            plt.show()

        return self.histogram

    def display_curvature_offset(self, frame=None, plot=False):
        """
        Display curvature and offset statistics on the image

        :param: plot Display the plot if True
        :return: Image with lane lines and curvature
        """
        image_copy = None
        if frame is None:
            image_copy = self.orig_frame.copy()
        else:
            image_copy = frame

        cv2.putText(image_copy, 'Curve Radius: '+str((
            self.left_curvem+self.right_curvem)/2)[:7]+' m', (
                self.config.TEXT_X_OFFSET,
                self.config.TEXT_Y_OFFSET_1),
            cv2.FONT_HERSHEY_SIMPLEX,
            (float(self.config.FONT_SCALE_RATIO * self.width)),
            self.config.TEXT_COLOR,
            self.config.TEXT_THICKNESS,
            cv2.LINE_AA)
        cv2.putText(image_copy, 'Center Offset: '+str(
            self.center_offset)[:7]+' cm', (
                self.config.TEXT_X_OFFSET,
                self.config.TEXT_Y_OFFSET_2),
            cv2.FONT_HERSHEY_SIMPLEX,
            (float(self.config.FONT_SCALE_RATIO * self.width)),
            self.config.TEXT_COLOR,
            self.config.TEXT_THICKNESS,
            cv2.LINE_AA)

        if plot == True:
            cv2.imshow("Image with Curvature and Offset", image_copy)

        return image_copy

    def get_lane_line_previous_window(self, left_fit, right_fit, plot=False):
        """
        Use the lane line from the previous sliding window to get the parameters
        for the polynomial line for filling in the lane line
        :param: left_fit Polynomial function of the left lane line
        :param: right_fit Polynomial function of the right lane line
        :param: plot To display an image or not
        """
        # margin is a sliding window parameter
        margin = self.margin

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store left and right lane pixel indices
        left_lane_inds = ((nonzerox > (left_fit[0]*(
            nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] - margin)) & (
            nonzerox < (left_fit[0]*(
                nonzeroy**2) + left_fit[1]*nonzeroy + left_fit[2] + margin)))
        right_lane_inds = ((nonzerox > (right_fit[0]*(
            nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] - margin)) & (
            nonzerox < (right_fit[0]*(
                nonzeroy**2) + right_fit[1]*nonzeroy + right_fit[2] + margin)))
        self.left_lane_inds = left_lane_inds
        self.right_lane_inds = right_lane_inds

        # Get the left and right lane line pixel locations
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        self.leftx = leftx
        self.rightx = rightx
        self.lefty = lefty
        self.righty = righty

        # Check if we have enough points to fit a polynomial
        if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
            # Keep using previous fit if no points detected
            return

        # Fit a second order polynomial curve to each lane line
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        self.left_fit = left_fit
        self.right_fit = right_fit

        # Create the x and y values to plot on the image
        ploty = np.linspace(
            0, self.warped_frame.shape[0]-1, self.warped_frame.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        self.ploty = ploty
        self.left_fitx = left_fitx
        self.right_fitx = right_fitx

        if plot == True:

            # Generate images to draw on
            out_img = np.dstack((self.warped_frame, self.warped_frame, (
                                 self.warped_frame)))*255
            window_img = np.zeros_like(out_img)

            # Add color to the left and right line pixels
            out_img[nonzeroy[left_lane_inds],
                    nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]
            # Create a polygon to show the search window area, and recast
            # the x and y points into a usable format for cv2.fillPoly()
            margin = self.margin
            left_line_window1 = np.array([np.transpose(np.vstack([
                                          left_fitx-margin, ploty]))])
            left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                          left_fitx+margin, ploty])))])
            left_line_pts = np.hstack((left_line_window1, left_line_window2))
            right_line_window1 = np.array([np.transpose(np.vstack([
                                           right_fitx-margin, ploty]))])
            right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([
                                           right_fitx+margin, ploty])))])
            right_line_pts = np.hstack(
                (right_line_window1, right_line_window2))

            # Draw the lane onto the warped blank image
            cv2.fillPoly(window_img, np.int32([left_line_pts]), (0, 255, 0))
            cv2.fillPoly(window_img, np.int32([right_line_pts]), (0, 255, 0))
            result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

            # Plot the figures
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(self.warped_frame, cmap='gray')
            ax3.imshow(result)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame")
            ax3.set_title("Warped Frame With Search Window")
            plt.show()

    def get_lane_line_indices_sliding_windows(self, plot=False):
        """
        Get the indices of the lane line pixels using the 
        sliding windows technique.

        :param: plot Show plot or not
        :return: Best fit lines for the left and right lines of the current lane 
        """
        # Sliding window width is +/- margin
        margin = self.margin

        frame_sliding_window = self.warped_frame.copy()

        # Set the height of the sliding windows
        window_height = np.int32(self.warped_frame.shape[0]/self.no_of_windows)

        # Find the x and y coordinates of all the nonzero
        # (i.e. white) pixels in the frame.
        nonzero = self.warped_frame.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])

        # Store the pixel indices for the left and right lane lines
        left_lane_inds = []
        right_lane_inds = []

        # Current positions for pixel indices for each window,
        # which we will continue to update
        leftx_base, rightx_base = self.histogram_peak()
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Go through one window at a time
        no_of_windows = self.no_of_windows

        for window in range(no_of_windows):

            # Identify window boundaries in x and y (and right and left)
            win_y_low = self.warped_frame.shape[0] - \
                (window + 1) * window_height
            win_y_high = self.warped_frame.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(frame_sliding_window, (win_xleft_low, win_y_low), (
                win_xleft_high, win_y_high), (255, 255, 255), 2)
            cv2.rectangle(frame_sliding_window, (win_xright_low, win_y_low), (
                win_xright_high, win_y_high), (255, 255, 255), 2)

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (
                nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (
                nonzerox < win_xright_high)).nonzero()[0]

            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # If you found > minpix pixels, recenter next window on mean position
            minpix = self.minpix
            if len(good_left_inds) > minpix:
                leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # Extract the pixel coordinates for the left and right lane lines
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        if len(lefty) == 0 or len(leftx) == 0 or len(righty) == 0 or len(rightx) == 0:
            return None, None

        # Fit a second order polynomial curve to the pixel coordinates for
        # the left and right lane lines
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        self.left_fit = left_fit
        self.right_fit = right_fit

        if plot == True:

            # Create the x and y values to plot on the image
            ploty = np.linspace(
                0, frame_sliding_window.shape[0]-1, frame_sliding_window.shape[0])
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + \
                right_fit[1]*ploty + right_fit[2]

            # Generate an image to visualize the result
            out_img = np.dstack((
                frame_sliding_window, frame_sliding_window, (
                    frame_sliding_window))) * 255

            # Add color to the left line pixels and right line pixels
            out_img[nonzeroy[left_lane_inds],
                    nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [
                0, 0, 255]

            # Plot the figure with the sliding windows
            figure, (ax1, ax2, ax3) = plt.subplots(3, 1)  # 3 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(frame_sliding_window, cmap='gray')
            ax3.imshow(out_img)
            ax3.plot(left_fitx, ploty, color='yellow')
            ax3.plot(right_fitx, ploty, color='yellow')
            ax1.set_title("Original Frame")
            ax2.set_title("Warped Frame with Sliding Windows")
            ax3.set_title("Detected Lane Lines with Sliding Windows")
            plt.show()

        return self.left_fit, self.right_fit

    def get_line_markings(self, frame=None):
        """
        Isolates lane lines.

          :param frame: The camera frame that contains the lanes we want to detect
        :return: Binary (i.e. black and white) image containing the lane lines.
        """
        if frame is None:
            frame = self.orig_frame

        # Convert the video frame from BGR (blue, green, red)
        # color space to HLS (hue, saturation, lightness).
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        ################### Isolate possible lane line edges ######################

        # Perform Sobel edge detection on the L (lightness) channel of
        # the image to detect sharp discontinuities in the pixel intensities
        # along the x and y axis of the video frame.
        # sxbinary is a matrix full of 0s (black) and 255 (white) intensity values
        # Relatively light pixels get made white. Dark pixels get made black.
        _, sxbinary = edge.threshold(
            hls[:, :, 1], thresh=self.config.LIGHTNESS_THRESH)
        sxbinary = edge.blur_gaussian(
            sxbinary, ksize=self.config.GAUSSIAN_BLUR_KERNEL)  # Reduce noise

        # 1s will be in the cells with the highest Sobel derivative values
        # (i.e. strongest lane line edges)
        sxbinary = edge.mag_thresh(
            sxbinary, sobel_kernel=self.config.SOBEL_KERNEL, thresh=self.config.SOBEL_THRESH)

        ######################## Isolate possible lane lines ######################

        # Perform binary thresholding on the S (saturation) channel
        # of the video frame. A high saturation value means the hue color is pure.
        # We expect lane lines to be nice, pure colors (i.e. solid white, yellow)
        # and have high saturation channel values.
        # s_binary is matrix full of 0s (black) and 255 (white) intensity values
        # White in the regions with the purest hue colors (e.g. >80...play with
        # this value for best results).
        s_channel = hls[:, :, 2]  # use only the saturation channel data
        _, s_binary = edge.threshold(s_channel, self.config.SATURATION_THRESH)

        # Perform binary thresholding on the R (red) channel of the
        # original BGR video frame.
        # r_thresh is a matrix full of 0s (black) and 255 (white) intensity values
        # White in the regions with the richest red channel values (e.g. >120).
        # Remember, pure white is bgr(255, 255, 255).
        # Pure yellow is bgr(0, 255, 255). Both have high red channel values.
        _, r_thresh = edge.threshold(
            frame[:, :, 2], thresh=self.config.RED_CHANNEL_THRESH)

        # Lane lines should be pure in color and have high red channel values
        # Bitwise AND operation to reduce noise and black-out any pixels that
        # don't appear to be nice, pure, solid colors (like white or yellow lane
        # lines.)
        rs_binary = cv2.bitwise_and(s_binary, r_thresh)

        ### Combine the possible lane lines with the possible lane line edges #####
        # If you show rs_binary visually, you'll see that it is not that different
        # from this return value. The edges of lane lines are thin lines of pixels.
        self.lane_line_markings = cv2.bitwise_or(rs_binary, sxbinary.astype(
            np.uint8))
        return self.lane_line_markings

    def histogram_peak(self):
        """
        Get the left and right peak of the histogram

        Return the x coordinate of the left histogram peak and the right histogram
        peak.
        """
        midpoint = np.int32(self.histogram.shape[0]/2)
        leftx_base = np.argmax(self.histogram[:midpoint])
        rightx_base = np.argmax(self.histogram[midpoint:]) + midpoint

        # (x coordinate of left peak, x coordinate of right peak)
        return leftx_base, rightx_base

    def overlay_lane_lines(self, plot=False):
        """
        Overlay lane lines on the original frame
        :param: Plot the lane lines if True
        :return: Lane with overlay
        """
        # Generate an image to draw the lane lines on
        warp_zero = np.zeros_like(self.warped_frame).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([
                             self.left_fitx, self.ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([
                              self.right_fitx, self.ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw lane on the warped blank image
        cv2.fillPoly(color_warp, np.int32([pts]), self.config.LANE_COLOR)

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, self.inv_transformation_matrix, (
                                      self.orig_frame.shape[
                                          1], self.orig_frame.shape[0]))

        # Combine the result with the original image
        result = cv2.addWeighted(
            self.orig_frame, 1, newwarp, self.config.LANE_ALPHA, 0)

        if plot == True:

            # Plot the figures
            figure, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows, 1 column
            figure.set_size_inches(10, 10)
            figure.tight_layout(pad=3.0)
            ax1.imshow(cv2.cvtColor(self.orig_frame, cv2.COLOR_BGR2RGB))
            ax2.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            ax1.set_title("Original Frame")
            ax2.set_title("Original Frame With Lane Overlay")
            plt.show()

        return result

    def perspective_transform(self, frame=None, plot=False):
        """
        Perform the perspective transform.
        :param: frame Current frame
        :param: plot Plot the warped image if True
        :return: Bird's eye view of the current lane
        """
        if frame is None:
            frame = self.lane_line_markings

        # Calculate the transformation matrix
        self.transformation_matrix = cv2.getPerspectiveTransform(
            self.roi_points, self.desired_roi_points)

        # Calculate the inverse transformation matrix
        self.inv_transformation_matrix = cv2.getPerspectiveTransform(
            self.desired_roi_points, self.roi_points)

        # Perform the transform using the transformation matrix
        self.warped_frame = cv2.warpPerspective(
            frame, self.transformation_matrix, self.orig_image_size, flags=(
                cv2.INTER_LINEAR))

        # Convert image to binary
        (thresh, binary_warped) = cv2.threshold(
            self.warped_frame, 127, 255, cv2.THRESH_BINARY)
        self.warped_frame = binary_warped

        # Display the perspective transformed (i.e. warped) frame
        if plot == True:
            warped_copy = self.warped_frame.copy()
            warped_plot = cv2.polylines(warped_copy, np.int3232([
                self.desired_roi_points]), True, (147, 20, 255), 3)

            # Display the image
            while (1):
                cv2.imshow('Warped Image', warped_plot)

                # Press any key to stop
                if cv2.waitKey(0):
                    break

            cv2.destroyAllWindows()

        return self.warped_frame

    def plot_roi(self, frame=None, plot=False):
        """
        Plot the region of interest on an image.
        :param: frame The current image frame
        :param: plot Plot the roi image if True
        """
        if plot == False:
            return

        if frame is None:
            frame = self.orig_frame.copy()

        # Overlay trapezoid on the frame
        this_image = cv2.polylines(frame, np.int3232([
            self.roi_points]), True, (147, 20, 255), 3)

        # Display the image
        while (1):
            cv2.imshow('ROI Image', this_image)

            # Press any key to stop
            if cv2.waitKey(0):
                break

        cv2.destroyAllWindows()

    def draw_roi_overlay(self, frame):
        """
        Draw a faded ROI overlay on the frame to show the detection region

        :param frame: Frame to draw ROI on
        :return: Frame with ROI overlay
        """
        if not self.config.SHOW_ROI:
            return frame

        # Create a copy to draw on
        overlay = frame.copy()
        output = frame.copy()

        # Convert ROI points to integer array
        roi_pts = np.array([self.roi_points], dtype=np.int32)

        # Draw filled polygon for the faded effect
        cv2.fillPoly(overlay, roi_pts, self.config.ROI_COLOR)

        # Blend the overlay with the original frame
        cv2.addWeighted(overlay, self.config.ROI_ALPHA, output,
                        1 - self.config.ROI_ALPHA, 0, output)

        # Draw border around ROI
        cv2.polylines(output, roi_pts, True, self.config.ROI_BORDER_COLOR,
                      self.config.ROI_BORDER_THICKNESS, cv2.LINE_AA)

        return output


def process_frame(frame, config=None, return_detection_result=False):
    """
    Process a single frame for lane detection

    :param frame: Input frame to process
    :param config: LaneConfig object (optional)
    :param return_detection_result: If True, return (frame, LaneDetectionResult), else just frame
    :return: Frame with lane lines and statistics overlay, or tuple (frame, data)
    """
    start_time = time.time()

    try:
        # Create a Lane object
        lane = Lane(orig_frame=frame, config=config)

        # Perform thresholding to isolate lane lines
        _ = lane.get_line_markings()

        # Perform the perspective transform to generate a bird's eye view
        _ = lane.perspective_transform(plot=False)

        # Generate the image histogram to serve as a starting point
        # for finding lane line pixels
        _ = lane.calculate_histogram(plot=False)

        # Find lane line pixels using the sliding window method
        left_fit, right_fit = lane.get_lane_line_indices_sliding_windows(
            plot=False)

        # Check if lane detection was successful
        if left_fit is None or right_fit is None:
            final_frame = lane.draw_roi_overlay(frame)

            if return_detection_result:
                processing_time = (time.time() - start_time) * 1000
                result = LaneDetectionResult(
                    timestamp=time.time(),
                    frame_id="camera",
                    image_shape=frame.shape[:2],
                    roi_points=lane.roi_points,
                    processing_time_ms=processing_time,
                    valid=False,
                    confidence=0.0
                )
                return final_frame, result
            return final_frame

        # Fill in the lane line
        lane.get_lane_line_previous_window(left_fit, right_fit, plot=False)

        # Overlay lines on the original frame
        frame_with_lane_lines = lane.overlay_lane_lines(plot=False)

        # Calculate lane line curvature (left and right lane lines)
        lane.calculate_curvature(print_to_terminal=False)

        # Calculate center offset
        lane.calculate_car_position(print_to_terminal=False)

        # Display curvature and center offset on image
        frame_with_lane_lines2 = lane.display_curvature_offset(
            frame=frame_with_lane_lines, plot=False)

        # Draw ROI overlay to show the detection region
        final_frame = lane.draw_roi_overlay(frame_with_lane_lines2)

        # Extract structured data if requested
        if return_detection_result:
            # Extract left boundary
            left_boundary = extract_boundary_points(
                polynomial=lane.left_fit,
                ploty=lane.ploty,
                leftx=lane.leftx,
                lefty=lane.lefty,
                curvature_radius=lane.left_curvem,
                num_points=50,
                xm_per_pix=config.XM_PER_PIX,
                ym_per_pix=config.YM_PER_PIX
            )

            # Extract right boundary
            right_boundary = extract_boundary_points(
                polynomial=lane.right_fit,
                ploty=lane.ploty,
                leftx=lane.rightx,
                lefty=lane.righty,
                curvature_radius=lane.right_curvem,
                num_points=50,
                xm_per_pix=config.XM_PER_PIX,
                ym_per_pix=config.YM_PER_PIX
            )

            # Calculate midline
            midline = calculate_midline(
                left_boundary, right_boundary, num_points=50)

            # Calculate lane width in meters
            if left_boundary.valid and right_boundary.valid and len(left_boundary.points) > 0:
                # Use bottom of image (index 0) for width calculation
                lane_width = abs(
                    right_boundary.points[0, 0] - left_boundary.points[0, 0])
            else:
                lane_width = 0.0

            # Calculate average curvature radius
            avg_curvature = (lane.left_curvem +
                             lane.right_curvem) / 2.0

            # Calculate overall confidence as average of boundary confidences
            overall_confidence = (
                left_boundary.confidence + right_boundary.confidence) / 2.0

            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000

            # Create detection result
            result = LaneDetectionResult(
                timestamp=time.time(),
                frame_id="camera",
                left_boundary=left_boundary,
                right_boundary=right_boundary,
                midline=midline,
                center_offset=lane.center_offset / 100.0,  # Convert cm to meters
                lane_width=lane_width,
                avg_curvature_radius=avg_curvature,
                image_shape=frame.shape[:2],
                roi_points=lane.roi_points,
                processing_time_ms=processing_time,
                valid=True,
                confidence=overall_confidence
            )

            return final_frame, result

        return final_frame
    except Exception as e:
        # Return original frame if any error occurs
        print(f"Warning: Frame processing failed - {e}")
        if return_detection_result:
            processing_time = (time.time() - start_time) * 1000
            result = LaneDetectionResult(
                timestamp=time.time(),
                frame_id="camera",
                image_shape=frame.shape[:2] if frame is not None else (0, 0),
                processing_time_ms=processing_time,
                valid=False,
                confidence=0.0
            )
            return frame, result
        return frame


class LaneConfig:
    """
    Centralized configuration for lane detection parameters
    """
    # Region of Interest (ROI) - Perspective Transform
    ROI_TOP_LEFT = (int(640 * 0.45), int(480 * 0.8))      # Top-left corner
    ROI_BOTTOM_LEFT = (int(640 * 0.3), int(480 * 0.95))   # Bottom-left corner
    ROI_BOTTOM_RIGHT = (int(640 * 0.55), int(480 * 0.95)
                        )  # Bottom-right corner
    ROI_TOP_RIGHT = (int(640 * 0.55), int(480 * 0.8))     # Top-right corner

    # Perspective transform padding (percentage of image width)
    PERSPECTIVE_PADDING = 0.25

    # Sliding window parameters
    NUM_WINDOWS = 9              # Number of sliding windows
    WINDOW_MARGIN_RATIO = 1/12   # Window width ratio (relative to image width)
    MIN_PIXELS_RATIO = 1/24      # Minimum pixels to recenter window

    # Edge detection thresholds
    LIGHTNESS_THRESH = (120, 255)  # L channel threshold
    SOBEL_KERNEL = 3               # Sobel kernel size
    SOBEL_THRESH = (110, 255)      # Sobel magnitude threshold
    GAUSSIAN_BLUR_KERNEL = 3       # Gaussian blur kernel size

    # Color channel thresholds
    # S channel threshold (increased for better lane detection)
    SATURATION_THRESH = (100, 255)
    # R channel threshold (increased for white/yellow lanes)
    RED_CHANNEL_THRESH = (200, 255)

    # Pixel to real-world conversion
    # meters per pixel in y dimension (adjusted for typical road)
    YM_PER_PIX = 30.0 / 720
    # meters per pixel in x dimension (standard US lane width)
    XM_PER_PIX = 3.7 / 700

    # Display settings
    FONT_SCALE_RATIO = 0.6 / 600  # Font scale relative to image width
    TEXT_THICKNESS = 2
    TEXT_COLOR = (255, 255, 255)
    TEXT_Y_OFFSET_1 = 30  # First line offset
    TEXT_Y_OFFSET_2 = 60  # Second line offset
    TEXT_X_OFFSET = 10

    # Lane overlay settings
    LANE_COLOR = (0, 255, 0)  # Green
    LANE_ALPHA = 0.3          # Transparency

    # ROI overlay settings
    SHOW_ROI = True                    # Show ROI overlay on output
    ROI_COLOR = (0, 165, 255)          # Orange color in BGR
    ROI_ALPHA = 0.2                    # ROI transparency (0.0-1.0)
    ROI_BORDER_COLOR = (0, 165, 255)   # Orange border
    ROI_BORDER_THICKNESS = 2           # Border thickness in pixels

    @classmethod
    def get_roi_points(cls, width, height):
        """
        Get ROI points scaled to actual image dimensions

        :param width: Image width
        :param height: Image height
        :return: ROI points as numpy array
        """
        scale_x = width / 640
        scale_y = height / 480

        return np.float32([
            (cls.ROI_TOP_LEFT[0] * scale_x, cls.ROI_TOP_LEFT[1] * scale_y),
            (cls.ROI_BOTTOM_LEFT[0] * scale_x,
             cls.ROI_BOTTOM_LEFT[1] * scale_y),
            (cls.ROI_BOTTOM_RIGHT[0] * scale_x,
             cls.ROI_BOTTOM_RIGHT[1] * scale_y),
            (cls.ROI_TOP_RIGHT[0] * scale_x, cls.ROI_TOP_RIGHT[1] * scale_y)
        ])

    @classmethod
    def get_desired_roi_points(cls, width, height):
        """
        Get desired ROI points after perspective transform

        :param width: Image width
        :param height: Image height
        :return: Desired ROI points as numpy array
        """
        padding = int(cls.PERSPECTIVE_PADDING * width)
        return np.float32([
            [padding, 0],
            [padding, height],
            [width - padding, height],
            [width - padding, 0]
        ])


class LaneDetectionNode(Node):
    """ROS2 node for real-time lane detection"""

    def __init__(self):
        super().__init__('lane_detection_node')
        
        # Declare and get parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize CV bridge for image conversion
        self.bridge = CvBridge()
        
        # Initialize lane detection configuration
        self.config = self._create_lane_config()
        
        # Frame counter for skipping
        self.frame_counter = 0
        
        # Create subscriber for camera images
        self.image_sub = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            self.queue_size
        )
        
        # Create publishers for lane detection results (using nav_msgs/Path)
        self.left_boundary_pub = self.create_publisher(
            Path,
            self.left_boundary_topic,
            self.queue_size
        )
        
        self.right_boundary_pub = self.create_publisher(
            Path,
            self.right_boundary_topic,
            self.queue_size
        )
        
        self.midline_pub = self.create_publisher(
            Path,
            self.midline_topic,
            self.queue_size
        )
        
        # Optional visualization publisher
        if self.publish_visualization:
            self.viz_pub = self.create_publisher(
                Image,
                self.visualization_topic,
                self.queue_size
            )
        
        self.get_logger().info('Lane Detection Node initialized')
        self.get_logger().info(f'Subscribing to: {self.camera_topic}')
        self.get_logger().info(f'Publishing to:')
        self.get_logger().info(f'  - Left boundary: {self.left_boundary_topic}')
        self.get_logger().info(f'  - Right boundary: {self.right_boundary_topic}')
        self.get_logger().info(f'  - Midline: {self.midline_topic}')
        self.get_logger().info(f'  - Full result: {self.detection_result_topic}')
        if self.publish_visualization:
            self.get_logger().info(f'  - Visualization: {self.visualization_topic}')

    def _declare_parameters(self):
        """Declare all ROS2 parameters"""
        # Topic configuration
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('left_boundary_topic', '/lane_detection/left_boundary')
        self.declare_parameter('right_boundary_topic', '/lane_detection/right_boundary')
        self.declare_parameter('midline_topic', '/lane_detection/midline')
        self.declare_parameter('detection_result_topic', '/lane_detection/result')
        self.declare_parameter('visualization_topic', '/lane_detection/visualization')
        self.declare_parameter('publish_visualization', True)
        self.declare_parameter('output_frame_id', 'camera')
        self.declare_parameter('queue_size', 10)
        self.declare_parameter('frame_skip', 1)
        
        # ROI parameters
        self.declare_parameter('roi.top_left_x', 0.45)
        self.declare_parameter('roi.top_left_y', 0.8)
        self.declare_parameter('roi.bottom_left_x', 0.3)
        self.declare_parameter('roi.bottom_left_y', 0.95)
        self.declare_parameter('roi.bottom_right_x', 0.55)
        self.declare_parameter('roi.bottom_right_y', 0.95)
        self.declare_parameter('roi.top_right_x', 0.55)
        self.declare_parameter('roi.top_right_y', 0.8)
        self.declare_parameter('roi.perspective_padding', 0.25)
        self.declare_parameter('roi.show_roi', True)
        self.declare_parameter('roi.roi_color_b', 0)
        self.declare_parameter('roi.roi_color_g', 165)
        self.declare_parameter('roi.roi_color_r', 255)
        self.declare_parameter('roi.roi_alpha', 0.2)
        self.declare_parameter('roi.roi_border_thickness', 2)
        
        # Sliding window parameters
        self.declare_parameter('sliding_window.num_windows', 9)
        self.declare_parameter('sliding_window.window_margin_ratio', 0.083333)
        self.declare_parameter('sliding_window.min_pixels_ratio', 0.041667)
        
        # Edge detection parameters
        self.declare_parameter('edge_detection.lightness_thresh_min', 120)
        self.declare_parameter('edge_detection.lightness_thresh_max', 255)
        self.declare_parameter('edge_detection.sobel_kernel', 3)
        self.declare_parameter('edge_detection.sobel_thresh_min', 110)
        self.declare_parameter('edge_detection.sobel_thresh_max', 255)
        self.declare_parameter('edge_detection.gaussian_blur_kernel', 3)
        
        # Color thresholds
        self.declare_parameter('color_thresholds.saturation_thresh_min', 100)
        self.declare_parameter('color_thresholds.saturation_thresh_max', 255)
        self.declare_parameter('color_thresholds.red_channel_thresh_min', 200)
        self.declare_parameter('color_thresholds.red_channel_thresh_max', 255)
        
        # Calibration
        self.declare_parameter('calibration.ym_per_pix', 0.041667)
        self.declare_parameter('calibration.xm_per_pix', 0.0052857)
        
        # Display
        self.declare_parameter('display.font_scale_ratio', 0.001)
        self.declare_parameter('display.text_thickness', 2)
        self.declare_parameter('display.text_color_b', 255)
        self.declare_parameter('display.text_color_g', 255)
        self.declare_parameter('display.text_color_r', 255)
        self.declare_parameter('display.text_y_offset_1', 30)
        self.declare_parameter('display.text_y_offset_2', 60)
        self.declare_parameter('display.text_x_offset', 10)
        self.declare_parameter('display.lane_color_b', 0)
        self.declare_parameter('display.lane_color_g', 255)
        self.declare_parameter('display.lane_color_r', 0)
        self.declare_parameter('display.lane_alpha', 0.3)
        
        # Output
        self.declare_parameter('output.num_boundary_points', 50)
        self.declare_parameter('output.num_midline_points', 50)
        self.declare_parameter('output.show_arrows', True)
        self.declare_parameter('output.arrow_spacing', 5)

    def _load_parameters(self):
        """Load parameters from ROS2 parameter server"""
        self.camera_topic = self.get_parameter('camera_topic').value
        self.left_boundary_topic = self.get_parameter('left_boundary_topic').value
        self.right_boundary_topic = self.get_parameter('right_boundary_topic').value
        self.midline_topic = self.get_parameter('midline_topic').value
        self.detection_result_topic = self.get_parameter('detection_result_topic').value
        self.visualization_topic = self.get_parameter('visualization_topic').value
        self.publish_visualization = self.get_parameter('publish_visualization').value
        self.output_frame_id = self.get_parameter('output_frame_id').value
        self.queue_size = self.get_parameter('queue_size').value
        self.frame_skip = self.get_parameter('frame_skip').value
        
        # Output configuration
        self.num_boundary_points = self.get_parameter('output.num_boundary_points').value
        self.num_midline_points = self.get_parameter('output.num_midline_points').value
        self.show_arrows = self.get_parameter('output.show_arrows').value
        self.arrow_spacing = self.get_parameter('output.arrow_spacing').value

    def _create_lane_config(self) -> LaneConfig:
        """Create LaneConfig object from ROS2 parameters"""
        config = LaneConfig()
        
        # Override config with ROS2 parameters
        # Note: ROI points will be set dynamically based on image size
        # Store normalized values for later scaling
        self.roi_normalized = {
            'top_left': (
                self.get_parameter('roi.top_left_x').value,
                self.get_parameter('roi.top_left_y').value
            ),
            'bottom_left': (
                self.get_parameter('roi.bottom_left_x').value,
                self.get_parameter('roi.bottom_left_y').value
            ),
            'bottom_right': (
                self.get_parameter('roi.bottom_right_x').value,
                self.get_parameter('roi.bottom_right_y').value
            ),
            'top_right': (
                self.get_parameter('roi.top_right_x').value,
                self.get_parameter('roi.top_right_y').value
            )
        }
        
        config.PERSPECTIVE_PADDING = self.get_parameter('roi.perspective_padding').value
        config.SHOW_ROI = self.get_parameter('roi.show_roi').value
        config.ROI_COLOR = (
            self.get_parameter('roi.roi_color_b').value,
            self.get_parameter('roi.roi_color_g').value,
            self.get_parameter('roi.roi_color_r').value
        )
        config.ROI_ALPHA = self.get_parameter('roi.roi_alpha').value
        config.ROI_BORDER_THICKNESS = self.get_parameter('roi.roi_border_thickness').value
        
        # Sliding window
        config.NUM_WINDOWS = self.get_parameter('sliding_window.num_windows').value
        config.WINDOW_MARGIN_RATIO = self.get_parameter('sliding_window.window_margin_ratio').value
        config.MIN_PIXELS_RATIO = self.get_parameter('sliding_window.min_pixels_ratio').value
        
        # Edge detection
        config.LIGHTNESS_THRESH = (
            self.get_parameter('edge_detection.lightness_thresh_min').value,
            self.get_parameter('edge_detection.lightness_thresh_max').value
        )
        config.SOBEL_KERNEL = self.get_parameter('edge_detection.sobel_kernel').value
        config.SOBEL_THRESH = (
            self.get_parameter('edge_detection.sobel_thresh_min').value,
            self.get_parameter('edge_detection.sobel_thresh_max').value
        )
        config.GAUSSIAN_BLUR_KERNEL = self.get_parameter('edge_detection.gaussian_blur_kernel').value
        
        # Color thresholds
        config.SATURATION_THRESH = (
            self.get_parameter('color_thresholds.saturation_thresh_min').value,
            self.get_parameter('color_thresholds.saturation_thresh_max').value
        )
        config.RED_CHANNEL_THRESH = (
            self.get_parameter('color_thresholds.red_channel_thresh_min').value,
            self.get_parameter('color_thresholds.red_channel_thresh_max').value
        )
        
        # Calibration
        config.YM_PER_PIX = self.get_parameter('calibration.ym_per_pix').value
        config.XM_PER_PIX = self.get_parameter('calibration.xm_per_pix').value
        
        # Display
        config.FONT_SCALE_RATIO = self.get_parameter('display.font_scale_ratio').value
        config.TEXT_THICKNESS = self.get_parameter('display.text_thickness').value
        config.TEXT_COLOR = (
            self.get_parameter('display.text_color_b').value,
            self.get_parameter('display.text_color_g').value,
            self.get_parameter('display.text_color_r').value
        )
        config.TEXT_Y_OFFSET_1 = self.get_parameter('display.text_y_offset_1').value
        config.TEXT_Y_OFFSET_2 = self.get_parameter('display.text_y_offset_2').value
        config.TEXT_X_OFFSET = self.get_parameter('display.text_x_offset').value
        config.LANE_COLOR = (
            self.get_parameter('display.lane_color_b').value,
            self.get_parameter('display.lane_color_g').value,
            self.get_parameter('display.lane_color_r').value
        )
        config.LANE_ALPHA = self.get_parameter('display.lane_alpha').value
        
        return config

    def _scale_roi_to_image(self, width: int, height: int):
        """Scale normalized ROI coordinates to actual image dimensions"""
        self.config.ROI_TOP_LEFT = (
            int(width * self.roi_normalized['top_left'][0]),
            int(height * self.roi_normalized['top_left'][1])
        )
        self.config.ROI_BOTTOM_LEFT = (
            int(width * self.roi_normalized['bottom_left'][0]),
            int(height * self.roi_normalized['bottom_left'][1])
        )
        self.config.ROI_BOTTOM_RIGHT = (
            int(width * self.roi_normalized['bottom_right'][0]),
            int(height * self.roi_normalized['bottom_right'][1])
        )
        self.config.ROI_TOP_RIGHT = (
            int(width * self.roi_normalized['top_right'][0]),
            int(height * self.roi_normalized['top_right'][1])
        )

    def image_callback(self, msg: Image):
        """Process incoming camera images"""
        # Frame skipping
        self.frame_counter += 1
        if self.frame_counter % self.frame_skip != 0:
            return
        
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Scale ROI to image dimensions
            height, width = cv_image.shape[:2]
            self._scale_roi_to_image(width, height)
            
            # Process frame for lane detection
            viz_frame, detection_result = process_frame(
                cv_image,
                config=self.config,
                return_detection_result=True
            )
            
            # Create header for messages
            header = Header()
            header.stamp = self.get_clock().now().to_msg()
            header.frame_id = self.output_frame_id
            
            # Publish results as Path messages
            if detection_result.valid:
                self._publish_paths(header, detection_result)
            else:
                self.get_logger().warn('Lane detection failed for this frame', throttle_duration_sec=2.0)
            
            # Publish visualization if enabled
            if self.publish_visualization and viz_frame is not None:
                # Add boundary and midline visualization
                viz_with_paths = visualize_boundaries_and_midline(
                    viz_frame,
                    detection_result.left_boundary,
                    detection_result.right_boundary,
                    detection_result.midline,
                    show_arrows=self.show_arrows,
                    arrow_spacing=self.arrow_spacing
                )
                viz_msg = self.bridge.cv2_to_imgmsg(viz_with_paths, encoding='bgr8')
                viz_msg.header = header
                self.viz_pub.publish(viz_msg)
                
        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def _publish_paths(self, header: Header, result):
        """Publish lane boundaries and midline as nav_msgs/Path messages"""
        
        # Publish left boundary as Path
        if result.left_boundary.valid and len(result.left_boundary.points) > 0:
            left_path = self._create_path_msg(header, result.left_boundary.points)
            self.left_boundary_pub.publish(left_path)
        
        # Publish right boundary as Path
        if result.right_boundary.valid and len(result.right_boundary.points) > 0:
            right_path = self._create_path_msg(header, result.right_boundary.points)
            self.right_boundary_pub.publish(right_path)
        
        # Publish midline as Path
        if result.midline.valid and len(result.midline.points) > 0:
            midline_path = self._create_path_msg(header, result.midline.points)
            self.midline_pub.publish(midline_path)

    def _create_path_msg(self, header: Header, points: np.ndarray) -> Path:
        """
        Create a nav_msgs/Path message from points array
        
        Args:
            header: Message header
            points: Nx2 array of [x, y] points in meters
            
        Returns:
            nav_msgs/Path message
        """
        path = Path()
        path.header = header
        
        # Convert each point to a PoseStamped
        for point in points:
            pose = PoseStamped()
            pose.header = header
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            
            # Orientation is not meaningful for lane boundaries, set to identity
            pose.pose.orientation.w = 1.0
            pose.pose.orientation.x = 0.0
            pose.pose.orientation.y = 0.0
            pose.pose.orientation.z = 0.0
            
            path.poses.append(pose)
        
        return path


def main(args=None):
    """Main entry point for the node"""
    rclpy.init(args=args)
    
    node = LaneDetectionNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
