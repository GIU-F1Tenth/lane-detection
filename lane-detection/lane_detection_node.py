#!/usr/bin/env python3
"""
ROS2 Lane Detection Node

Subscribes to camera images and publishes lane detection results as Path messages:
- Left lane boundary path
- Right lane boundary path
- Lane midline/centerline path

Author: Fam Shihata
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from .main import process_frame
from .lane_utils import visualize_boundaries_and_midline


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
