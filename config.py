# Lane Detection Configuration File
# Adjust these parameters to optimize lane detection for your specific use case

# ============================================================================
# Region of Interest (ROI) Configuration
# ============================================================================
# These define the trapezoid region to focus on for lane detection
# Values are in pixels, scaled relative to a 640x480 reference resolution
# Adjust based on camera position, mounting angle, and road perspective

# ROI corners as (x, y) coordinates
# Format: (horizontal_position, vertical_position)
# Top-left corner of trapezoid
ROI_TOP_LEFT = (int(640 * 0.25), int(480 * 0.4))
ROI_BOTTOM_LEFT = (int(640 * 0.1), int(480 * 0.4))   # Bottom-left corner
ROI_BOTTOM_RIGHT = (int(640 * 0.2), int(480 * 0.2))  # Bottom-right corner
ROI_TOP_RIGHT = (int(640 * 0.2), int(480 * 0.2))     # Top-right corner

# Perspective transform padding (0.0 to 0.5)
# How much to pad the sides after bird's-eye transformation
PERSPECTIVE_PADDING = 0.25

# ============================================================================
# Sliding Window Parameters
# ============================================================================
# Controls how lane lines are detected using the sliding window technique

NUM_WINDOWS = 9              # Number of sliding windows (5-15 recommended)
WINDOW_MARGIN_RATIO = 1/12   # Window width as fraction of image width
MIN_PIXELS_RATIO = 1/24      # Minimum pixels to recenter window

# ============================================================================
# Edge Detection Thresholds
# ============================================================================
# Controls sensitivity of edge detection for lane line identification

LIGHTNESS_THRESH = (120, 255)   # L channel threshold (0-255)
# Sobel kernel size (must be odd: 3, 5, 7, etc.)
SOBEL_KERNEL = 3
SOBEL_THRESH = (110, 255)       # Sobel magnitude threshold (0-255)
GAUSSIAN_BLUR_KERNEL = 3        # Gaussian blur kernel size (odd number)

# ============================================================================
# Color Channel Thresholds
# ============================================================================
# Controls which colors are identified as lane lines

# S channel threshold - higher values = purer colors
SATURATION_THRESH = (100, 255)
# R channel threshold - detects white/yellow lanes
RED_CHANNEL_THRESH = (200, 255)

# ============================================================================
# Real-World Conversion Factors
# ============================================================================
# Convert pixel measurements to real-world distances

YM_PER_PIX = 30.0 / 720   # Meters per pixel in y dimension (vertical)
XM_PER_PIX = 3.7 / 700    # Meters per pixel in x dimension (horizontal)
# 3.7m is standard US highway lane width

# ============================================================================
# Display Settings
# ============================================================================
# Controls overlay appearance

FONT_SCALE_RATIO = 0.6 / 600  # Font scale relative to image width
TEXT_THICKNESS = 2            # Text line thickness
TEXT_COLOR = (255, 255, 255)  # Text color in BGR (white)
TEXT_Y_OFFSET_1 = 30          # First line vertical offset
TEXT_Y_OFFSET_2 = 60          # Second line vertical offset
TEXT_X_OFFSET = 10            # Horizontal offset from left edge

# Lane overlay settings
LANE_COLOR = (0, 255, 0)      # Lane fill color in BGR (green)
LANE_ALPHA = 0.3              # Lane transparency (0.0-1.0)

# ROI (Region of Interest) overlay settings
SHOW_ROI = True                    # Set to False to hide ROI overlay
ROI_COLOR = (0, 165, 255)          # ROI fill color in BGR (orange)
ROI_ALPHA = 0.2                    # ROI transparency (0.0-1.0)
ROI_BORDER_COLOR = (0, 165, 255)   # ROI border color in BGR (orange)
ROI_BORDER_THICKNESS = 2           # ROI border thickness in pixels

# ============================================================================
# Tips for Adjusting Parameters
# ============================================================================
#
# If lanes are not detected:
#   - Lower SATURATION_THRESH and RED_CHANNEL_THRESH
#   - Lower SOBEL_THRESH and LIGHTNESS_THRESH
#   - Adjust ROI points to better match your camera view
#
# If too many false positives:
#   - Increase SATURATION_THRESH and RED_CHANNEL_THRESH
#   - Increase MIN_PIXELS_RATIO
#   - Narrow the ROI region
#
# If detection is jittery:
#   - Increase NUM_WINDOWS
#   - Increase GAUSSIAN_BLUR_KERNEL
#   - Increase MIN_PIXELS_RATIO
#
# For different camera angles:
#   - Adjust ROI_TOP_LEFT and ROI_TOP_RIGHT for horizon line
#   - Adjust ROI_BOTTOM_LEFT and ROI_BOTTOM_RIGHT for near field
#   - Fine-tune PERSPECTIVE_PADDING for bird's eye view
