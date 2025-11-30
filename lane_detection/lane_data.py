import numpy as np
from dataclasses import dataclass, field
from typing import Tuple
import time


@dataclass
class _LaneBoundary:
    """
    Represents one lane boundary (left or right)

    Attributes:
        points: (N, 2) array of [x, y] points in bird's-eye view (meters)
        points_pixel: (N, 2) array of [x, y] points in original image (pixels)
        polynomial: [a, b, c] coefficients for quadratic fit (ax² + bx + c)
        curvature_radius: Radius of curvature in meters
        confidence: Detection confidence score (0.0 to 1.0)
        valid: Whether this boundary detection is valid
        num_pixels: Number of pixels detected for this boundary
    """
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    points_pixel: np.ndarray = field(default_factory=lambda: np.array([]))
    polynomial: np.ndarray = field(default_factory=lambda: np.array([]))
    curvature_radius: float = 0.0
    confidence: float = 0.0
    valid: bool = False
    num_pixels: int = 0

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'points': self.points.tolist() if len(self.points) > 0 else [],
            'points_pixel': self.points_pixel.tolist() if len(self.points_pixel) > 0 else [],
            'polynomial': self.polynomial.tolist() if len(self.polynomial) > 0 else [],
            'curvature_radius': float(self.curvature_radius),
            'confidence': float(self.confidence),
            'valid': bool(self.valid),
            'num_pixels': int(self.num_pixels)
        }


@dataclass
class _LaneMidline:
    """
    Represents the center path between lane boundaries

    Attributes:
        points: (N, 2) array of [x, y] waypoints in bird's-eye view (meters)
        points_pixel: (N, 2) array of [x, y] waypoints in original image (pixels)
        heading_angles: Heading angle at each point in radians (from +x axis)
        curvature: Curvature at each point (1/radius in 1/meters)
        distances: Cumulative arc-length distance from start (meters)
        valid: Whether midline calculation is valid
    """
    points: np.ndarray = field(default_factory=lambda: np.array([]))
    points_pixel: np.ndarray = field(default_factory=lambda: np.array([]))
    heading_angles: np.ndarray = field(default_factory=lambda: np.array([]))
    curvature: np.ndarray = field(default_factory=lambda: np.array([]))
    distances: np.ndarray = field(default_factory=lambda: np.array([]))
    valid: bool = False

    def to_dict(self):
        """Convert to dictionary for serialization"""
        return {
            'points': self.points.tolist() if len(self.points) > 0 else [],
            'points_pixel': self.points_pixel.tolist() if len(self.points_pixel) > 0 else [],
            'heading_angles': self.heading_angles.tolist() if len(self.heading_angles) > 0 else [],
            'curvature': self.curvature.tolist() if len(self.curvature) > 0 else [],
            'distances': self.distances.tolist() if len(self.distances) > 0 else [],
            'valid': bool(self.valid)
        }


@dataclass
class LaneDetectionResult:
    """
    Complete lane detection output

    Attributes:
        timestamp: Unix timestamp (seconds since epoch)
        frame_id: Coordinate frame identifier (e.g., "camera", "base_link")
        left_boundary: Left lane boundary detection result
        right_boundary: Right lane boundary detection result
        midline: Calculated midline between boundaries
        center_offset: Lateral offset from lane center in meters (positive = right)
        lane_width: Width of the lane in meters
        avg_curvature_radius: Average curvature radius in meters
        image_shape: (height, width) of the original image
        roi_points: (4, 2) array of ROI trapezoid corners used
        processing_time_ms: Processing time in milliseconds
        valid: Overall detection validity
        confidence: Overall detection confidence (0.0 to 1.0)
    """
    timestamp: float = field(default_factory=time.time)
    frame_id: str = "camera"
    left_boundary: _LaneBoundary = field(default_factory=_LaneBoundary)
    right_boundary: _LaneBoundary = field(default_factory=_LaneBoundary)
    midline: _LaneMidline = field(default_factory=_LaneMidline)
    center_offset: float = 0.0
    lane_width: float = 0.0
    avg_curvature_radius: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    roi_points: np.ndarray = field(default_factory=lambda: np.array([]))
    processing_time_ms: float = 0.0
    valid: bool = False
    confidence: float = 0.0

    def to_dict(self):
        """Convert to dictionary for serialization (ROS2-ready)"""
        return {
            'timestamp': float(self.timestamp),
            'frame_id': str(self.frame_id),
            'left_boundary': self.left_boundary.to_dict(),
            'right_boundary': self.right_boundary.to_dict(),
            'midline': self.midline.to_dict(),
            'center_offset': float(self.center_offset),
            'lane_width': float(self.lane_width),
            'avg_curvature_radius': float(self.avg_curvature_radius),
            'image_shape': list(self.image_shape),
            'roi_points': self.roi_points.tolist() if len(self.roi_points) > 0 else [],
            'processing_time_ms': float(self.processing_time_ms),
            'valid': bool(self.valid),
            'confidence': float(self.confidence)
        }

    def __str__(self):
        """Human-readable string representation"""
        return (
            f"LaneDetectionResult(\n"
            f"  Valid: {self.valid}\n"
            f"  Confidence: {self.confidence:.2f}\n"
            f"  Center Offset: {self.center_offset:.3f} m\n"
            f"  Lane Width: {self.lane_width:.3f} m\n"
            f"  Avg Curvature Radius: {self.avg_curvature_radius:.1f} m\n"
            f"  Processing Time: {self.processing_time_ms:.2f} ms\n"
            f"  Left Boundary: Valid={self.left_boundary.valid}, "
            f"Confidence={self.left_boundary.confidence:.2f}, "
            f"Points={len(self.left_boundary.points)}\n"
            f"  Right Boundary: Valid={self.right_boundary.valid}, "
            f"Confidence={self.right_boundary.confidence:.2f}, "
            f"Points={len(self.right_boundary.points)}\n"
            f"  Midline: Valid={self.midline.valid}, "
            f"Points={len(self.midline.points)}\n"
            f")"
        )
