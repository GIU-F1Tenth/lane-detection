import numpy as np
import cv2
from .lane_data import _LaneBoundary, _LaneMidline


def extract_boundary_points(
    polynomial: np.ndarray,
    ploty: np.ndarray,
    leftx: np.ndarray,
    lefty: np.ndarray,
    curvature_radius: float,
    num_points: int = 50,
    xm_per_pix: float = 3.7 / 700,
    ym_per_pix: float = 30.0 / 720
) -> _LaneBoundary:
    """
    Extract boundary points from polynomial fit

    Args:
        polynomial: [a, b, c] coefficients for quadratic fit
        ploty: Y coordinates for the polynomial
        leftx: Raw x pixel coordinates detected
        lefty: Raw y pixel coordinates detected
        curvature_radius: Radius of curvature in meters
        num_points: Number of points to sample along the boundary
        xm_per_pix: Meters per pixel in x dimension
        ym_per_pix: Meters per pixel in y dimension

    Returns:
        LaneBoundary object with extracted points
    """
    if polynomial is None or len(polynomial) == 0:
        return _LaneBoundary(valid=False)

    # Calculate pixel coordinates along the polynomial
    fitx = polynomial[0] * ploty**2 + polynomial[1] * ploty + polynomial[2]

    # Resample to uniform number of points
    if len(ploty) > num_points:
        indices = np.linspace(0, len(ploty) - 1, num_points, dtype=int)
        sampled_x = fitx[indices]
        sampled_y = ploty[indices]
    else:
        sampled_x = fitx
        sampled_y = ploty

    # Create points array in pixel coordinates [x, y]
    points_pixel = np.column_stack([sampled_x, sampled_y])

    # Convert to meters (bird's-eye view coordinates)
    # Origin at bottom-left of image
    points_meters = np.column_stack([
        sampled_x * xm_per_pix,
        sampled_y * ym_per_pix
    ])

    # Calculate confidence based on number of detected pixels
    num_pixels = len(leftx)
    expected_pixels = len(ploty) * 0.5  # Expect at least 50% coverage
    confidence = min(1.0, num_pixels / max(1, expected_pixels))

    # Calculate polynomial fit error if we have raw points
    if num_pixels > 0:
        predicted_x = polynomial[0] * lefty**2 + \
            polynomial[1] * lefty + polynomial[2]
        residuals = np.abs(leftx - predicted_x)
        mean_error = np.mean(residuals)
        # Reduce confidence if fit error is high (>10 pixels)
        if mean_error > 10:
            confidence *= max(0.3, 1.0 - (mean_error - 10) / 50)

    return _LaneBoundary(
        points=points_meters,
        points_pixel=points_pixel,
        polynomial=polynomial,
        curvature_radius=curvature_radius,
        confidence=float(np.clip(confidence, 0.0, 1.0)),
        valid=True,
        num_pixels=num_pixels
    )


def calculate_midline(
    left_boundary: _LaneBoundary,
    right_boundary: _LaneBoundary,
    num_points: int = 50
) -> _LaneMidline:
    """
    Calculate the midline between two lane boundaries

    Args:
        left_boundary: Left lane boundary
        right_boundary: Right lane boundary
        num_points: Number of waypoints for the midline

    Returns:
        LaneMidline object with waypoints and heading angles
    """
    if not left_boundary.valid or not right_boundary.valid:
        return _LaneMidline(valid=False)

    if len(left_boundary.points) == 0 or len(right_boundary.points) == 0:
        return _LaneMidline(valid=False)

    # Ensure both boundaries have same number of points
    n_left = len(left_boundary.points)
    n_right = len(right_boundary.points)
    n_points = min(n_left, n_right, num_points)

    # Resample if needed
    if n_left != n_points:
        left_indices = np.linspace(0, n_left - 1, n_points, dtype=int)
        left_pts = left_boundary.points[left_indices]
        left_pts_pix = left_boundary.points_pixel[left_indices]
    else:
        left_pts = left_boundary.points
        left_pts_pix = left_boundary.points_pixel

    if n_right != n_points:
        right_indices = np.linspace(0, n_right - 1, n_points, dtype=int)
        right_pts = right_boundary.points[right_indices]
        right_pts_pix = right_boundary.points_pixel[right_indices]
    else:
        right_pts = right_boundary.points
        right_pts_pix = right_boundary.points_pixel

    # Calculate midline as average of left and right boundaries
    midline_pts = (left_pts + right_pts) / 2.0
    midline_pts_pix = (left_pts_pix + right_pts_pix) / 2.0

    # Calculate heading angles between consecutive points
    heading_angles = np.zeros(n_points)
    if n_points > 1:
        for i in range(n_points - 1):
            dx = midline_pts[i + 1, 0] - midline_pts[i, 0]
            dy = midline_pts[i + 1, 1] - midline_pts[i, 1]
            heading_angles[i] = np.arctan2(dy, dx)
        # Last point has same heading as previous
        heading_angles[-1] = heading_angles[-2] if n_points > 1 else 0.0

    # Calculate curvature at each point (finite differences)
    curvature = np.zeros(n_points)
    if n_points > 2:
        for i in range(1, n_points - 1):
            # Use three points to estimate curvature
            p1 = midline_pts[i - 1]
            p2 = midline_pts[i]
            p3 = midline_pts[i + 1]

            # Calculate curvature using finite differences
            dx1 = p2[0] - p1[0]
            dy1 = p2[1] - p1[1]
            dx2 = p3[0] - p2[0]
            dy2 = p3[1] - p2[1]

            # Curvature = |dθ/ds|
            angle1 = np.arctan2(dy1, dx1)
            angle2 = np.arctan2(dy2, dx2)
            d_angle = angle2 - angle1

            # Normalize angle difference to [-π, π]
            while d_angle > np.pi:
                d_angle -= 2 * np.pi
            while d_angle < -np.pi:
                d_angle += 2 * np.pi

            ds = np.sqrt(dx2**2 + dy2**2) + 1e-6  # Avoid division by zero
            curvature[i] = abs(d_angle) / ds

    # Calculate cumulative arc-length distances
    distances = np.zeros(n_points)
    if n_points > 1:
        for i in range(1, n_points):
            segment_length = np.linalg.norm(
                midline_pts[i] - midline_pts[i - 1])
            distances[i] = distances[i - 1] + segment_length

    return _LaneMidline(
        points=midline_pts,
        points_pixel=midline_pts_pix,
        heading_angles=heading_angles,
        curvature=curvature,
        distances=distances,
        valid=True
    )


def transform_points_to_original(
    points_warped: np.ndarray,
    inv_transformation_matrix: np.ndarray
) -> np.ndarray:
    """
    Transform points from bird's-eye view back to original perspective

    Args:
        points_warped: (N, 2) array of points in warped coordinates
        inv_transformation_matrix: 3x3 inverse perspective transform matrix

    Returns:
        (N, 2) array of points in original image coordinates
    """
    if len(points_warped) == 0:
        return np.array([])

    # Convert to homogeneous coordinates
    points_homogeneous = np.column_stack(
        [points_warped, np.ones(len(points_warped))])

    # Apply transformation
    points_transformed = (inv_transformation_matrix @ points_homogeneous.T).T

    # Convert back from homogeneous coordinates
    points_original = points_transformed[:, :2] / points_transformed[:, 2:3]

    return points_original


def visualize_boundaries_and_midline(
    frame: np.ndarray,
    left_boundary: _LaneBoundary,
    right_boundary: _LaneBoundary,
    midline: _LaneMidline,
    show_arrows: bool = True,
    arrow_spacing: int = 5
) -> np.ndarray:
    """
    Draw boundaries and midline on the frame for visualization

    Args:
        frame: Input frame to draw on
        left_boundary: Left lane boundary
        right_boundary: Right lane boundary
        midline: Calculated midline
        show_arrows: Whether to show direction arrows on midline
        arrow_spacing: Spacing between arrows (every Nth point)

    Returns:
        Frame with visualizations drawn
    """
    vis_frame = frame.copy()

    # Draw left boundary (blue)
    if left_boundary.valid and len(left_boundary.points_pixel) > 0:
        pts = left_boundary.points_pixel.astype(np.int32)
        cv2.polylines(vis_frame, [pts], False, (255, 0, 0), 2)
        # Draw confidence as text
        if len(pts) > 0:
            cv2.putText(vis_frame, f'L:{left_boundary.confidence:.2f}',
                        (pts[0, 0] - 50, pts[0, 1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 1)

    # Draw right boundary (red)
    if right_boundary.valid and len(right_boundary.points_pixel) > 0:
        pts = right_boundary.points_pixel.astype(np.int32)
        cv2.polylines(vis_frame, [pts], False, (0, 0, 255), 2)
        # Draw confidence as text
        if len(pts) > 0:
            cv2.putText(vis_frame, f'R:{right_boundary.confidence:.2f}',
                        (pts[0, 0] + 10, pts[0, 1]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 1)

    # Draw midline (green)
    if midline.valid and len(midline.points_pixel) > 0:
        pts = midline.points_pixel.astype(np.int32)
        cv2.polylines(vis_frame, [pts], False, (0, 255, 0), 3)

        # Draw direction arrows along midline
        if show_arrows and len(midline.heading_angles) > 0:
            for i in range(0, len(pts), arrow_spacing):
                if i >= len(midline.heading_angles):
                    break
                pt = pts[i]
                angle = midline.heading_angles[i]

                # Calculate arrow endpoint
                arrow_length = 15
                end_x = int(pt[0] + arrow_length * np.cos(angle))
                end_y = int(pt[1] + arrow_length * np.sin(angle))

                cv2.arrowedLine(vis_frame, tuple(pt), (end_x, end_y),
                                (0, 255, 0), 1, tipLength=0.3)

    return vis_frame
