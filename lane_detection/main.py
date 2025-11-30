import cv2
import numpy as np
try:
    import edge_detection as edge
except ImportError:
    from . import edge_detection as edge
import matplotlib.pyplot as plt
import argparse
import time
try:
    from lane_data import LaneDetectionResult
    from lane_utils import extract_boundary_points, calculate_midline, visualize_boundaries_and_midline
except ImportError:
    from .lane_data import LaneDetectionResult
    from .lane_utils import extract_boundary_points, calculate_midline, visualize_boundaries_and_midline

# Author: Addison Sears-Collins
# https://automaticaddison.com
# Description: Implementation of the Lane class

def process_image(image_path, output_path=None, config=None):
    """
    Process a single image for lane detection

    :param image_path: Path to the input image
    :param output_path: Optional path to save the output image
    :param config: LaneConfig object (optional)
    """
    # Load the image
    original_frame = cv2.imread(image_path)

    if original_frame is None:
        print(f"Error: Could not load image from {image_path}")
        return

    # Process the frame
    frame, result = process_frame(
        original_frame, config, return_detection_result=True)
    print("Lane Detection Result:")
    print(result)
    # Save output if path provided
    if output_path:
        cv2.imwrite(output_path, frame)
        print(f"Output saved to {output_path}")

    # Display the image
    cv2.imshow("Lane Detection Result", frame)
    cv2.waitKey(0)

    empty_frame = np.zeros_like(frame)
    visualization = visualize_boundaries_and_midline(
        empty_frame,
        result.left_boundary,
        result.right_boundary,
        result.midline
    )
    cv2.imshow("Lane Detection Result", visualization)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def process_video(video_path, output_path=None, config=None):
    """
    Process a video file for lane detection

    :param video_path: Path to the input video
    :param output_path: Optional path to save the output video
    :param config: LaneConfig object (optional)
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video from {video_path}")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Initialize video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (frame_width, frame_height))

    print("Processing video... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Process the frame
        frame, lane_detection_result = process_frame(
            frame, config, return_detection_result=True)

        # Write to output video if specified
        if out:
            out.write(frame)

        if lane_detection_result.valid:
            empty_frame = np.zeros_like(frame)
            visualization = visualize_boundaries_and_midline(
                empty_frame,
                lane_detection_result.left_boundary,
                lane_detection_result.right_boundary,
                lane_detection_result.midline
            )
            cv2.imshow("Lane Boundaries and Midline", visualization)

        cv2.imshow("Lane Detection - Video", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if out:
        out.release()
        print(f"Output video saved to {output_path}")
    cv2.destroyAllWindows()


def process_camera(camera_id=0, output_path=None, config=None):
    """
    Process live camera feed for lane detection

    :param camera_id: Camera device ID (default: 0)
    :param output_path: Optional path to save the output video
    :param config: LaneConfig object (optional)
    """
    # Open the camera
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Error: Could not open camera with ID {camera_id}")
        return

    # Get camera properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 20  # Set a default fps for recording

    # Initialize video writer if output path provided
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps,
                              (frame_width, frame_height))

    print("Processing camera feed... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame from camera")
            break

        # Process the frame
        result = process_frame(frame, config)

        # Write to output video if specified
        if out:
            out.write(result)

        # Display the result
        cv2.imshow("Lane Detection - Camera", result)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    if out:
        out.release()
        print(f"Output video saved to {output_path}")
    cv2.destroyAllWindows()


def main():
    """
    Main function with argument parsing
    """
    parser = argparse.ArgumentParser(description='Lane Detection System')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'camera'],
                        default='image', help='Processing mode: image, video, or camera')
    parser.add_argument('--input', type=str,
                        help='Input file path (for image or video mode)')
    parser.add_argument('--output', type=str,
                        help='Output file path (optional)')
    parser.add_argument('--camera-id', type=int, default=0,
                        help='Camera device ID (default: 0, for camera mode)')
    parser.add_argument('--config', type=str,
                        help='Path to custom configuration file (optional)')

    args = parser.parse_args()

    # Load configuration if provided
    config = LaneConfig()
    if args.config:
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location(
                "custom_config", args.config)
            custom_config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(custom_config_module)

            # Override default config values
            for attr in dir(custom_config_module):
                if not attr.startswith('_') and hasattr(config, attr):
                    setattr(config, attr, getattr(custom_config_module, attr))
            print(f"Loaded custom configuration from {args.config}")
        except Exception as e:
            print(f"Warning: Could not load config file {args.config}: {e}")
            print("Using default configuration")

    if args.mode == 'image':
        if not args.input:
            print("Error: --input is required for image mode")
            parser.print_help()
            return
        process_image(args.input, args.output, config)

    elif args.mode == 'video':
        if not args.input:
            print("Error: --input is required for video mode")
            parser.print_help()
            return
        process_video(args.input, args.output, config)

    elif args.mode == 'camera':
        process_camera(args.camera_id, args.output, config)


if __name__ == '__main__':
    main()
