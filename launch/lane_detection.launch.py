from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():    
    # Default parameter file path
    default_params_file = PathJoinSubstitution([
        FindPackageShare('lane_detection'),
        'config',
        'params.yaml'
    ])
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to the parameter file'
    )
    
    camera_topic_arg = DeclareLaunchArgument(
        'camera_topic',
        default_value='/camera/image_raw',
        description='Camera image topic to subscribe to'
    )
    
    visualization_arg = DeclareLaunchArgument(
        'publish_visualization',
        default_value='true',
        description='Whether to publish visualization images'
    )
    
    # Lane detection node
    lane_detection_node = Node(
        package='lane_detection',
        executable='lane_detection_node',
        name='lane_detection_node',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'camera_topic': LaunchConfiguration('camera_topic'),
                'publish_visualization': LaunchConfiguration('publish_visualization'),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ]
    )
    
    return LaunchDescription([
        params_file_arg,
        camera_topic_arg,
        visualization_arg,
        lane_detection_node,
    ])
