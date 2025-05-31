#!/usr/bin/env python

"""

    Color Detection Node

    Author: Stephen A
    Date: 5-26-25
    Based on: Example ROS Node (https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython)

    Sources Used:
    https://github.com/ros-perception/vision_opencv
    https://wiki.ros.org/cv_bridge
    https://www.geeksforgeeks.org/choosing-the-correct-upper-and-lower-hsv-boundaries-for-color-detection-with-cv-inrange-opencv/
    https://wiki.ros.org/cv_bridge/Tutorials/ConvertingBetweenROSImagesAndOpenCVImagesPython
    https://github.com/sabrinamkb/ROSbot-color-detection

"""

import math
import time
import numpy as np

import rclpy # module for ROS APIs
from rclpy.node import Node

# http://docs.ros.org/en/noetic/api/nav_msgs/html/msg/OccupancyGrid.html
from nav_msgs.msg import OccupancyGrid, MapMetaData
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseArray
from sensor_msgs.msg import LaserScan # message type for laser measurement.
from geometry_msgs.msg import Twist # message type for cmd_vel
from std_msgs.msg import Header

# Libraries for listeners for static_transform_publisher
import tf2_ros
import tf_transformations
from geometry_msgs.msg import TransformStamped
from tf2_ros import TransformException, Buffer, TransformListener

# Color Detection Essential Libraries
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import cv2

# Topic names
NODE_NAME = "color_detection"

MAP_TOPIC = "map"
IMAGE_PUBLISHER_TOPIC = "/camera/color/image_processed"
IMAGE_SUBSCRIBER_TOPIC = "/camera/color/image_raw"
COLOR_DETECTION_TOPIC = "/color_detection/detected_color"

OCCUPANCY_TOPIC = "occupancy_grid"
POSE_TOPIC = "pose"
POSE_SEQUENCE_TOPIC = "pose_sequence"
DEFAULT_SCAN_TOPIC = 'base_scan' # name of topic for Stage simulator. For Gazebo, 'scan'
USE_SIM_TIME = True

COLOR_OPTION = -1

BLUE_UPPER = (140, 255, 255)
BLUE_LOWER = (100, 150, 0)

RED_UPPER = (10, 255, 255)
RED_LOWER = (0, 100, 100)

GREEN_UPPER = (86, 255, 255)
GREEN_LOWER = (36, 100, 100)

YELLOW_UPPER = (30, 255, 255)
YELLOW_LOWER = (20, 100, 100)

LOW_H = 0
LOW_S = 0
LOW_V = 0

UP_H = 180
UP_S = 256
UP_V = 256

class ColorDetection(Node):
    def __init__(self, node_name=NODE_NAME, context=None):
        super().__init__(node_name, context=context)

        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )

        self.image_publisher = self.create_publisher(Image, IMAGE_PUBLISHER_TOPIC, 1)
        self.image_subscriber = self.create_subscription(Image, IMAGE_SUBSCRIBER_TOPIC, self.image_callback, 1)
        self.color_publisher = self.create_publisher(String, COLOR_DETECTION_TOPIC, 1)
        self.bridge = CvBridge()

        self.red = False
        self.blue = False
        self.green = False
        self.yellow = False

    def stop(self):
        pass
    
    def image_callback(self, msg):

        global COLOR_OPTION

        self.color_option = 0  
        self.color_msg = String()

        try:
            # print("Recieved Image...")
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except CvBridgeError as e:
            print("Error cvBridge: ", e)

        # Show image in window for testing purposes
        (rows, cols, channels) = cv_image.shape

        if cols > 60 and rows > 60 :
            cv2.circle(cv_image, (50, 50), 10, 255)

        cv2.imshow("Image window", cv_image)
        cv2.waitKey(1)  # Add this line
        
        # Convert BGR image to HSV Image
        hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Create masks
        blue_mask = cv2.inRange(hsv_image, BLUE_LOWER, BLUE_UPPER)
        red_mask = cv2.inRange(hsv_image, RED_LOWER, RED_UPPER)
        green_mask = cv2.inRange(hsv_image, GREEN_LOWER, GREEN_UPPER)
        yellow_mask = cv2.inRange(hsv_image, YELLOW_LOWER, YELLOW_UPPER)

        # Process result mask representing best fit color
        result_mask = cv2.bitwise_or(red_mask, blue_mask)
        result_mask = cv2.bitwise_or(result_mask, green_mask)
        result_mask = cv2.bitwise_or(result_mask, yellow_mask)

        # Set COLOR_OPTION boolean
        if np.any(red_mask):
            print("RED DETECTED")
            COLOR_OPTION = 0
            self.color_msg.data = "RED"
        elif np.any(green_mask):
            print("GREEN DETECTED")
            COLOR_OPTION = 1
            self.color_msg.data = "GREEN"
        elif np.any(blue_mask):
            print("BLUE DETECTED")
            COLOR_OPTION = 2
            self.color_msg.data = "BLUE"
        elif np.any(yellow_mask):
            print("YELLOW DETECTED")
            COLOR_OPTION = 3
            self.color_msg.data = "YELLOW"
        else:
            COLOR_OPTION = -1

        # Publish resulting Mask.
        try:
            self.image_publisher.publish(self.bridge.cv2_to_imgmsg(result_mask, encoding="mono8"))
            self.color_publisher.publish(self.color_msg)
        except CvBridgeError as e:
            print("Error cvBridge: ", e)
        
    def spin(self):
        # Simply spin the ROS node. 
        # The logic within this program lies within the lidar_callback() function
        while rclpy.ok():
            try:
                # print('Spin...')
                rclpy.spin(self)
            except Exception as e:
                print(f'Waiting for Published Transform... {e}')
                time.sleep(1)


def main(args=None):

    # Initiate and spin the ColorDetection() object
    rclpy.init(args=args)
    color = ColorDetection()
    interrupted = False

    try:
        # Spin mapper
        color.spin()
    except KeyboardInterrupt:
        interrupted = True
        color.get_logger().error("ROS node interrupted.")

    if interrupted:
        new_context = rclpy.Context()
        rclpy.init(context=new_context)
        color = ColorDetection(node_name="color_detection_end", context=new_context)
        color.get_logger().error("ROS node interrupted.")
        color.stop()
        rclpy.try_shutdown()
    
if __name__ == "__main__":
    main()
