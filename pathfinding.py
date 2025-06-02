#!/usr/bin/env python

# Author: Vincent Vey

# Import of python modules.
import math # use of pi.
import numpy as np # use of numpy for array manipulation
from collections import deque 
import cv2 # use for wall inflation

# import of relevant libraries.
import rclpy # module for ROS APIs
from rclpy.node import Node
from geometry_msgs.msg import Twist # message type for cmd_vel
from nav_msgs.msg import OccupancyGrid
from rclpy.duration import Duration 
from geometry_msgs.msg import Pose, PoseArray


import tf2_ros # library for transformations.
from tf_transformations import euler_from_quaternion
from tf_transformations import quaternion_from_euler


# Topic names
DEFAULT_CMD_VEL_TOPIC = 'cmd_vel'
DEFAULT_SCAN_TOPIC = 'base_scan'
DEFAULT_MAP_TOPIC = 'map'
DEFAULT_POSE_TOPIC = 'pose_sequence'

# Frequency at which the loop operates
FREQUENCY = 10 #Hz.

# Default Velocities that will be used 
LINEAR_VELOCITY = 0.125 # m/s
ANGULAR_VELOCITY = math.pi/4 # rad/s

USE_SIM_TIME = True

# constants
TF_BASE_LINK = 'rosbot/base_link'
TF_ODOM = 'rosbot/odom'
TF_MAP = 'map'

class GridNode():
    def __init__ (self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = float('inf')
        self.h = float('inf')
        self.f = float('inf')
        self.dir_from_parent = None 

    def __eq__(self, node):
        return self.x == node.x and self.y == node.y

    def __hash__(self):
        return hash((self.x, self.y))
    
    def __lt__(self, other):
        return self.f < other.f

class SearchMaze(Node):
    def __init__(self, linear_velocity=LINEAR_VELOCITY, angular_velocity=ANGULAR_VELOCITY,
        node_name="searchMaze", context=None):
        """Constructor."""
        super().__init__(node_name, context=context)

        # Workaround not to use roslaunch
        use_sim_time_param = rclpy.parameter.Parameter(
            'use_sim_time',
            rclpy.Parameter.Type.BOOL,
            USE_SIM_TIME
        )
        self.set_parameters([use_sim_time_param])

        # Setting up publishers/subscribers.
        # Setting up the publisher to send velocity commands.
        self._cmd_pub = self.create_publisher(Twist, DEFAULT_CMD_VEL_TOPIC, 1)
        self._pose_pub = self.create_publisher(PoseArray, DEFAULT_POSE_TOPIC, 1)

        self._map_sub = self.create_subscription(OccupancyGrid, DEFAULT_MAP_TOPIC, self._map_callback, 1)

        # Parameters.
        self.linear_velocity = linear_velocity # Constant linear velocity set.
        self.angular_velocity = angular_velocity # Constant angular velocity set.

        # Rate at which to operate the while loop.
        self.rate = self.create_rate(FREQUENCY)

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.is_map_received = False
        self.map_grid_msg = None

    def move(self, linear_vel, angular_vel):
        """Send a velocity command (linear vel in m/s, angular vel in rad/s)."""
        # Setting velocities.
        twist_msg = Twist()

        twist_msg.linear.x = linear_vel
        twist_msg.angular.z = angular_vel
        self._cmd_pub.publish(twist_msg)
    
    def stop(self):
        """Stop the robot."""
        twist_msg = Twist()
        self._cmd_pub.publish(twist_msg)

    def _map_callback(self, msg):
        self.get_logger().info("Received OccupancyGrid message")
        self.get_logger().info(f"Map size: {msg.info.width} x {msg.info.height}")
        self.get_logger().info(f"Resolution: {msg.info.resolution} m/pixel")

        self.map_grid_msg = msg

    def move_straight(self, distance, linear_vel=LINEAR_VELOCITY):
        """Move straight for a given distance."""
     
        self.get_logger().info(f"Moving straight for {distance} m.")
        duration = Duration(seconds=(distance / linear_vel))
        
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time) <= duration and rclpy.ok():
            self.move(linear_vel, 0.0)
            rclpy.spin_once(self)
        self.stop()

    def turn(self, angle, angular_vel=ANGULAR_VELOCITY):
        """Turn for a given angle."""

        rclpy.spin_once(self)
        self.get_logger().info(f"Turning for {angle} rad.")
        angular_vel = self.angular_velocity
        duration = Duration(seconds=(abs(angle) / angular_vel))
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time) <= duration and rclpy.ok():
            if (angle > 0):
                self.move(0.0, angular_vel)
            else:
                self.move(0.0, (-1 * angular_vel))

            rclpy.spin_once(self)

    def wait(self, duration):
        """Wait for a given duration."""
        start_time = self.get_clock().now()
        while (self.get_clock().now() - start_time).nanoseconds <= (duration * 1e9) and rclpy.ok():
            rclpy.spin_once(self)
        self.stop()

    def follow_path(self, waypoints):
        """Follow a path with given waypoints."""

        self.get_logger().info("Following path...")

        # Get the current position of the robot
        now = rclpy.time.Time()
        self.move_straight(0.01) # Trick to get the odom tf to update
        
        curr_x = waypoints[0][0]
        curr_y = waypoints[0][1]
        curr_theta = waypoints[0][2]

        resolution = 0.05 # 5cm resolution
        for (target_x, target_y, target_theta) in waypoints[1:]:
            # Calculate the difference between the target and current position
            
            dx = (target_x - curr_x) * resolution
            dy = (target_y - curr_y) * resolution
            d_theta = math.atan2(dy, dx) - curr_theta
            d_theta = (d_theta + math.pi) % (2 * math.pi) - math.pi # Normalize to [-pi, pi]

            dis_2_target = math.hypot(dy, dx)

            # Move to target
            self.turn(d_theta)
            self.move_straight(dis_2_target)
            
            # Update current position
            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                TF_MAP,       # target_frame
                TF_BASE_LINK,  # source_frame
                now
            )

            curr_x = (transform.transform.translation.x)/resolution
            curr_y = (transform.transform.translation.y)/resolution
            curr_q = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            _, _, curr_theta = euler_from_quaternion(curr_q)

        self.get_logger().info("Done drawing polygon.")

    def inflate_obstacles(self, map_grid, robot_radius_m, resolution):
        # Convert to binary: 1 for obstacle, 0 for free
        binary_map = (map_grid > 0).astype(np.uint8)

        # Calculate kernel size: how many pixels to inflate based on robot size
        inflation_radius_cells = int(np.ceil(robot_radius_m / resolution))
        kernel_size = 2 * inflation_radius_cells + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

        # Inflate the map
        inflated = cv2.dilate(binary_map, kernel)

        # Return inflated map: mark inflated cells as occupied (e.g., 100), rest as original
        inflated_map = np.where(inflated == 1, 100, 0)
        return inflated_map
    
    def backchain(self, child_node):
        "Backchaning from target to start"
        
        solution = []
        node = child_node
        while node:
            # Calculate the angle from parent to current node
            theta = 0.0
            if node.parent:
                dx = node.x - node.parent.x
                dy = node.y - node.parent.y
                theta = math.atan2(dy, dx)
                theta = (theta + math.pi) % (2 * math.pi) - math.pi

            solution.insert(0, (node.x, node.y, theta))
            node = node.parent

        return solution
    
    def angle_between(self, v1, v2):
        """Return angle in radians between two 2D vectors."""
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        mag1 = math.hypot(*v1)
        mag2 = math.hypot(*v2)
        if mag1 == 0 or mag2 == 0:
            return 0
        cos_angle = dot / (mag1 * mag2)
        angle = math.acos(max(min(cos_angle, 1), -1))
        return angle


    def a_star(self, start, goal, map_grid):
        def h(node, goal):
            return math.hypot(node.x - goal.x, node.y - goal.y)

        map_height = len(map_grid)
        map_width = len(map_grid[0]) if map_height > 0 else 0

        open_set = []
        closed_set = set()
        node_map = {}

        startx, starty = start
        goalx, goaly = goal

        start_node = GridNode(startx, starty)
        goal_node = GridNode(goalx, goaly)

        start_node.g = 0
        start_node.h = h(start_node, goal_node)
        start_node.f = start_node.h
        open_set.append(start_node)
        node_map[(startx, starty)] = start_node

        directions = [
            (1, 0), (0, -1), (-1, 0), (0, 1),          # cardinal
            (1, 1), (-1, -1), (-1, 1), (1, -1),        # diagonal
            (2, 1), (1, 2), (-2, 1), (-1, 2),
            (-2, -1), (-1, -2), (2, -1), (1, -2),
            (3, 1), (1, -3), (-3, 1), (1, 3),
            (-3, -1), (-1, -3), (3, -1), (-1, 3),
            (4, 1), (1, -4), (-4, 1), (1, 4),
            (-4, -1), (-1, -4), (4, -1), (-1, 4),
            (5, 1), (1, -5), (-5, 1), (1, 5),
            (-5, -1), (-1, -5), (5, -1), (-1, 5),
            (6, 1), (1, -6), (-6, 1), (1, 6),
            (-6, -1), (-1, -6), (6, -1), (-1, 6),
            (7, 1), (1, -7), (-7, 1), (1, 7),
            (-7, -1), (-1, -7), (7, -1), (-1, 7),
            (8, 1), (1, -8), (-8, 1), (1, 8),
            (-8, -1), (-1, -8), (8, -1), (-1, 8)
        ]

        while open_set:
            curr_node = min(open_set, key=lambda node: node.f)

            if (curr_node.x, curr_node.y) == (goalx, goaly):
                self.get_logger().info("Found goal!")
                return self.backchain(curr_node)

            open_set.remove(curr_node)
            closed_set.add((curr_node.x, curr_node.y))

            for dx, dy in directions:
                nx, ny = curr_node.x + dx, curr_node.y + dy

                if not (0 <= nx < map_width and 0 <= ny < map_height): # bounds check
                    continue
                if map_grid[ny][nx] != 0: # check if the cell is free
                    continue
                if (nx, ny) in closed_set: # already evaluated
                    continue

                # Prevent corner-cutting
                if dx != 0 and dy != 0:
                    if map_grid[curr_node.y][curr_node.x + dx] != 0 or map_grid[curr_node.y + dy][curr_node.x] != 0:
                        continue

                direction = (dx, dy)
                direction_change_penalty = 0
                if curr_node.dir_from_parent is not None:
                    angle = self.angle_between(curr_node.dir_from_parent, direction)
                    direction_change_penalty = angle * 50  # angle penalty factor

                movement_cost = math.hypot(dx, dy)
                tentative_g = curr_node.g + movement_cost + direction_change_penalty

                if (nx, ny) in node_map:
                    neighbor = node_map[(nx, ny)]
                    if tentative_g < neighbor.g:
                        neighbor.g = tentative_g
                        neighbor.f = neighbor.g + neighbor.h
                        neighbor.parent = curr_node
                        neighbor.dir_from_parent = direction
                else:
                    neighbor = GridNode(nx, ny)
                    neighbor.g = tentative_g
                    neighbor.h = h(neighbor, goal_node)
                    neighbor.f = neighbor.g + neighbor.h
                    neighbor.parent = curr_node
                    neighbor.dir_from_parent = direction
                    node_map[(nx, ny)] = neighbor
                    open_set.append(neighbor)

        self.get_logger().info("No path found to goal.")
        return None


    def publish_path(self, path):
        """Publish the path as a PoseArray."""
        pose_arr = PoseArray()
        pose_arr.header.frame_id = 'map'
        pose_arr.header.stamp = self.get_clock().now().to_msg()
        resolution = 0.05  # 5cm resolution

        for (x, y, theta) in path:
            pose = Pose()
            pose.position.x = x * resolution
            pose.position.y = y * resolution
            pose.position.z = 0.0

            # Convert theta (yaw) to quaternion
            q = quaternion_from_euler(0, 0, theta)
            pose.orientation.x = q[0]
            pose.orientation.y = q[1]
            pose.orientation.z = q[2]
            pose.orientation.w = q[3]

            pose_arr.poses.append(pose)

        self._pose_pub.publish(pose_arr)
        self.get_logger().info(f"Published path with {len(pose_arr.poses)} waypoints.")

    def get_curr_pose(self):
        """Get the current pose of the robot in the map frame."""
        try:
            self.move_straight(0.01) # Trick to get the odom tf to update
            self.wait(1) # Wait for the transform to be available

            now = rclpy.time.Time()
            transform = self.tf_buffer.lookup_transform(
                TF_MAP,       # target_frame
                TF_BASE_LINK,  # source_frame
                now
            )
  
            resolution = 0.05 # 5cm resolution

            curr_x = transform.transform.translation.x / resolution
            curr_y = transform.transform.translation.y / resolution
            curr_q = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
            _, _, curr_theta = euler_from_quaternion(curr_q)
                   
            return curr_x, curr_y, curr_theta
        
        except tf2_ros.TransformException as e:
            self.get_logger().warn(f"Could not transform: {e}")
            return None

    def prompt_and_run(self):
        """Prompt user and run pathfinding"""
        print("hello world")
        while rclpy.ok():
            if self.map_grid_msg is None:
                self.get_logger().info("Waiting for map...")
                rclpy.spin_once(self)
                continue

            try:
                input_str = input("Enter goal coordinate (x, y) with x and y between 0 and 200: ")
                x_str, y_str = input_str.strip("() ").split(",")
                x = int(x_str)
                y = int(y_str)
                if not (0 <= x <= 200 and 0 <= y <= 200):
                    print("Both x and y must be between 0 and 200. Please try again.")
                    continue

                goal = (x, y)

                pose = self.get_curr_pose()
                if pose is None:
                    self.get_logger().warn("Could not get current pose. Skipping...")
                    continue

                curr_x, curr_y, theta = pose
                self.get_logger().info(f"Current pose: x={curr_x}, y={curr_y}, theta={theta}")

                map_grid = np.array(self.map_grid_msg.data).reshape((self.map_grid_msg.info.height, self.map_grid_msg.info.width))
                map_grid = self.inflate_obstacles(map_grid, robot_radius_m=0.35, resolution=0.05)
                solution = self.a_star((int(curr_x), int(curr_y)), (x, y), map_grid)

                self.publish_path(solution)
                
                if solution is None:
                    self.get_logger().warn("No solution found.")
                    continue

                self.get_logger().info(f"Solution: {solution}")
                self.follow_path(solution)
                self.get_logger().info("Done following path.")

            except ValueError:
                print("Invalid input format. Please enter in the form: x, y") 

def main(args=None):
    """Main function."""

    # 1st. initialization of node.
    rclpy.init(args=args)
    search_maze = SearchMaze()

    interrupted = False

    try:
        rclpy.spin_once(search_maze)
        search_maze.move_straight(0.01) # Trick to get the odom tf to update
        search_maze.prompt_and_run()
        rclpy.spin_once(search_maze)
    
    except KeyboardInterrupt:
        interrupted = True
        search_maze.get_logger().error("ROS node interrupted.")
    finally:
        if rclpy.ok():
            search_maze.stop()

    if interrupted:
        new_context = rclpy.Context()
        rclpy.init(context=new_context)
        search_maze = SearchMaze(node_name="search_maze_end", context=new_context)
        search_maze.get_logger().error("ROS node interrupted.")
        search_maze.stop()
        rclpy.try_shutdown()


if __name__ == "__main__":
    """Run the main function."""
    main()
