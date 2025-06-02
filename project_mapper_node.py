import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
import tf_transformations
import tf2_ros
from geometry_msgs.msg import TransformStamped
import numpy as np

class MapperNode(Node):
    def __init__(self):
        super().__init__('project_mapper_node')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.scan_sub = self.create_subscription(
            LaserScan,
            '/scan',
            self.laser_callback,
            10
        )

        self.map_pub = self.create_publisher(OccupancyGrid, '/map', 10)

        self.grid_size = 100
        self.resolution = 0.1
        self.map = np.full((self.grid_size, self.grid_size), -1, dtype=np.int8)

        self.get_logger().info('MapperNode initialized.')

    def world_to_map(self, x, y):
        i = int((x + (self.grid_size * self.resolution / 2)) / self.resolution)
        j = int((y + (self.grid_size * self.resolution / 2)) / self.resolution)
        return i, j

    def bresenham(self, x0, y0, x1, y1):
        x0 = int(round(x0))
        y0 = int(round(y0))
        x1 = int(round(x1))
        y1 = int(round(y1))

        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        x, y = x0, y0
        sx = -1 if x0 > x1 else 1
        sy = -1 if y0 > y1 else 1

        if dx > dy:
            err = dx / 2.0
            while x != x1:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2.0
            while y != y1:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x1, y1))
        return points

    def publish_map(self):
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'odom'
        msg.info.resolution = self.resolution
        msg.info.width = self.grid_size
        msg.info.height = self.grid_size
        msg.info.origin.position.x = -self.grid_size * self.resolution / 2
        msg.info.origin.position.y = -self.grid_size * self.resolution / 2
        msg.info.origin.position.z = 0.0
        msg.info.origin.orientation.w = 1.0
        msg.data = self.map.flatten().tolist()
        self.map_pub.publish(msg)

        self.get_logger().info(f"Published map: free={(self.map == 0).sum()}, occ={(self.map == 100).sum()}")

    def laser_callback(self, msg: LaserScan):
        self.get_logger().info('Laser callback triggered')

        try:
            now = rclpy.time.Time()
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                'odom',  
                'laser',
                now,
                timeout=rclpy.duration.Duration(seconds=1.0)
            )

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            yaw = tf_transformations.euler_from_quaternion([
                transform.transform.rotation.x,
                transform.transform.rotation.y,
                transform.transform.rotation.z,
                transform.transform.rotation.w
            ])[2]

            angle = msg.angle_min
            for r in msg.ranges:
                if np.isinf(r) or np.isnan(r):
                    angle += msg.angle_increment
                    continue

                lx = r * np.cos(angle)
                ly = r * np.sin(angle)

                gx = tx + (np.cos(yaw) * lx - np.sin(yaw) * ly)
                gy = ty + (np.sin(yaw) * lx + np.cos(yaw) * ly)

                robot_i, robot_j = self.world_to_map(tx, ty)
                goal_i, goal_j = self.world_to_map(gx, gy)

                for i, j in self.bresenham(robot_i, robot_j, goal_i, goal_j):
                    if 0 <= i < self.grid_size and 0 <= j < self.grid_size:
                        self.map[j, i] = 0  # Free

                if 0 <= goal_i < self.grid_size and 0 <= goal_j < self.grid_size:
                    self.map[goal_j, goal_i] = 100  # Occupied

                angle += msg.angle_increment

            self.publish_map()

        except Exception as e:
            self.get_logger().warn(f'Transform error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down MapperNode.')
    finally:
        node.destroy_node()
        rclpy.shutdown()
