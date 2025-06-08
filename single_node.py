import rclpy
from pathfinding import SearchMaze
from project_mapper_node import MapperNode
from rclpy.executors import MultiThreadedExecutor


def main():
    rclpy.init()

    mapper_node = MapperNode()
    search_node = SearchMaze()

    executor = MultiThreadedExecutor()
    executor.add_node(mapper_node)
    executor.add_node(search_node)

    try:
        print("[INFO] Spinning mapper and search nodes together...")
        executor.spin()
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        mapper_node.destroy_node()
        search_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
