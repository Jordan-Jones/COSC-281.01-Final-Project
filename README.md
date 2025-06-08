# COSC-281.01-Final-Project: Multi-Robot Communication for Object Transportation
## Stephen Adjei, Jordan Jones, Vuthy Vey, Zachary Farris
In this COSC 81 Final Project, our group aims to develop a multi-ROSbot system capable of collaboratively transporting objects in an unknown environment, using RGB camera  data, LiDAR sensors, and OpenCV. This system will enable robots to map their surroundings, share information, and coordinate to deliver objects to designated goals through optimal paths

## Requirements
- ROS2 -- tested on ROS2 humble, but other versions may work.

## Run
Terminal 1 – Run docker with ```docker compose up```

Terminal 2 – Visualize the map with ```ros2 run rviz2 rviz2```

Terminal 3 – Move into the directory of the ptyhon files and run each accordingly, example:
```python3 color_detection_node.py```