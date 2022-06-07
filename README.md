# Slam_using_rplidar

## Lidar port check
```bash
ls /dev/ttyUSB* 
```

## Give port permission
```bash
sudo chmod 666 /dev/ttyUSB0
```

## View lidar sensor value using rviz
```bash
roslaunch rplidar_ros view_rplidar.launch
```

## Rplidar slam using gmapping
```bash
roslaunch rplidar_slam rplidar_gmapping.launch
```

## Rplidar slam using gmapping on real robot
```bash
roslaunch rplidar_slam real_robot.launch
```

## Rplidar slam using gmapping in gazebo
```bash
roslaunch rplidar_slam rplidar_gazebo_env.launch
```




## record && check ros bag data
```bash
rosbag record -a

roscore
rosparam set /use_sim_time true
rosbag play --clock --pause 2022-06-05-16-50-00.bag --topics /cur_vel /scan /cmd_vel
roslaunch rplidar_slam check_ros_bag.launch 
```

