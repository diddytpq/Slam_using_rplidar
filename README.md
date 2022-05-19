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

## Rplidar slam using gmapping in gazebo
```bash
roslaunch rplidar_slam rplidar_gazebo_env.launch
```

