# Slam_using_rplidar

## lidar port check
```bash
ls /dev/ttyUSB* 
```
## give port permission
```bash
sudo chmod 666 /dev/ttyUSB0
```

## view lidar sensor value using rviz
```bash
roslaunch rplidar_ros view_rplidar.launch
```
