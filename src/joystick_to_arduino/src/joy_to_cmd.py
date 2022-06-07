#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Joy

rospy.init_node("Joystick_to_Cmd_vel")
pub = rospy.Publisher("/cmd_vel",Twist, queue_size = 1)
r = rospy.Rate(100)

robot = Twist()

def callback(data):
	robot.linear.x = -0.20*data.axes[2]
	robot.angular.z = 1. * data.axes[1]
	pub.publish(robot)


sub = rospy.Subscriber("/joy",Joy,callback)


rospy.spin()
