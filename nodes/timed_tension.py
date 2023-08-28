#!/usr/bin/env python

import sys
import numpy as np
import time
from casadi import *
from pyquaternion import Quaternion
import rospy

from sensor_msgs.msg import Joy
from std_msgs.msg import Float64MultiArray


class TimeTension: 

    def __init__(self, tension, time_): 

        """Tension in terms of """

        self._time_required = time_ 
        self._tension_message = Float64MultiArray()
        self._tension_message.data = tension
        self.initpubsub()
        self.main_loop()

        rospy.spin()

    def initpubsub(self): 

        rospy.init_node('time_tension_node', anonymous=True)
        self.ref_tensions_publisher = rospy.Publisher('/ref_tensions', Float64MultiArray, queue_size=1)
        print("time_tension_node initalised.")

    def main_loop(self):

        t0 = time.time()

        while not rospy.is_shutdown(): 

            while time.time() - t0 < self._time_required: 

                self.ref_tensions_publisher.publish(self._tension_message)
                time.sleep(1e-2)
                
            self._tension_message.data = np.zeros(4)
            self.ref_tensions_publisher.publish(self._tension_message)

if __name__ == "__main__":

    tension = np.array([0, 0, 3000, 0])

    TimeTension(tension, 30)
