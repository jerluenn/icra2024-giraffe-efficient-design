import time
from pymavlink import mavutil
import rospy
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose

class pymavlink_node: 

    def __init__(self): 

        rospy.init_node('uav_pose_node')

        self.initialisePublishers()
        self.initialiseSubscribers()
        self.mainLoop()
        rospy.spin()

    def initialisePublishers(self): 

        self.posePublisher = rospy.Publisher('/uav_position', Pose)

    def initialiseSubscribers(self): 

        # Set the connection parameters
        connection_string = '/dev/ttyUSB0'  # Replace with the appropriate connection string for your setup
        baud_rate = 57600  # Replace with the appropriate baud rate for your setup

        # Create the MAVLink connection
        self.master = mavutil.mavlink_connection(connection_string, baud=baud_rate)

        # Wait for the heartbeat message to establish connection
        self.master.wait_heartbeat()

        # Request data stream for the drone's local position
        self.master.mav.request_data_stream_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_DATA_STREAM_POSITION, 1, 1
        )

        self.poseMsg = Pose()

    def mainLoop(self): 

        while not rospy.is_shutdown():

            try:
                # Wait for the next message
                position_message = self.master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
                orientation_message = self.master.recv_match(type='ATTITUDE_QUATERNION', blocking=True)

                self.updateMsg(position_message, orientation_message)

                # Publish the local position
                self.posePublisher.publish(self.poseMsg)

            except Exception as e:
                print(f"Error: {str(e)}")
                continue

    def updateMsg(self, position_message, orientation_message):

        self.poseMsg.position.x = position_message.x
        self.poseMsg.position.y = position_message.y
        self.poseMsg.position.z = position_message.z
        self.poseMsg.orientation.w = orientation_message.q1
        self.poseMsg.orientation.x = orientation_message.q2
        self.poseMsg.orientation.y = orientation_message.q3
        self.poseMsg.orientation.z = orientation_message.q4

# Initialize the ROS node and publisher
rospy.init_node('drone_position_publisher', anonymous=True)
publisher_x = rospy.Publisher('/drone/local_position/x', Float64, queue_size=10)
publisher_y = rospy.Publisher('/drone/local_position/y', Float64, queue_size=10)
publisher_z = rospy.Publisher('/drone/local_position/z', Float64, queue_size=10)

# Set the connection parameters
connection_string = '/dev/ttyUSB0'  # Replace with the appropriate connection string for your setup
baud_rate = 57600  # Replace with the appropriate baud rate for your setup

# Create the MAVLink connection
master = mavutil.mavlink_connection(connection_string, baud=baud_rate)

# Wait for the heartbeat message to establish connection
master.wait_heartbeat()

# Request data stream for the drone's local position
master.mav.request_data_stream_send(
    master.target_system, master.target_component,
    mavutil.mavlink.MAV_DATA_STREAM_POSITION, 1, 1
)

# Main loop
while not rospy.is_shutdown():

    try:
        # Wait for the next message
        position_message = master.recv_match(type='LOCAL_POSITION_NED', blocking=True)
        orientation_message = master.recv_match(type='ATTITUDE_QUATERNION', blocking=True)

        # Extract the local position information
        x = position_message.x
        y = position_message.y
        z = position_message.z
        qw = orientation_message.q1
        qx = orientation_message.q2
        qy = orientation_message.q3
        qz = orientation_message.q4

        # Publish the local position
        publisher_x.publish(x)
        publisher_y.publish(y)
        publisher_z.publish(z)

    except Exception as e:
        print(f"Error: {str(e)}")
        continue

    time.sleep(0.1) 