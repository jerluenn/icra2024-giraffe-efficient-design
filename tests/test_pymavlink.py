import time
from pymavlink import mavutil

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

t = time.time()  

# Main loop
while True:
    try:

        message = master.recv_msg()

        if message:


            if message.get_type() == 'SERVO_OUTPUT_RAW':
                # Extract the local position information
                print(message)
                print("Time taken for every loop: ", time.time() - t)
                t = time.time()   

        time.sleep(0.001)

    except KeyboardInterrupt:
        # Close the connection and exit the script
        master.close()
        break

    except Exception as e:
        print(f"Error: {str(e)}")
        continue

    # time.sleep(0.1)  # Adjust the sleep time as per your requirement
