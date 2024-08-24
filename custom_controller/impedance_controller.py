import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ServoCommandType
from geometry_msgs.msg import WrenchStamped, TwistStamped, Vector3
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R
import numpy as np
import math

class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')

        # Subscribe to the force-torque sensor data
        self.force_torque_subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.wrench_callback,
            10)

        # Compliance parameters
        self.Kp = 1.0  # Stiffness (inverse of compliance)
        self.Kd = 0.1  # Damping

        self.current_twist = TwistStamped()  # Keep track of the current twist for damping effect

        # TF Buffer and Listener to get transformations
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher for the computed Twist
        self.twist_pub_ = self.create_publisher(TwistStamped, '/servo_node/delta_twist_cmds', 10)

        # Set the rate to 500 Hz
        self.timer = self.create_timer(1.0 / 500.0, self.timer_callback)

        self.latest_wrench = None

        # Create a client for the ServoCommandType service
        self.switch_input_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')

        # Call the service to enable TWIST command type
        self.enable_twist_command()

    def enable_twist_command(self):
        if not self.switch_input_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Service not available, waiting again...')
            return

        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST

        future = self.switch_input_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Switched to input type: TWIST')
        else:
            self.get_logger().warn('Could not switch input to: TWIST')

    def wrench_callback(self, msg):
        self.latest_wrench = msg

    def timer_callback(self):
        if self.latest_wrench is not None:
            try:
                # Look up the transformation from ft_frame to tool0 and then tool0 to base_link
                ft_to_tool0 = self.tf_buffer.lookup_transform('tool0', self.latest_wrench.header.frame_id, rclpy.time.Time())
                # tool0_to_base_link = self.tf_buffer.lookup_transform('base_link', 'tool0', rclpy.time.Time())

                # Transform the force/torque from ft_frame to tool0
                force = self.transform_vector(ft_to_tool0, self.latest_wrench.wrench.force)
                torque = self.transform_vector(ft_to_tool0, self.latest_wrench.wrench.torque)

                # Transform the force/torque from tool0 to base_link
                # force = self.transform_vector(tool0_to_base_link, force)
                # torque = self.transform_vector(tool0_to_base_link, torque)

                # Nullify force/torque readings with magnitude < 3
                force = self.nullify_small_magnitudes(force, 3.0)
                torque = self.nullify_small_magnitudes(torque, 3.0)

                # Compute the twist in base_link frame
                twist = TwistStamped()
                twist.header.stamp = self.get_clock().now().to_msg()
                twist.header.frame_id = 'tool0'

                twist.twist.linear.x = (1 / self.Kp) * force.x - self.Kd * self.current_twist.twist.linear.x
                twist.twist.linear.y = (1 / self.Kp) * force.y - self.Kd * self.current_twist.twist.linear.y
                twist.twist.linear.z = (1 / self.Kp) * force.z - self.Kd * self.current_twist.twist.linear.z

                twist.twist.angular.x = (1 / self.Kp) * torque.x - self.Kd * self.current_twist.twist.angular.x
                twist.twist.angular.y = (1 / self.Kp) * torque.y - self.Kd * self.current_twist.twist.angular.y
                twist.twist.angular.z = (1 / self.Kp) * torque.z - self.Kd * self.current_twist.twist.angular.z

                # Update the current twist for the next callback
                self.current_twist = twist

                # Publish the computed twist
                self.twist_pub_.publish(twist)

            except (LookupException, ConnectivityException, ExtrapolationException) as e:
                self.get_logger().warn(f"Could not transform wrench to base_link frame: {str(e)}")

    def transform_vector(self, transform, vector):
        # Extract rotation (quaternion) and translation from TransformStamped
        q = transform.transform.rotation

        # Convert quaternion to rotation matrix using scipy
        r = R.from_quat([q.x, q.y, q.z, q.w])

        # Convert Vector3 to numpy array for easy multiplication
        vector_np = np.array([vector.x, vector.y, vector.z])

        # Apply the rotation
        rotated_vector = r.apply(vector_np)

        # Return the transformed vector as a Vector3
        return Vector3(x=rotated_vector[0], y=rotated_vector[1], z=rotated_vector[2])

    def nullify_small_magnitudes(self, vector, threshold):
        magnitude = math.sqrt(vector.x ** 2 + vector.y ** 2 + vector.z ** 2)
        if magnitude < threshold:
            return Vector3(x=0.0, y=0.0, z=0.0)
        else:
            return vector

def main(args=None):
    rclpy.init(args=args)
    impedance_controller = ImpedanceController()
    rclpy.spin(impedance_controller)
    impedance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
