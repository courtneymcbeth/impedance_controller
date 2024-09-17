import rclpy
from rclpy.node import Node
from moveit_msgs.srv import ServoCommandType
from geometry_msgs.msg import WrenchStamped, TwistStamped, Vector3
from tf2_ros import Buffer, TransformListener, LookupException, ConnectivityException, ExtrapolationException
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
from std_srvs.srv import Trigger
from controller_manager_msgs.srv import SwitchController
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool
import time



class ServoController(Node):
    def __init__(self):
        super().__init__('servo_controller')

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
        self.wrench_timer = self.create_timer(1.0 / 500.0, self.wrench_timer_callback)

        # self.vel_timer = self.create_timer(1.0 / 500.0, self.vel_timer_callback)
        self.prev_interaction = False
        self.prev_interaction_time = None
        self.interaction = False
        self.interaction_msg = Bool()
        self.interaction_pub = self.create_publisher(Bool, '/interaction', 10)

        self.latest_wrench = None

        self.switch_controller_client = self.create_client(SwitchController, '/controller_manager/switch_controller')
        self.deactivate_controller('scaled_joint_trajectory_controller')

        # Forward Velocity Controller
        # self.activate_controller('forward_position_controller')
        self.activate_controller('forward_velocity_controller')
        
        # Create a client for the ServoCommandType service
        self.switch_input_client = self.create_client(ServoCommandType, '/servo_node/switch_command_type')
        # Call the service to enable TWIST command type
        self.enable_twist_command()
        
        self.joint_state_sub = self.create_subscription(JointState, '/joint_states', self.joint_state_callback, 10)
        self.joint_torque_pub = self.create_publisher(JointState, '/joint_torques', 10)

        self.interaction_pub_timer = self.create_timer(0.01, self.interaction_callback)


    def activate_controller(self, controller_name):
        if not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Switch control sensor service not available, waiting again...')
            return
         
        request = SwitchController.Request()
        request.activate_controllers = [controller_name]

        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().ok:
            self.get_logger().info(f'Activated controller: {controller_name}')
        else:
            self.get_logger().warn(f'Could not activate controller: {controller_name}')


    def deactivate_controller(self, controller_name):
        if not self.switch_controller_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Switch control service not available, waiting again...')
            return
        
        request = SwitchController.Request()
        request.deactivate_controllers = [controller_name]

        future = self.switch_controller_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().ok:
            self.get_logger().info(f'Deactivated controller: {controller_name}')
        else:
            self.get_logger().warn(f'Could not deactivate controller: {controller_name}')
            
    
    def enable_twist_command(self):
        if not self.switch_input_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Enable twist command service not available, waiting again...')
            return

        request = ServoCommandType.Request()
        request.command_type = ServoCommandType.Request.TWIST

        future = self.switch_input_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        
        if future.result() is not None and future.result().success:
            self.get_logger().info('Switched to input type: TWIST')
        else:
            self.get_logger().warn('Could not switch input to: TWIST')
            
            
    def zero_ft_sensor(self):
        if not self.zero_ft_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('Zero ft sensor service not available, waiting again...')
            return
        
        request = Trigger.Request()
        future = self.zero_ft_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None and future.result().success:
            self.get_logger().info('Zero ft sensor complete!')
        else:
            self.get_logger().warn("Could not zero ft sensor!")
            
            
    def joint_state_callback(self, msg):
        joint_torque_msg = JointState()
        # Convert joint current to torque using UR5e torque constants
        joint_torque_msg.effort = np.roll(np.array(msg.effort), 1) * [10.0, 10.0, 10.0, 2.0, 2.0, 2.0] # * [0.125, 0.125, 0.125, 0.092, 0.092, 0.092]
        self.joint_torque_pub.publish(joint_torque_msg)

    def wrench_callback(self, msg):
        self.latest_wrench = msg

    def interaction_callback(self):
        self.interaction_msg.data = self.interaction
        if self.interaction:
            self.get_logger().info('Interaction!')
            self.interaction_pub.publish(self.interaction_msg)
        else:
            if self.prev_interaction_time is not None:
                time_diff = time.time() - self.prev_interaction_time
                if time_diff > 2.0:
                    self.get_logger().info('No Interaction')
                    self.interaction_pub.publish(self.interaction_msg)
                    self.prev_interaction_time = None
        # self.prev_interaction = self.interaction

    # def vel_timer_callback(self):
    #     if self.interaction == True and self.prev_interaction == False:
    #         self.activate_controller('forward_position_controller')
    #     elif self.interaction == False and self.prev_interaction == True :
    #         self.activate_controller('forward_velocity_controller')


    def wrench_timer_callback(self):
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

                self.prev_interaction = self.interaction

                if math.sqrt(force.x ** 2 + force.y ** 2 + force.z ** 2) < 10.0:
                    self.interaction = False
                    if self.prev_interaction == True and self.prev_interaction_time is None:
                        self.prev_interaction_time = time.time()
                    return

                self.interaction = True

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
    servo_controller = ServoController()
    rclpy.spin(servo_controller)
    servo_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
