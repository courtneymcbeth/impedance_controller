import rclpy
from rclpy.node import Node
from sensor_msgs.msg import WrenchStamped
from geometry_msgs.msg import Twist
import numpy as np
from scipy.linalg import expm

class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')
        
        # Subscribe to the force-torque sensor data
        self.subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor',
            self.force_torque_callback,
            10)
        self.subscription  # prevent unused variable warning

        # Publisher for the desired velocity
        self.velocity_publisher = self.create_publisher(Twist, '/desired_velocity', 10)
        
        # Impedance control parameters
        self.mass_matrix = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.damping_matrix = np.diag([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        self.stiffness_matrix = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        # Desired position and velocity (target state)
        self.desired_position = np.zeros(6)
        self.desired_velocity = np.zeros(6)

        # Current state
        self.current_position = np.zeros(6)
        self.current_velocity = np.zeros(6)

        # Force-torque data
        self.force_torque = np.zeros(6)

    def force_torque_callback(self, msg):
        """Callback to update the force-torque data from the sensor."""
        self.force_torque = np.array([msg.wrench.force.x,
                                      msg.wrench.force.y,
                                      msg.wrench.force.z,
                                      msg.wrench.torque.x,
                                      msg.wrench.torque.y,
                                      msg.wrench.torque.z])
        self.update_control()

    def update_control(self):
        """Calculate the impedance control output and apply it."""
        # Impedance control law: F_ext = M * (d2x) + D * (dx) + K * (x - x_d)
        # Solving for acceleration: d2x = M^(-1) * (F_ext - D * (dx) - K * (x - x_d))

        position_error = self.current_position - self.desired_position
        velocity_error = self.current_velocity - self.desired_velocity

        acceleration = np.linalg.inv(self.mass_matrix) @ (
            self.force_torque - self.damping_matrix @ velocity_error - self.stiffness_matrix @ position_error)

        # Update velocity and position (simple Euler integration)
        self.current_velocity += acceleration * 0.001  # Assume 1 ms time step
        self.current_position += self.current_velocity * 0.001

        # Publish the desired velocity
        self.publish_desired_velocity()

    def publish_desired_velocity(self):
        """Publish the desired velocity as a Twist message."""
        twist_msg = Twist()
        twist_msg.linear.x = self.current_velocity[0]
        twist_msg.linear.y = self.current_velocity[1]
        twist_msg.linear.z = self.current_velocity[2]
        twist_msg.angular.x = self.current_velocity[3]
        twist_msg.angular.y = self.current_velocity[4]
        twist_msg.angular.z = self.current_velocity[5]

        self.velocity_publisher.publish(twist_msg)

    def apply_control_command(self):
        """Send the control command to the robot."""
        # Placeholder function to send the control command to the robot.
        # You will need to integrate this with your specific robot API.
        pass

def main(args=None):
    rclpy.init(args=args)
    impedance_controller = ImpedanceController()
    rclpy.spin(impedance_controller)
    impedance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
