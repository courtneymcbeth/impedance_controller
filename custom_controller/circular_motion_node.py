import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from numpy.linalg import norm, solve
import pinocchio as pin
import math
import os
from ament_index_python.packages import get_package_share_directory

class CircularMotionNode(Node):
    def __init__(self):
        super().__init__('circular_motion_node')
        # Subscribe to the joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        
        # self.trajectory_pub = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.trajectory_pub = self.create_publisher(Float64MultiArray, '/forward_velocity_controller/commands', 10)

        self.joint_names = None
        self.initial_joint_positions = None
        self.circle_radius = 0.1  # 0.1m circle radius
        self.angular_velocity = 0.1  # rad/s, adjust as needed
        self.time_step = 0.002  # 500 Hz (2 milliseconds)
        self.timer = self.create_timer(self.time_step, self.publish_trajectory)
        self.start_time = self.get_clock().now()

        # Declare parameters
        self.declare_parameter('robot_description', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('end_effector_link', 'tool0')

        # Get URDF from the parameter server
        self.urdf = self.get_parameter('robot_description').value

        urdf_file = os.path.join(get_package_share_directory('custom_controller'), 'urdf', 'ur5e.urdf')
        self.robot_model = pin.buildModelFromUrdf(urdf_file)
        self.robot_data = self.robot_model.createData()
        self.ee_frame_id = self.robot_model.getFrameId('tool0')  # Replace with actual frame name

        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12
        self.initial_positions_published = False

    def joint_state_callback(self, msg):
        if self.joint_names is None:
            self.joint_names = np.roll(np.array(msg.name), 1)
        if self.initial_joint_positions is None:
            self.initial_joint_positions = np.roll(np.array(msg.position), 1)
            self.joint_positions = self.initial_joint_positions
            # self.publish_initial_positions()

    def compute_jacobian(self, joint_positions):
        # Placeholder: Compute the Jacobian matrix here based on your robot's kinematics.
        # This is a simplified example, and you will need the actual kinematics for your robot.
        # The Jacobian should map joint velocities to end-effector velocities.
        # J = np.eye(len(joint_positions))  # Identity matrix as a placeholder
        J = pin.computeFrameJacobian(self.robot_model, self.robot_data, joint_positions, self.ee_frame_id)
        return J

    def publish_trajectory(self):
        if self.initial_joint_positions is None:
            return
        
        # current_time = (self.get_clock().now() - self.start_time).nanoseconds * 1e-9
        # angle = self.angular_velocity * current_time
        # dx = -self.circle_radius * math.sin(angle) * self.angular_velocity
        # dy = self.circle_radius * math.cos(angle) * self.angular_velocity
        # dz = 0.0  # Assuming planar motion for simplicity
        
        # End-effector velocity in Cartesian space
        # v_cartesian = np.array([dx, dy, dz, 0, 0, 0])  # 6D velocity (linear + angular, assuming only planar motion)

        # Compute the Jacobian for the current joint positions
        # J = self.compute_jacobian(self.initial_joint_positions)
        
        # Compute joint velocities from Cartesian velocities using the pseudo-inverse of the Jacobian
        joint_velocities = [0.1, 0.0, 0.0, 0.0, 0.0, 0.0]
        # self.joint_positions += joint_velocities * self.time_step
        
        # Float64MultiArray
        traj_msg = Float64MultiArray()
        traj_msg.data = joint_velocities

        # Create and publish the trajectory message
        # traj_msg = JointTrajectory()
        # traj_msg.joint_names = self.joint_names

        # point = JointTrajectoryPoint()
        # duration = Duration()
        # # duration.nanosec = 1e9
        # duration.sec = 1
        # point.time_from_start = duration
        # # point.time_from_start = rclpy.duration.Duration(nanoseconds=self.time_step)
        # point.positions = point.positions = [0.0] * len(self.joint_names) 
        # point.velocities = list(joint_velocities)

        # traj_msg.points.append(point)
        self.trajectory_pub.publish(traj_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CircularMotionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()