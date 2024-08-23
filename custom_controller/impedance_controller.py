import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped, Pose
from rcl_interfaces.msg import ParameterDescriptor  # Enables the description of parameters

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from numpy.linalg import norm, solve
import os
from ament_index_python.packages import get_package_share_directory

# moveit python library
from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
)

import tf_transformations


class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')

        # Subscribe to the joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        self.joint_state_subscription  # prevent unused variable warning

        # Publisher for the joint trajectory and end-effector pose
        self.ee_pose_publisher = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.joint_trajectory_publisher = self.create_publisher(JointTrajectory, '/test_traj', 10)

        # Impedance control parameters
        self.mass_matrix = np.diag([5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
        self.stiffness_matrix = np.diag([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])
        self.damping_matrix = np.diag([15.0, 15.0, 15.0, 1.5, 1.5, 1.5])

        # Current state
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)
        self.current_orientation = np.array([1.0, 0.0, 0.0, 0.0])  # Quaternion (w, x, y, z)

        # Force-torque data
        self.force_torque = np.zeros(6)
        self.ee_pose = np.zeros(3)

        # Current joint states
        self.current_joint_positions = None
        self.current_joint_velocities = None

        self.ur5e = MoveItPy(node_name="moveit_py")
        self.arm = self.ur5e.get_planning_component("ur_manipulator")
        self.psm = self.ur5e.get_planning_scene_monitor()

        with self.psm.read_only() as scene:
            psm_robot_state = scene.current_state
            ee_pose = psm_robot_state.get_global_link_transform("tool0")
            self.ee_pose = ee_pose[:3, 3]
            self.current_orientation = self.quaternion_from_matrix(ee_pose)

            pose_goal = Pose()
            pose_goal.position.x = 0.25
            pose_goal.position.y = 0.25
            pose_goal.position.z = 0.5
            pose_goal.orientation.w = 1.0
            psm_robot_state.set_from_ik("ur_manipulator", pose_goal, "tool0")

    def joint_state_callback(self, msg):
        """Callback to update the current joint states."""
        self.current_joint_positions = np.array(msg.position)
        self.current_joint_velocities = np.array(msg.velocity)  # Obtain joint velocities

        # Calculate the end-effector velocity using the Jacobian
        with self.psm.read_only() as scene:
            psm_robot_state = scene.current_state
            jacobian = psm_robot_state.get_jacobian("tool0")  # Get the Jacobian for the end-effector

            # Compute the end-effector velocity
            end_effector_velocity = jacobian[:3, :] @ self.current_joint_velocities
            self.current_velocity = end_effector_velocity

            ee_pose = psm_robot_state.get_global_link_transform("tool0")
            self.current_position = ee_pose[:3, 3]
            self.current_orientation = self.quaternion_from_matrix(ee_pose)

            self.publish_ee_pose(self.current_position, self.current_orientation)

    def force_torque_callback(self, msg):
        """Callback to update the force-torque data from the sensor."""
        self.force_torque = np.array([msg.wrench.force.x,
                                      msg.wrench.force.y,
                                      msg.wrench.force.z,
                                      msg.wrench.torque.x,
                                      msg.wrench.torque.y,
                                      msg.wrench.torque.z])

        # Nullify small force/torque values
        for i in range(len(self.force_torque)):
            if abs(self.force_torque[i]) < 3.0:
                self.force_torque[i] = 0.0

        if self.current_joint_positions is not None:
            self.update_control()

    def update_control(self):
        """Calculate the compliance control output and apply it, ensuring velocity ends at zero when no force is applied."""
        time_step = 0.001  # Time step for integration, adjust as needed

        # Calculate acceleration based on the applied force/torque
        acceleration = np.linalg.inv(self.mass_matrix[:3, :3]) @ (
            self.force_torque[:3] - self.damping_matrix[:3, :3] @ self.current_velocity)

        # Calculate the change in velocity
        delta_velocity = acceleration * time_step

        # Update the current velocity
        self.current_velocity += delta_velocity

        # Integrate velocity to update position
        self.current_position += self.current_velocity * time_step

        # Check if the force is below a certain threshold, implying no external force
        if np.linalg.norm(self.force_torque[:3]) < 3.0:
            # If no significant force is applied, reset the velocity to zero
            self.current_velocity = np.zeros(3)

        # Calculate the inverse kinematics to get the joint positions
        joint_positions = self.solve_inverse_kinematics(self.current_position, self.current_orientation)

        # Publish the desired joint positions
        self.publish_joint_positions(joint_positions)

    def solve_inverse_kinematics(self, desired_position, desired_orientation):
        """Calculate the inverse kinematics to obtain the joint positions for the desired end-effector position and orientation."""
        # Set the desired pose
        pose_goal = Pose()
        pose_goal.position.x = desired_position[0]
        pose_goal.position.y = desired_position[1]
        pose_goal.position.z = desired_position[2]
        pose_goal.orientation.w = desired_orientation[0]
        pose_goal.orientation.x = desired_orientation[1]
        pose_goal.orientation.y = desired_orientation[2]
        pose_goal.orientation.z = desired_orientation[3]

        # Calculate the inverse kinematics
        with self.psm.read_only() as scene:
            psm_robot_state = scene.current_state
            psm_robot_state.set_from_ik("ur_manipulator", pose_goal, "tool0")

            joint_positions = psm_robot_state.get_joint_positions("ur_manipulator")

        return joint_positions

    def publish_joint_positions(self, joint_positions):
        """Publish the desired joint positions as a JointTrajectory message."""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        point = JointTrajectoryPoint()
        if joint_positions is not None:
            point.positions = joint_positions.tolist()
        duration = Duration()
        duration.nanosec = 1000000
        point.time_from_start = duration

        trajectory_msg.points = [point]

        self.joint_trajectory_publisher.publish(trajectory_msg)

    def publish_ee_pose(self, position, orientation):
        """Publish the current end-effector pose."""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = position[0]
        pose_msg.pose.position.y = position[1]
        pose_msg.pose.position.z = position[2]
        pose_msg.pose.orientation.w = orientation[0]
        pose_msg.pose.orientation.x = orientation[1]
        pose_msg.pose.orientation.y = orientation[2]
        pose_msg.pose.orientation.z = orientation[3]

        self.ee_pose_publisher.publish(pose_msg)

    def quaternion_from_matrix(self, matrix):
      """Convert a 4x4 rotation matrix to a quaternion (x, y, z, w) using tf_transformations."""
      return tf_transformations.quaternion_from_matrix(matrix)


def main(args=None):
    rclpy.init(args=args)
    impedance_controller = ImpedanceController()
    rclpy.spin(impedance_controller)
    impedance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
