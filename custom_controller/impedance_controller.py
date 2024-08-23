import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration

class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')

        # Subscribe to the joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)

        # Subscribe to the force-torque sensor data
        self.force_torque_subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor',
            self.force_torque_callback,
            10)

        # Publisher for the joint trajectory and end-effector pose
        self.ee_pose_publisher = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.joint_trajectory_publisher = self.create_publisher(JointTrajectory, '/test_traj', 10)

        # Impedance control parameters
        self.mass_matrix = np.diag([5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
        self.stiffness_matrix = np.diag([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])
        self.damping_matrix = np.diag([15.0, 15.0, 15.0, 1.5, 1.5, 1.5])

        # Current state
        self.current_transform = np.eye(4)  # 4x4 identity matrix as initial transform
        self.current_velocity = np.zeros(6)  # 6D velocity: linear (x, y, z) and angular (roll, pitch, yaw)

        # Force-torque data
        self.force_torque = np.zeros(6)

        # Current joint states
        self.current_joint_positions = None
        self.current_joint_velocities = None

    def joint_state_callback(self, msg):
        """Callback to update the current joint states."""
        self.current_joint_positions = np.array(msg.position)
        self.current_joint_velocities = np.array(msg.velocity)  # Obtain joint velocities

        with self.psm.read_only() as scene:
            psm_robot_state = scene.current_state
            self.current_transform = psm_robot_state.get_global_link_transform("tool0")

            # Extract position and orientation directly from the transformation matrix
            position = self.current_transform[:3, 3]
            orientation = self.quaternion_from_matrix(self.current_transform)

            # Publish the end-effector pose
            self.publish_ee_pose(position, orientation)

    def quaternion_from_matrix(self, matrix):
        """Convert a 4x4 transformation matrix to a quaternion using scipy."""
        rotation = R.from_matrix(matrix[:3, :3])  # Extract rotation matrix and convert
        return rotation.as_quat()  # Returns quaternion in the form [x, y, z, w]

    def force_torque_callback(self, msg):
        """Callback to update the force-torque data from the sensor."""
        self.force_torque = np.array([msg.wrench.force.x,
                                      msg.wrench.force.y,
                                      msg.wrench.force.z,
                                      msg.wrench.torque.x,
                                      msg.wrench.torque.y,
                                      msg.wrench.torque.z])

        # Nullify small force/torque values to avoid noise-induced motions
        self.force_torque = np.where(np.abs(self.force_torque) < 3.0, 0.0, self.force_torque)

        # Update control if joint positions are available
        if self.current_joint_positions is not None:
            self.update_control()

    def update_control(self):
        """Calculate the compliance control output and apply it."""
        time_step = 0.001  # Time step for integration, adjust as needed

        # Calculate acceleration based on the applied force/torque
        acceleration = np.linalg.inv(self.mass_matrix) @ (
            self.force_torque - self.damping_matrix @ self.current_velocity - self.stiffness_matrix @ self.current_transform[:3, 3])

        # Calculate the change in velocity
        delta_velocity = acceleration * time_step

        # Update the current velocity
        self.current_velocity += delta_velocity

        # Integrate velocity to update position
        delta_position = self.current_velocity[:3] * time_step
        delta_rotation = R.from_rotvec(self.current_velocity[3:] * time_step).as_matrix()

        # Update the current transformation matrix
        self.current_transform[:3, 3] += delta_position
        self.current_transform[:3, :3] = delta_rotation @ self.current_transform[:3, :3]

        # Check if the force is below a certain threshold, implying no external force
        if np.linalg.norm(self.force_torque) < 3.0:
            # If no significant force is applied, reset the velocity to zero
            self.current_velocity = np.zeros(6)

        # Calculate the inverse kinematics to get the joint positions
        joint_positions = self.solve_inverse_kinematics(self.current_transform)

        # Publish the desired joint positions
        self.publish_joint_positions(joint_positions)

    def solve_inverse_kinematics(self, desired_transform):
        """Calculate the inverse kinematics using the full transformation matrix."""
        pose_goal = Pose()
        pose_goal.position.x = desired_transform[0, 3]
        pose_goal.position.y = desired_transform[1, 3]
        pose_goal.position.z = desired_transform[2, 3]

        quaternion = self.quaternion_from_matrix(desired_transform)
        pose_goal.orientation.x = quaternion[0]
        pose_goal.orientation.y = quaternion[1]
        pose_goal.orientation.z = quaternion[2]
        pose_goal.orientation.w = quaternion[3]

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
        pose_msg.pose.orientation.x = orientation[0]
        pose_msg.pose.orientation.y = orientation[1]
        pose_msg.pose.orientation.z = orientation[2]
        pose_msg.pose.orientation.w = orientation[3]

        self.ee_pose_publisher.publish(pose_msg)

def main(args=None):
    rclpy.init(args=args)
    impedance_controller = ImpedanceController()
    rclpy.spin(impedance_controller)
    impedance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
