import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from . import urdf as kdl_parser
import PyKDL as kdl


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
            '/force_torque_sensor_broadcaster/wrench',
            self.force_torque_callback,
            10)

        # self.robot_description_subscription = self.create_subscription(
        #     String,
        #     '/robot_description',
        #     self.robot_description_callback,
        #     10)

        # Subscribe to the joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.ee_pose_publisher = self.create_publisher(PoseStamped, '/ee_pose', 10)

        # Publisher for the joint trajectory and end-effector pose
        self.ee_pose_publisher = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.joint_trajectory_publisher = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)

        # Impedance control parameters
        self.mass_matrix = np.diag([5.0, 5.0, 5.0, 0.5, 0.5, 0.5])
        self.stiffness_matrix = np.diag([300.0, 300.0, 300.0, 30.0, 30.0, 30.0])
        self.damping_matrix = np.diag([15.0, 15.0, 15.0, 1.5, 1.5, 1.5])

        # Current state
        # self.current_transform = np.eye(4)  # 4x4 identity matrix as initial transform
        self.current_transform = None
        self.current_velocity = np.zeros(6)  # 6D velocity: linear (x, y, z) and angular (roll, pitch, yaw)

        # Force-torque data
        self.force_torque = np.zeros(6)

        # Current joint states
        self.current_joint_positions = None
        self.current_joint_velocities = None
        self.initialized = False

        # Declare parameters
        self.declare_parameter('robot_description', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('end_effector_link', 'tool0')

        # Get URDF from the parameter server
        self.urdf = self.get_parameter('robot_description').value

        # Parse URDF to KDL Tree
        self.get_logger().error('Constructing KDL tree')
        success, kdl_tree = kdl_parser.treeFromString(self.urdf)
        if not success:
            self.get_logger().error('Failed to construct KDL tree')
            return

        # Extract the KDL chain from base_link to end_effector_link
        self.base_link = self.get_parameter('base_link').value
        end_effector_link = self.get_parameter('end_effector_link').value

        self.kdl_chain = kdl_tree.getChain(self.base_link, end_effector_link)
        if self.kdl_chain is None:
            self.get_logger().error('Failed to get KDL chain')
            return
        self.num_joints = self.kdl_chain.getNrOfJoints()

        # Initialize the forward kinematics solver
        self.fk_solver = kdl.ChainFkSolverPos_recursive(self.kdl_chain)
        ik_v_solver = kdl.ChainIkSolverVel_pinv(self.kdl_chain)  # Velocity IK solver
        self.ik_solver = kdl.ChainIkSolverPos_NR(self.kdl_chain, self.fk_solver, ik_v_solver)  # Position IK solver


    def joint_state_callback(self, msg):
        """Callback to update the current joint states."""
        if len(msg.position) != self.num_joints:
            self.get_logger().error('Joint state size does not match the number of joints in the chain')
            return
        self.current_joint_positions = np.array(msg.position)
        self.current_joint_velocities = np.array(msg.velocity)  # Obtain joint velocities

        # joint_positions = kdl.JntArray(self.num_joints)
        # for i, position in enumerate(msg.position):
        #     # joint_positions[i] = position
        #     # self.get_logger().info(f'i: {i}')
        #     if i == (self.num_joints-1):
        #         joint_positions[0] = position
        #     else:
        #         joint_positions[i+1] = position
        joint_positions = self.array_to_jnt_array(msg.position)
        
        end_effector_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_positions, end_effector_frame)
        self.current_transform = self.frame_to_transformation_matrix(end_effector_frame)

        pos = [end_effector_frame.p.x(), end_effector_frame.p.y(), end_effector_frame.p.z()]
        quat = end_effector_frame.M.GetQuaternion()
        ori = [quat[0], quat[1], quat[2], quat[3]]

        # Publish the end-effector pose
        self.publish_ee_pose(pos, ori)

    def array_to_jnt_array(self, array):
        """
        Converts a NumPy array or list to a PyKDL JntArray.

        Parameters:
        - array: NumPy array or list-like object representing joint positions

        Returns:
        - jnt_array: PyKDL.JntArray object
        """
        # Create a JntArray with the same size as the input array
        jnt_array = kdl.JntArray(len(array))

        # Populate the JntArray with the values from the input array
        for i in range(len(array)):
            if i ==(self.num_joints-1):
                jnt_array[0] = array[i]
            else:
                jnt_array[i+1] = array[i]

        return jnt_array

    def jnt_array_to_array(self, jnt_array):
        """
        Converts a PyKDL JntArray to a NumPy array.

        Parameters:
        - jnt_array: PyKDL.JntArray object

        Returns:
        - array: NumPy array representing joint positions
        """
        # Create a NumPy array with the same size as the JntArray
        array = np.zeros(jnt_array.rows())

        # Populate the NumPy array with the values from the JntArray
        for i in range(jnt_array.rows()):
            if i == 0:
                array[self.num_joints-1] = jnt_array[i]
            else:
                array[i] = jnt_array[i-1]

        return array

    def transformation_matrix_to_frame(self, matrix):
        """
        Converts a 4x4 transformation matrix to a PyKDL Frame.

        Parameters:
        - matrix: 4x4 numpy array representing the transformation matrix

        Returns:
        - frame: PyKDL.Frame object
        """
        # Extract the rotation matrix (3x3) and translation vector (3x1) from the transformation matrix
        rotation_matrix = matrix[:3, :3]
        translation_vector = matrix[:3, 3]

        # Convert the rotation matrix to a PyKDL.Rotation object
        rotation = kdl.Rotation(
            rotation_matrix[0, 0], rotation_matrix[0, 1], rotation_matrix[0, 2],
            rotation_matrix[1, 0], rotation_matrix[1, 1], rotation_matrix[1, 2],
            rotation_matrix[2, 0], rotation_matrix[2, 1], rotation_matrix[2, 2]
        )

        # Convert the translation vector to a PyKDL.Vector object
        translation = kdl.Vector(translation_vector[0], translation_vector[1], translation_vector[2])

        # Create the PyKDL.Frame using the rotation and translation
        frame = kdl.Frame(rotation, translation)

        return frame

    def frame_to_transformation_matrix(self, frame):
        """
        Converts a PyKDL Frame to a 4x4 transformation matrix.

        Parameters:
        - frame: PyKDL.Frame object

        Returns:
        - transformation_matrix: 4x4 numpy array representing the transformation matrix
        """
        # Initialize a 4x4 identity matrix
        transformation_matrix = np.eye(4)

        # Extract the rotation matrix from the frame
        rotation_matrix = frame.M  # This is a PyKDL.Rotation

        # Convert the PyKDL.Rotation to a 3x3 numpy array
        for i in range(3):
            for j in range(3):
                transformation_matrix[i, j] = rotation_matrix[i, j]

        # Extract the translation vector from the frame
        translation_vector = frame.p  # This is a PyKDL.Vector

        # Assign the translation vector to the transformation matrix
        transformation_matrix[0, 3] = translation_vector[0]
        transformation_matrix[1, 3] = translation_vector[1]
        transformation_matrix[2, 3] = translation_vector[2]

        return transformation_matrix

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
        ft_str = np.array2string(self.force_torque)
        # self.get_logger().info(f'ft: {ft_str}')

        # Update control if joint positions are available
        if self.current_joint_positions is not None:
            self.update_control()

    def update_control(self):
        """Calculate the compliance control output and apply it."""
        time_step = 0.001  # Time step for integration, adjust as needed
        old = self.current_transform

        # Calculate acceleration based on the applied force/torque
        # acceleration = np.linalg.inv(self.mass_matrix) @ (
        #     self.force_torque - self.damping_matrix @ self.current_velocity - self.stiffness_matrix @ self.current_transform[:3, 3])
        acceleration = np.linalg.inv(self.mass_matrix) @ (
            self.force_torque - self.damping_matrix @ self.current_velocity)

        # Calculate the change in velocity
        delta_velocity = acceleration * time_step

        # Update the current velocity
        self.current_velocity += delta_velocity
        vel_str = np.array2string(self.current_velocity)
        self.get_logger().info(f'vel: {vel_str}')

        # Integrate velocity to update position
        delta_position = self.current_velocity[:3] * time_step
        delta_rotation = R.from_rotvec(self.current_velocity[3:] * time_step).as_matrix()

        # Update the current transformation matrix
        self.current_transform[:3, 3] += delta_position
        self.current_transform[:3, :3] = delta_rotation @ self.current_transform[:3, :3]

        pos_str = np.array2string(self.current_transform[:3, 3])
        self.get_logger().info(f'new pos: {pos_str}')

        pos_str = np.array2string(old[:3, 3] - self.current_transform[:3, 3])
        self.get_logger().info(f'diff: {pos_str}\n')


        # Check if the force is below a certain threshold, implying no external force
        if np.linalg.norm(self.force_torque) < 3.0:
            # If no significant force is applied, reset the velocity to zero
            self.current_velocity = np.zeros(6)

        # Calculate the inverse kinematics to get the joint positions 
        joint_positions = self.solve_inverse_kinematics(self.current_transform)

        # Publish the desired joint positions
        # self.publish_joint_positions(joint_positions)
        self.get_logger().error('6')

    def solve_inverse_kinematics(self, desired_transform):
        """Calculate the inverse kinematics using the full transformation matrix."""

        desired_pose = self.transformation_matrix_to_frame(desired_transform)

        initial_joints = self.array_to_jnt_array(self.current_joint_positions)
        self.get_logger().info('cur py joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]' %
                               (self.current_joint_positions[0], self.current_joint_positions[1], self.current_joint_positions[2],
                                self.current_joint_positions[3], self.current_joint_positions[4], self.current_joint_positions[5]))
        
        self.get_logger().info('kdl py joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]' %
                               (initial_joints[0], initial_joints[1], initial_joints[2], initial_joints[3], initial_joints[4], initial_joints[5]))

        joint_positions = kdl.JntArray(self.kdl_chain.getNrOfJoints())

        result = self.ik_solver.CartToJnt(initial_joints, desired_pose, joint_positions)
        self.get_logger().error('5')
        if result >= 0:
             self.get_logger().info("IK solution found:")
             return self.jnt_array_to_array(joint_positions)
        else:
             self.get_logger().info("IK solver failed to find a solution")
             return self.current_joint_positions

    def publish_joint_positions(self, joint_positions):
        """Publish the desired joint positions as a JointTrajectory message."""
        trajectory_msg = JointTrajectory()
        trajectory_msg.joint_names = ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]

        point = JointTrajectoryPoint()
        if joint_positions is not None:
            point.positions = joint_positions
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
