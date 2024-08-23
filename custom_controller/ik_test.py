import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, Pose
from . import urdf as kdl_parser
import PyKDL as kdl
from scipy.spatial.transform import Rotation as R
import numpy as np

from moveit.core.robot_state import RobotState
from moveit.planning import (
    MoveItPy,
)

class KDLPositionNode(Node):
    def __init__(self):
        super().__init__('kdl_position_node')

        # Declare parameters
        self.declare_parameter('robot_description', '')
        self.declare_parameter('base_link', 'base_link')
        self.declare_parameter('end_effector_link', 'tool0')

        # Get URDF from the parameter server
        urdf = self.get_parameter('robot_description').value
        # self.get_logger().info(f'urdf: {urdf}')

        # Parse URDF to KDL Tree
        success, kdl_tree = kdl_parser.treeFromString(urdf)
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

        # Subscribe to the joint states
        self.joint_state_subscriber = self.create_subscription(
            JointState, '/joint_states', self.joint_state_callback, 10)

        self.ee_pose_publisher = self.create_publisher(PoseStamped, '/ee_pose', 10)

        self.ur5e = MoveItPy(node_name="moveit_py")
        self.arm = self.ur5e.get_planning_component("ur_manipulator")
        self.psm = self.ur5e.get_planning_scene_monitor()


    def joint_state_callback(self, msg):
        if len(msg.position) != self.num_joints:
            self.get_logger().error('Joint state size does not match the number of joints in the chain')
            return

        joint_positions = kdl.JntArray(self.num_joints)
        for i, position in enumerate(msg.position):
            # joint_positions[i] = position
            # self.get_logger().info(f'i: {i}')
            if i == (self.num_joints-1):
                joint_positions[0] = position
            else:
                joint_positions[i+1] = position
        # self.get_logger().info('kdl py joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]' %
        #                        (joint_positions[0], joint_positions[1], joint_positions[2], joint_positions[3], joint_positions[4], joint_positions[5]))

        end_effector_frame = kdl.Frame()
        self.fk_solver.JntToCart(joint_positions, end_effector_frame)

        self.get_logger().info('kdl py: [%.3f, %.3f, %.3f]' %
                               (end_effector_frame.p.x(), end_effector_frame.p.y(), end_effector_frame.p.z()))
        pos = [end_effector_frame.p.x(), end_effector_frame.p.y(), end_effector_frame.p.z()]
        quat = end_effector_frame.M.GetQuaternion()
        ori = [quat[0], quat[1], quat[2], quat[3]]
        self.publish_ee_pose(pos, ori)

        with self.psm.read_only() as scene:
            psm_robot_state = scene.current_state
            psm_joint_pos = psm_robot_state.get_joint_group_positions("ur_manipulator")
            # self.get_logger().info('moveit joints: [%.3f, %.3f, %.3f, %.3f, %.3f, %.3f]\n' %
            #                    (psm_joint_pos[0], psm_joint_pos[1], psm_joint_pos[2], psm_joint_pos[3], psm_joint_pos[4], psm_joint_pos[5]))

            current_transform = psm_robot_state.get_global_link_transform("tool0")

            # Extract position and orientation directly from the transformation matrix
            position = current_transform[:3, 3]
            orientation = self.quaternion_from_matrix(current_transform)
            self.get_logger().info('moveit: [%.3f, %.3f, %.3f]\n' %
                               (position[0], position[1], position[2]))

    def quaternion_from_matrix(self, matrix):
        """Convert a 4x4 transformation matrix to a quaternion using scipy."""
        rotation = R.from_matrix(matrix[:3, :3])  # Extract rotation matrix and convert
        return rotation.as_quat()  # Returns quaternion in the form [x, y, z, w]

    def publish_ee_pose(self, position, orientation):
        """Publish the current end-effector pose."""
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = self.base_link
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
    node = KDLPositionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()