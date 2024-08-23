import numpy as np
from scipy.spatial.transform import Rotation as R
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped, WrenchStamped, Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from numpy.linalg import norm, solve
import pinocchio as pin
import os
from ament_index_python.packages import get_package_share_directory

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

        urdf_file = os.path.join(get_package_share_directory('custom_controller'), 'urdf', 'ur5e.urdf')
        self.robot_model = pin.buildModelFromUrdf(urdf_file)
        self.robot_data = self.robot_model.createData()
        self.ee_frame_id = self.robot_model.getFrameId('tool0')  # Replace with actual frame name

        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12


    def joint_state_callback(self, msg):
        """Callback to update the current joint states."""
        self.current_joint_positions = np.roll(np.array(msg.position), 1)
        self.current_joint_velocities = np.roll(np.array(msg.velocity), 1)  # Obtain joint velocities

        self.compute_forward_kinematics()
    
    def compute_forward_kinematics(self):
        """Compute the end-effector position using forward kinematics."""
        if self.current_joint_positions is not None:
            pin.forwardKinematics(self.robot_model, self.robot_data, self.current_joint_positions)
            ee_pose = pin.updateFramePlacement(self.robot_model, self.robot_data, self.ee_frame_id)
            # print("ee_pose: ", ee_pose)
            self.current_transform = ee_pose

            pos = ee_pose.translation
            ori = pin.Quaternion(ee_pose.rotation).normalize()
            self.publish_ee_pose(pos, ori)


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
        self.get_logger().info(f'ft: {ft_str}')

        # Update control if joint positions are available
        if self.current_joint_positions is not None:
            self.update_control()

    def update_control(self):
        """Calculate the compliance control output and apply it."""
        time_step = 0.1  # Time step for integration, adjust as needed
        old = self.current_transform
        transform = self.current_transform
        
        old_str = np.array2string(old.translation)

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
        # self.get_logger().info(f'vel: {vel_str}')

        # Integrate velocity to update position
        delta_position = self.current_velocity[:3] * time_step
        delta_rotation =self.current_velocity[3:] * time_step

        # Update the current transformation matrix
        # self.current_transform[:3, 3] += delta_position
        # self.current_transform[:3, :3] = delta_rotation @ self.current_transform[:3, :3]
        transform.translation += delta_position
        # transform.rotation = delta_rotation @ transform.rotation
        

        # pos_str = np.array2string(transform.translation)
        # self.get_logger().info(f'old pos: {old_str}')
        # self.get_logger().info(f'new pos: {pos_str}')

        # pos_str = np.array2string(old.translation - transform.translation)
        self.get_logger().info(f'diff: {norm(old.translation - transform.translation, 2)}')


        # Check if the force is below a certain threshold, implying no external force
        self.get_logger().info(f'torque norm: {np.linalg.norm(self.force_torque, 2)}')
        if np.linalg.norm(self.force_torque) < 3.0:
            self.get_logger().info('HERE')
            # If no significant force is applied, reset the velocity to zero
            self.current_velocity = np.zeros(6)

        # Calculate the inverse kinematics to get the joint positions 
        joint_positions = self.solve_inverse_kinematics(transform)

        # Publish the desired joint positions
        self.publish_joint_positions(joint_positions)

    def solve_inverse_kinematics(self, desired_transform):
        """Calculate the inverse kinematics using the full transformation matrix."""

        q_init = self.current_joint_positions
        target_placement = pin.SE3(desired_transform.rotation, desired_transform.translation)  # Only position, no rotation

        if norm(self.current_transform.translation-desired_transform.translation) < self.eps:
            success, q_sol = True, q_init
        else:
            # Solve IK
            success, q_sol = self.compute_joint_placement(self.robot_model, self.robot_data, self.ee_frame_id, target_placement, q_init)

        if success:
            return q_sol
        else:
            self.get_logger().warn("IK solver did not converge to a solution.")
            return q_init  # Return initial guess if IK fails

    def compute_joint_placement(self, model, data, ee_frame_id, target_placement, q_init):
        q = q_init
        # print("q: ", q)
        i = 0
        while True:
            pin.forwardKinematics(model, data, q)
            iMd = data.oMf[ee_frame_id].actInv(target_placement)
            err = pin.log(iMd).vector
            if norm(err) < self.eps:
                success = True
                break
            if i >= self.IT_MAX:
                success = False
                break
            J = pin.computeFrameJacobian(model, data, q, ee_frame_id)
            J = -np.dot(pin.Jlog6(iMd.inverse()), J)
            v = -J.T.dot(solve(J.dot(J.T) + self.damp * np.eye(6), err))
            q = pin.integrate(model, q, v * self.DT)
            # if not i % 10:
            #     print("%d: error = %s" % (i, err.T))
            i += 1
        if success:
            print("Convergence achieved!")
        else:
            print(
                "\nWarning: the iterative algorithm has not reached convergence to the desired precision"
            )
        return success, q

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
