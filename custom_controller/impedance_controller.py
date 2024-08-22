import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import WrenchStamped, PoseStamped, Pose
from rcl_interfaces.msg import ParameterDescriptor # Enables the description of parameters

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
import numpy as np
from numpy.linalg import norm, solve
import pinocchio as pin
import os
from ament_index_python.packages import get_package_share_directory


class ImpedanceController(Node):
    def __init__(self):
        super().__init__('impedance_controller')

        # Subscribe to the force-torque sensor data
        self.force_subscription = self.create_subscription(
            WrenchStamped,
            '/force_torque_sensor_broadcaster/wrench',
            self.force_torque_callback,
            10)
        self.force_subscription  # prevent unused variable warning

        # Subscribe to the current joint states
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10)
        self.joint_state_subscription  # prevent unused variable warning

        # Publisher for the joint trajectory
        # self.joint_trajectory_publisher = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.ee_pose_publisher = self.create_publisher(PoseStamped, '/ee_pose', 10)
        self.joint_trajectory_publisher = self.create_publisher(JointTrajectory, '/test_traj', 10)
        
        # Impedance control parameters
        self.mass_matrix = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.damping_matrix = np.diag([0.7, 0.7, 0.7, 0.7, 0.7, 0.7])
        self.stiffness_matrix = np.diag([50.0, 50.0, 50.0, 50.0, 50.0, 50.0])

        # Desired position and velocity (target state)
        self.desired_position = np.zeros(3)  # Assuming only XYZ for simplicity
        self.desired_velocity = np.zeros(3)

        # Current state
        self.current_position = np.zeros(3)
        self.current_velocity = np.zeros(3)

        # Force-torque data
        self.force_torque = np.zeros(6)

        # Current joint states
        self.current_joint_positions = None

        # Load the robot model from the URDF
        urdf_file = os.path.join(get_package_share_directory('custom_controller'), 'urdf', 'ur5e.urdf')
        self.robot_model = pin.buildModelFromUrdf(urdf_file)
        self.robot_data = self.robot_model.createData()
        self.ee_frame_id = self.robot_model.getFrameId('tool0')  # Replace with actual frame name
        # self.ee_frame_id = 6
        # print(self.ee_frame_id)
        # print(self.robot_data.oMi[6])

        self.desired_set = False

        self.eps = 1e-4
        self.IT_MAX = 1000
        self.DT = 1e-1
        self.damp = 1e-12

    def joint_state_callback(self, msg):
        """Callback to update the current joint states."""
        self.current_joint_positions = np.array(msg.position)
        self.compute_forward_kinematics()

    def compute_forward_kinematics(self):
        """Compute the end-effector position using forward kinematics."""
        if self.current_joint_positions is not None:
            pin.forwardKinematics(self.robot_model, self.robot_data, self.current_joint_positions)
            ee_pose = pin.updateFramePlacement(self.robot_model, self.robot_data, self.ee_frame_id)
            # print("ee_pose: ", ee_pose)
            self.current_position = ee_pose.translation
            # print("current pos: ", self.current_position)
            if not self.desired_set:
                self.desired_position = self.current_position
                self.desired_set = True

    def force_torque_callback(self, msg):
        """Callback to update the force-torque data from the sensor."""
        self.force_torque = np.array([msg.wrench.force.x,
                                      msg.wrench.force.y,
                                      msg.wrench.force.z,
                                      msg.wrench.torque.x,
                                      msg.wrench.torque.y,
                                      msg.wrench.torque.z])

        # nullify
        for i in range(len(self.force_torque)):
            if abs(self.force_torque[i]) < 3.0:
                self.force_torque[i] = 0.0
        if np.sum(self.force_torque) < 3.0:
            self.current_velocity = np.zeros(3)

        if self.current_joint_positions is not None:
            self.update_control()

    def update_control(self):
        """Calculate the impedance control output and apply it."""
        position_error = self.desired_position - self.current_position
        position_error = np.where(np.abs(position_error) < 0.1, 0.0, position_error)
        # print("pos err: ", position_error)

        velocity_error = self.desired_velocity - self.current_velocity
        # print("cur vel: ", self.current_velocity)
        velocity_error = np.where(np.abs(velocity_error) < 0.1, 0.0, velocity_error)
        # print("vel err: ", velocity_error)


        # print("f/t: ", self.force_torque[:3])
        acceleration = np.linalg.inv(self.mass_matrix[:3,:3]) @ (
            self.force_torque[:3] - self.damping_matrix[:3,:3] @ velocity_error - self.stiffness_matrix[:3,:3] @ position_error)
        # print("accel: ", acceleration)

        # Update velocity and position (simple Euler integration)
        self.current_velocity += acceleration * 0.001  # Assume 1 ms time step
        self.current_position += self.current_velocity * 0.001

        self.publish_ee_pose(self.current_position)

        # Calculate the inverse kinematics to get the joint positions
        joint_positions = self.solve_inverse_kinematics(self.current_position)

        # Publish the desired joint positions
        self.publish_joint_positions(joint_positions)

    def solve_inverse_kinematics(self, target_position):
        """Solve the inverse kinematics to get joint positions."""
        # q_init = np.zeros(self.robot_model.nq)  # Initial guess
        q_init = self.current_joint_positions
        target_placement = pin.SE3(np.eye(3), target_position)  # Only position, no rotation

        # print("current: ", q_init)
        # print("target: ", target_position)

        if norm(self.current_position-target_position) < 0.05:
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

    def publish_ee_pose(self, ee_pose):
        pose_msg = PoseStamped()
        pose_msg.header.frame_id = 'ft_frame'
        pose_msg.pose.position.x = ee_pose[0]
        pose_msg.pose.position.y = ee_pose[1]
        pose_msg.pose.position.z = ee_pose[2]
        pose_msg.pose.orientation.w = 1.0

        self.ee_pose_publisher.publish(pose_msg)




def main(args=None):
    rclpy.init(args=args)
    impedance_controller = ImpedanceController()
    rclpy.spin(impedance_controller)
    impedance_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
