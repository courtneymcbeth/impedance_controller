# SERVO Instructions

1. Start robot driver
```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.1.5 launch_rviz:=true
```

2. Start robot on teach pendant

3. Zero force/torque sensor
```bash
ros2 service call /io_and_status_controller/zero_ftsensor std_srvs/srv/Trigger
```

4. Activate controller
```bash
ros2 control switch_controllers --deactivate scaled_joint_trajectory_controller --activate forward_position_controller
```

5. Start compliance
```bash
ros2 launch custom_controller impedance_controller.launch.py ur_type:=ur5e launch_rviz:=true launch_servo:=true
```

6. Apply force to end-effector with hands
