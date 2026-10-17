```python
#!/usr/bin/env python3

"""
leader_drone.py

COMPLETE LEADER + LiDAR DRONE
=============================

Drone 0 = Leader / Mother Drone

Functions
---------
1. PX4 local position
2. PX4 velocity
3. PX4 GPS position
4. PX4 vehicle status
5. RC/manual flight through PX4
6. Leader position broadcasting
7. Leader status broadcasting
8. LiDAR LaserScan input
9. Front / left / right obstacle detection
10. Minimum obstacle distance
11. Obstacle warning
12. LiDAR information broadcasting

IMPORTANT
---------
This node DOES NOT:
    - arm the drone
    - disarm the drone
    - take off
    - land
    - switch to Offboard mode
    - send PX4 trajectory commands

Therefore, the real Leader Drone remains under
normal RC/PX4 control.

LiDAR topic:
    /scan

Leader pose:
    /swarm/poses/drone_0

Leader status:
    /swarm/status/drone_0

LiDAR status:
    /swarm/lidar/drone_0
"""

import math
import json

import numpy as np

import rclpy
from rclpy.node import Node

from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    DurabilityPolicy,
    HistoryPolicy,
)

from px4_msgs.msg import (
    VehicleLocalPosition,
    VehicleGlobalPosition,
    VehicleStatus,
)

from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import LaserScan

from std_msgs.msg import String


class LeaderDrone(Node):

    def __init__(self):

        super().__init__("leader_drone")

        # ======================================================
        # DRONE CONFIGURATION
        # ======================================================

        self.drone_id = 0

        self.role = "LEADER"

        self.publish_rate = 20.0

        self.dt = 1.0 / self.publish_rate

        # ======================================================
        # SAFETY PARAMETERS
        # ======================================================

        # Minimum safe distance
        self.safety_distance = 2.0

        # Critical distance
        self.critical_distance = 1.0

        # LiDAR maximum useful distance
        self.lidar_max_distance = 20.0

        # ======================================================
        # POSITION
        # ======================================================

        self.position = np.zeros(
            3,
            dtype=np.float64
        )

        self.velocity = np.zeros(
            3,
            dtype=np.float64
        )

        self.position_valid = False

        self.local_position_valid = False

        self.global_position_valid = False

        # ======================================================
        # GPS
        # ======================================================

        self.latitude = 0.0

        self.longitude = 0.0

        self.altitude = 0.0

        # ======================================================
        # VEHICLE STATUS
        # ======================================================

        self.vehicle_status_received = False

        self.armed = False

        self.nav_state = None

        # ======================================================
        # LIDAR
        # ======================================================

        self.lidar_received = False

        self.lidar_min_distance = float("inf")

        self.front_distance = float("inf")

        self.left_distance = float("inf")

        self.right_distance = float("inf")

        self.obstacle_detected = False

        self.critical_obstacle = False

        # ======================================================
        # GLOBAL POSITION REFERENCE
        # ======================================================

        # Example PX4/Gazebo reference.
        # Change these for your actual operating area.

        self.reference_latitude = 47.3979700

        self.reference_longitude = 8.5461638

        self.reference_altitude = 0.0

        # ======================================================
        # QOS
        # ======================================================

        self.px4_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.swarm_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        # ======================================================
        # PX4 LOCAL POSITION
        # ======================================================

        self.position_sub = self.create_subscription(

            VehicleLocalPosition,

            "/fmu/out/vehicle_local_position_v1",

            self.position_callback,

            self.px4_qos,
        )

        # ======================================================
        # PX4 GLOBAL POSITION
        # ======================================================

        self.global_position_sub = self.create_subscription(

            VehicleGlobalPosition,

            "/fmu/out/vehicle_global_position",

            self.global_position_callback,

            self.px4_qos,
        )

        # ======================================================
        # PX4 STATUS
        # ======================================================

        self.status_sub = self.create_subscription(

            VehicleStatus,

            "/fmu/out/vehicle_status_v4",

            self.status_callback,

            self.px4_qos,
        )

        # ======================================================
        # LIDAR SUBSCRIBER
        # ======================================================

        self.lidar_sub = self.create_subscription(

            LaserScan,

            "/scan",

            self.lidar_callback,

            10,
        )

        # ======================================================
        # LEADER POSITION PUBLISHER
        # ======================================================

        self.pose_pub = self.create_publisher(

            PoseStamped,

            "/swarm/poses/drone_0",

            self.swarm_qos,
        )

        # ======================================================
        # LEADER STATUS PUBLISHER
        # ======================================================

        self.status_pub = self.create_publisher(

            String,

            "/swarm/status/drone_0",

            10,
        )

        # ======================================================
        # LIDAR STATUS PUBLISHER
        # ======================================================

        self.lidar_status_pub = self.create_publisher(

            String,

            "/swarm/lidar/drone_0",

            10,
        )

        # ======================================================
        # POSITION TIMER
        # ======================================================

        self.pose_timer = self.create_timer(

            self.dt,

            self.publish_own_pose,
        )

        # ======================================================
        # STATUS TIMER
        # ======================================================

        self.status_timer = self.create_timer(

            1.0,

            self.publish_status,
        )

        # ======================================================
        # LIDAR STATUS TIMER
        # ======================================================

        self.lidar_status_timer = self.create_timer(

            0.2,

            self.publish_lidar_status,
        )

        # ======================================================
        # STARTUP
        # ======================================================

        self.get_logger().info(
            "=============================================="
        )

        self.get_logger().info(
            "LEADER + LiDAR DRONE STARTED"
        )

        self.get_logger().info(
            "Drone ID: 0"
        )

        self.get_logger().info(
            "Role: LEADER"
        )

        self.get_logger().info(
            "RC Control: ENABLED THROUGH PX4"
        )

        self.get_logger().info(
            "Autonomous Offboard Control: DISABLED"
        )

        self.get_logger().info(
            "LiDAR Topic: /scan"
        )

        self.get_logger().info(
            "Leader Pose: /swarm/poses/drone_0"
        )

        self.get_logger().info(
            "Leader Status: /swarm/status/drone_0"
        )

        self.get_logger().info(
            "LiDAR Status: /swarm/lidar/drone_0"
        )

        self.get_logger().info(
            "=============================================="
        )

    # ==========================================================
    # PX4 LOCAL POSITION
    # ==========================================================

    def position_callback(self, msg):

        if hasattr(msg, "xy_valid"):

            if not msg.xy_valid:
                return

        if hasattr(msg, "z_valid"):

            if not msg.z_valid:
                return

        position = np.array(
            [
                msg.x,
                msg.y,
                msg.z,
            ],
            dtype=np.float64,
        )

        velocity = np.array(
            [
                msg.vx,
                msg.vy,
                msg.vz,
            ],
            dtype=np.float64,
        )

        if not np.all(
            np.isfinite(position)
        ):

            return

        if not np.all(
            np.isfinite(velocity)
        ):

            velocity = np.zeros(3)

        self.position = position

        self.velocity = velocity

        self.local_position_valid = True

        self.position_valid = True

    # ==========================================================
    # PX4 GLOBAL POSITION
    # ==========================================================

    def global_position_callback(self, msg):

        try:

            latitude = float(msg.lat)

            longitude = float(msg.lon)

            altitude = float(msg.alt)

        except Exception:

            return

        if not np.all(
            np.isfinite(
                [
                    latitude,
                    longitude,
                    altitude,
                ]
            )
        ):

            return

        self.latitude = latitude

        self.longitude = longitude

        self.altitude = altitude

        self.global_position_valid = True

    # ==========================================================
    # PX4 VEHICLE STATUS
    # ==========================================================

    def status_callback(self, msg):

        self.vehicle_status_received = True

        self.armed = (

            msg.arming_state
            ==
            VehicleStatus.ARMING_STATE_ARMED
        )

        self.nav_state = msg.nav_state

    # ==========================================================
    # LIDAR CALLBACK
    # ==========================================================

    def lidar_callback(self, msg):

        if len(msg.ranges) == 0:

            return

        ranges = np.array(
            msg.ranges,
            dtype=np.float64,
        )

        # ------------------------------------------------------
        # Remove invalid values
        # ------------------------------------------------------

        valid = ranges[
            np.isfinite(ranges)
        ]

        valid = valid[
            valid >= msg.range_min
        ]

        valid = valid[
            valid <= min(
                msg.range_max,
                self.lidar_max_distance
            )
        ]

        if len(valid) == 0:

            self.lidar_received = False

            self.lidar_min_distance = float("inf")

            self.obstacle_detected = False

            self.critical_obstacle = False

            return

        # ------------------------------------------------------
        # Overall minimum distance
        # ------------------------------------------------------

        self.lidar_min_distance = float(
            np.min(valid)
        )

        self.lidar_received = True

        # ------------------------------------------------------
        # Convert angles
        # ------------------------------------------------------

        angles = (

            msg.angle_min
            +
            np.arange(
                len(ranges)
            )
            *
            msg.angle_increment
        )

        # ------------------------------------------------------
        # Valid measurement mask
        # ------------------------------------------------------

        valid_mask = (

            np.isfinite(ranges)
            &
            (ranges >= msg.range_min)
            &
            (
                ranges
                <=
                min(
                    msg.range_max,
                    self.lidar_max_distance
                )
            )
        )

        # ------------------------------------------------------
        # FRONT
        #
        # -30° to +30°
        # ------------------------------------------------------

        front_mask = (

            valid_mask
            &
            (
                np.abs(angles)
                <=
                math.radians(30.0)
            )
        )

        # ------------------------------------------------------
        # LEFT
        #
        # +30° to +120°
        # ------------------------------------------------------

        left_mask = (

            valid_mask
            &
            (angles >= math.radians(30.0))
            &
            (angles <= math.radians(120.0))
        )

        # ------------------------------------------------------
        # RIGHT
        #
        # -120° to -30°
        # ------------------------------------------------------

        right_mask = (

            valid_mask
            &
            (angles >= math.radians(-120.0))
            &
            (angles <= math.radians(-30.0))
        )

        # ------------------------------------------------------
        # Calculate distances
        # ------------------------------------------------------

        self.front_distance = self.calculate_sector_distance(
            ranges,
            front_mask
        )

        self.left_distance = self.calculate_sector_distance(
            ranges,
            left_mask
        )

        self.right_distance = self.calculate_sector_distance(
            ranges,
            right_mask
        )

        # ------------------------------------------------------
        # Obstacle detection
        # ------------------------------------------------------

        self.obstacle_detected = (

            self.front_distance
            <
            self.safety_distance
        )

        self.critical_obstacle = (

            self.front_distance
            <
            self.critical_distance
        )

        # ------------------------------------------------------
        # Console warning
        # ------------------------------------------------------

        if self.critical_obstacle:

            self.get_logger().warn(

                "CRITICAL OBSTACLE! "
                f"Front = "
                f"{self.front_distance:.2f} m"
            )

        elif self.obstacle_detected:

            self.get_logger().warn(

                "Obstacle detected. "
                f"Front = "
                f"{self.front_distance:.2f} m"
            )

    # ==========================================================
    # SECTOR DISTANCE
    # ==========================================================

    def calculate_sector_distance(
        self,
        ranges,
        mask
    ):

        sector = ranges[mask]

        sector = sector[
            np.isfinite(sector)
        ]

        if len(sector) == 0:

            return float("inf")

        return float(
            np.min(sector)
        )

    # ==========================================================
    # PUBLISH LEADER POSITION
    # ==========================================================

    def publish_own_pose(self):

        if not self.position_valid:

            return

        msg = PoseStamped()

        msg.header.stamp = (

            self.get_clock()
            .now()
            .to_msg()
        )

        # PX4 local NED frame
        msg.header.frame_id = (
            "px4_local_ned"
        )

        msg.pose.position.x = float(
            self.position[0]
        )

        msg.pose.position.y = float(
            self.position[1]
        )

        msg.pose.position.z = float(
            self.position[2]
        )

        # No yaw information is published
        # by this node.

        msg.pose.orientation.x = 0.0

        msg.pose.orientation.y = 0.0

        msg.pose.orientation.z = 0.0

        msg.pose.orientation.w = 1.0

        self.pose_pub.publish(
            msg
        )

    # ==========================================================
    # PUBLISH LEADER STATUS
    # ==========================================================

    def publish_status(self):

        speed = float(
            np.linalg.norm(
                self.velocity
            )
        )

        status = {

            "drone_id": 0,

            "role": "LEADER",

            "control_mode": "RC_MANUAL",

            "offboard_control": False,

            "armed":
                bool(
                    self.armed
                ),

            "position_valid":
                bool(
                    self.position_valid
                ),

            "position": [

                float(
                    self.position[0]
                ),

                float(
                    self.position[1]
                ),

                float(
                    self.position[2]
                ),
            ],

            "velocity": [

                float(
                    self.velocity[0]
                ),

                float(
                    self.velocity[1]
                ),

                float(
                    self.velocity[2]
                ),
            ],

            "speed":
                speed,

            "latitude":
                self.latitude,

            "longitude":
                self.longitude,

            "altitude":
                self.altitude,

            "lidar_obstacle":
                bool(
                    self.obstacle_detected
                ),

            "critical_obstacle":
                bool(
                    self.critical_obstacle
                ),
        }

        msg = String()

        msg.data = json.dumps(
            status
        )

        self.status_pub.publish(
            msg
        )

    # ==========================================================
    # PUBLISH LIDAR STATUS
    # ==========================================================

    def publish_lidar_status(self):

        lidar_status = {

            "drone_id": 0,

            "lidar_available":
                bool(
                    self.lidar_received
                ),

            "minimum_distance":
                float(
                    self.lidar_min_distance
                ),

            "front_distance":
                float(
                    self.front_distance
                ),

            "left_distance":
                float(
                    self.left_distance
                ),

            "right_distance":
                float(
                    self.right_distance
                ),

            "obstacle_detected":
                bool(
                    self.obstacle_detected
                ),

            "critical_obstacle":
                bool(
                    self.critical_obstacle
                ),

            "safety_distance":
                self.safety_distance,

            "critical_distance":
                self.critical_distance,
        }

        msg = String()

        msg.data = json.dumps(
            lidar_status
        )

        self.lidar_status_pub.publish(
            msg
        )

    # ==========================================================
    # SHUTDOWN
    # ==========================================================

    def shutdown(self):

        self.get_logger().info(
            "Leader + LiDAR Drone shutting down."
        )

        # No arm/disarm command.
        # No takeoff command.
        # No landing command.
        # No Offboard command.

        self.get_logger().info(
            "PX4 control remains with RC/manual system."
        )


# ==============================================================
# MAIN
# ==============================================================

def main(args=None):

    rclpy.init(
        args=args
    )

    node = None

    try:

        node = LeaderDrone()

        rclpy.spin(
            node
        )

    except KeyboardInterrupt:

        pass

    except Exception as exc:

        if node is not None:

            node.get_logger().error(
                f"Fatal error: {exc}"
            )

        else:

            print(
                f"Fatal error: {exc}"
            )

    finally:

        if node is not None:

            node.shutdown()

            node.destroy_node()

        rclpy.shutdown()


if __name__ == "__main__":

    main()
```
