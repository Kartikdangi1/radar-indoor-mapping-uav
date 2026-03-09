#!/usr/bin/env python3
"""
Radar + IMU Frame Calibration GUI
Interactively adjust rotation parameters and switch between two calibration poses.
"""

import sys
import json
from pathlib import Path
from dataclasses import dataclass, asdict

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue

try:
    from PyQt5.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QSlider, QLabel, QPushButton, QGroupBox, QTabWidget, QSpinBox,
        QDoubleSpinBox, QMessageBox, QFileDialog, QFormLayout
    )
    from PyQt5.QtCore import Qt, QTimer
    from PyQt5.QtGui import QFont
except ImportError:
    print("PyQt5 not found. Install with: pip install PyQt5")
    sys.exit(1)


@dataclass
class CalibrationPose:
    """Represents a single calibration pose (6 DOF: 2 frames × 3 angles)"""
    imu_roll: float = 0.0
    imu_pitch: float = 0.0
    imu_yaw: float = 0.0
    radar_roll: float = 0.0
    radar_pitch: float = 0.0
    radar_yaw: float = 3.14159  # Default 180° for backward radar

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d):
        return CalibrationPose(**d)


class CalibrationNode(Node):
    """ROS 2 node for setting parameters on the fused odometry node"""

    def __init__(self):
        super().__init__('fused_odom_calibration_gui')
        self.client = self.create_client(SetParameters, '/fused_odom_node/set_parameters')
        self.get_logger().info("Waiting for /fused_odom_node parameter service...")
        while not self.client.wait_for_service(timeout_sec=2.0):
            self.get_logger().info("  Service not available, retrying...")

    def set_parameter(self, name: str, value: float):
        """Set a single parameter on the fused odometry node"""
        param = Parameter()
        param.name = f'/fused_odom_node/{name}'
        param.value = ParameterValue(type=2)  # PARAMETER_DOUBLE = 2
        param.value.double_value = value

        request = SetParameters.Request()
        request.parameters = [param]

        try:
            future = self.client.call_async(request)
            # Optionally wait for response
            # rclpy.spin_until_future_complete(self, future)
            self.get_logger().debug(f"Set {name} = {value:.4f}")
        except Exception as e:
            self.get_logger().error(f"Failed to set {name}: {e}")

    def apply_pose(self, pose: CalibrationPose):
        """Apply all 6 parameters for a pose"""
        params = [
            ('imu_to_base_roll', pose.imu_roll),
            ('imu_to_base_pitch', pose.imu_pitch),
            ('imu_to_base_yaw', pose.imu_yaw),
            ('radar_to_base_roll', pose.radar_roll),
            ('radar_to_base_pitch', pose.radar_pitch),
            ('radar_to_base_yaw', pose.radar_yaw),
        ]
        for name, value in params:
            self.set_parameter(name, value)


class CalibrationGUI(QMainWindow):
    """Main GUI window for calibration"""

    def __init__(self, ros_node):
        super().__init__()
        self.node = ros_node
        self.pose1 = CalibrationPose()
        self.pose2 = CalibrationPose()
        self.current_pose_idx = 1
        self.sliders = {}

        self.initUI()
        self.load_defaults()

    def initUI(self):
        """Build the GUI layout"""
        self.setWindowTitle("Radar + IMU Calibration GUI")
        self.setGeometry(100, 100, 1000, 600)

        main_layout = QHBoxLayout()

        # ── Left side: Pose 1 Controls ──
        pose1_box = self.build_pose_controls("Pose 1 (Current)", self.pose1)
        main_layout.addWidget(pose1_box, 1)

        # ── Right side: Pose 2 Controls ──
        pose2_box = self.build_pose_controls("Pose 2 (Current)", self.pose2)
        main_layout.addWidget(pose2_box, 1)

        # ── Center: Action Buttons ──
        button_layout = QVBoxLayout()
        button_layout.addStretch()

        btn_apply1 = QPushButton("Apply Pose 1")
        btn_apply1.setStyleSheet("background-color: #4CAF50; color: white; font-weight: bold;")
        btn_apply1.clicked.connect(self.apply_pose1)
        button_layout.addWidget(btn_apply1)

        btn_apply2 = QPushButton("Apply Pose 2")
        btn_apply2.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold;")
        btn_apply2.clicked.connect(self.apply_pose2)
        button_layout.addWidget(btn_apply2)

        btn_swap = QPushButton("Swap\nPose 1 ↔ 2")
        btn_swap.setStyleSheet("background-color: #FF9800; color: white;")
        btn_swap.clicked.connect(self.swap_poses)
        button_layout.addWidget(btn_swap)

        btn_save = QPushButton("Save\nCalibration")
        btn_save.clicked.connect(self.save_calibration)
        button_layout.addWidget(btn_save)

        btn_load = QPushButton("Load\nCalibration")
        btn_load.clicked.connect(self.load_calibration)
        button_layout.addWidget(btn_load)

        button_layout.addStretch()
        main_layout.addLayout(button_layout, 0)

        widget = QWidget()
        widget.setLayout(main_layout)
        self.setCentralWidget(widget)

    def build_pose_controls(self, title, pose):
        """Build a group of sliders for one pose"""
        group = QGroupBox(title)
        layout = QFormLayout()

        frame_font = QFont()
        frame_font.setBold(True)

        # IMU Frame Label
        imu_label = QLabel("IMU Frame:")
        imu_label.setFont(frame_font)
        layout.addRow(imu_label, QLabel(""))

        # IMU Sliders
        self.sliders['imu_roll'] = self.add_slider(
            layout, "Roll (rad)", pose.imu_roll, -3.15, 3.15)
        self.sliders['imu_pitch'] = self.add_slider(
            layout, "Pitch (rad)", pose.imu_pitch, -3.15, 3.15)
        self.sliders['imu_yaw'] = self.add_slider(
            layout, "Yaw (rad)", pose.imu_yaw, -3.15, 3.15)

        # Radar Frame Label
        radar_label = QLabel("Radar Frame:")
        radar_label.setFont(frame_font)
        layout.addRow(radar_label, QLabel(""))

        # Radar Sliders
        self.sliders['radar_roll'] = self.add_slider(
            layout, "Roll (rad)", pose.radar_roll, -3.15, 3.15)
        self.sliders['radar_pitch'] = self.add_slider(
            layout, "Pitch (rad)", pose.radar_pitch, -3.15, 3.15)
        self.sliders['radar_yaw'] = self.add_slider(
            layout, "Yaw (rad)", pose.radar_yaw, -3.15, 3.15)

        group.setLayout(layout)
        return group

    def add_slider(self, layout, label, value, min_val, max_val):
        """Add a slider + spinbox for a parameter"""
        spinbox = QDoubleSpinBox()
        spinbox.setRange(min_val, max_val)
        spinbox.setValue(value)
        spinbox.setSingleStep(0.1)
        spinbox.setDecimals(4)

        slider = QSlider(Qt.Horizontal)
        slider.setRange(int(min_val * 1000), int(max_val * 1000))
        slider.setValue(int(value * 1000))
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(500)

        # Connect slider ↔ spinbox
        slider.valueChanged.connect(
            lambda v: spinbox.blockSignals(True) or spinbox.setValue(v / 1000.0) or spinbox.blockSignals(False))
        spinbox.valueChanged.connect(
            lambda v: slider.blockSignals(True) or slider.setValue(int(v * 1000)) or slider.blockSignals(False))

        h_layout = QHBoxLayout()
        h_layout.addWidget(slider, 4)
        h_layout.addWidget(spinbox, 1)
        layout.addRow(label, h_layout)

        return {'slider': slider, 'spinbox': spinbox, 'label': label}

    def get_current_pose(self):
        """Read all slider values and return as a CalibrationPose"""
        return CalibrationPose(
            imu_roll=self.sliders['imu_roll']['spinbox'].value(),
            imu_pitch=self.sliders['imu_pitch']['spinbox'].value(),
            imu_yaw=self.sliders['imu_yaw']['spinbox'].value(),
            radar_roll=self.sliders['radar_roll']['spinbox'].value(),
            radar_pitch=self.sliders['radar_pitch']['spinbox'].value(),
            radar_yaw=self.sliders['radar_yaw']['spinbox'].value(),
        )

    def set_pose_on_ui(self, pose):
        """Update all sliders to match a pose"""
        fields = ['imu_roll', 'imu_pitch', 'imu_yaw', 'radar_roll', 'radar_pitch', 'radar_yaw']
        pose_dict = pose.to_dict()

        for field in fields:
            spinbox = self.sliders[field]['spinbox']
            spinbox.setValue(pose_dict[field])

    def apply_pose1(self):
        """Apply Pose 1 to the ROS node"""
        self.pose1 = self.get_current_pose()
        self.node.apply_pose(self.pose1)
        self.node.get_logger().info(f"Applied Pose 1: {self.pose1}")

    def apply_pose2(self):
        """Apply Pose 2 to the ROS node"""
        self.pose2 = self.get_current_pose()
        self.node.apply_pose(self.pose2)
        self.node.get_logger().info(f"Applied Pose 2: {self.pose2}")

    def swap_poses(self):
        """Swap Pose 1 and Pose 2 values and apply Pose 1"""
        self.pose1, self.pose2 = self.pose2, self.pose1
        self.set_pose_on_ui(self.pose1)
        self.apply_pose1()
        self.node.get_logger().info("Swapped Pose 1 and Pose 2")

    def save_calibration(self):
        """Save both poses to a JSON file"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Calibration", "", "JSON Files (*.json)")
        if not filepath:
            return

        data = {
            'pose1': self.pose1.to_dict(),
            'pose2': self.pose2.to_dict(),
        }
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2)
            QMessageBox.information(self, "Success", f"Saved to {filepath}")
            self.node.get_logger().info(f"Saved calibration to {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save: {e}")

    def load_calibration(self):
        """Load poses from a JSON file"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "Load Calibration", "", "JSON Files (*.json)")
        if not filepath:
            return

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            self.pose1 = CalibrationPose.from_dict(data.get('pose1', {}))
            self.pose2 = CalibrationPose.from_dict(data.get('pose2', {}))
            self.set_pose_on_ui(self.pose1)
            QMessageBox.information(self, "Success", f"Loaded from {filepath}")
            self.node.get_logger().info(f"Loaded calibration from {filepath}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load: {e}")

    def load_defaults(self):
        """Set reasonable defaults"""
        self.pose1 = CalibrationPose(
            imu_roll=0.0, imu_pitch=0.0, imu_yaw=0.0,
            radar_roll=0.0, radar_pitch=0.0, radar_yaw=3.14159  # 180°
        )
        self.pose2 = CalibrationPose(
            imu_roll=0.0, imu_pitch=0.0, imu_yaw=0.0,
            radar_roll=0.0, radar_pitch=0.0, radar_yaw=3.14159
        )
        self.set_pose_on_ui(self.pose1)


def main():
    # Initialize ROS 2
    rclpy.init()
    ros_node = CalibrationNode()

    # Start Qt app
    app = QApplication(sys.argv)
    gui = CalibrationGUI(ros_node)
    gui.show()

    # Spin ROS in background
    def ros_spin():
        rclpy.spin_once(ros_node, timeout_sec=0.01)

    timer = QTimer()
    timer.timeout.connect(ros_spin)
    timer.start(50)  # 20 Hz

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
