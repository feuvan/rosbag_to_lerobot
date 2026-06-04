#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
macOS (ROS2-free) port of convert_rosbag_with_markers.py.

Converts ROS2 bag files with X/Y episode markers to LeRobot dataset format v2.1.
Uses the pure-Python `rosbags` library instead of rclpy/rosbag2_py, and a streaming
decode -> re-encode pipeline for video (handles mixed codecs + mid-GOP starts).

Usage:
    uv run python scripts/convert_rosbag_with_markers_macos.py \
        --multibag \
        --input_directory ./data/rosbags \
        --output username/dataset_name \
        --fps 30 \
        --task "task description"
"""

import os
import sys
import numpy as np
import argparse
from pathlib import Path
import logging
import shutil
import traceback
from datetime import datetime

from rosbag_macos_io import BagReader, CameraTranscoder, make_typestore, STEREO_EYES

from lerobot.record import sample_frames_from_video
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_ACTION_DIM = 62
MIN_EPISODE_LENGTH = 30
ACTION_OFFSET_RATIO = 1.0 / 3.0


class MultiVideoRosBag2LeRobotConverter:

    def __init__(self, input_directory: str, output_repo_id: str, fps: int = 30, eye: str = "both"):
        self.input_directory = Path(input_directory)
        self.output_repo_id = output_repo_id
        self.fps = fps
        self.eye = eye
        self.frame_duration = 1.0 / self.fps

        if not logging.getLogger().handlers:
            logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        self.video_topics = {
            'left_color':  '/left/color/image_raw/ffmpeg',
            'right_color': '/right/color/image_raw/ffmpeg',
            'head_camera': '/xr_video_topic/ffmpeg',
        }
        self.video_topics_set = set(self.video_topics.values())
        self.state_topics_set = {
            '/left_arm/joint_states', '/left_arm/current_ee_pose', '/left_gripper/joint_states',
            '/right_arm/joint_states', '/right_arm/current_ee_pose', '/right_gripper/joint_states',
        }
        self.action_topics_set = {
            '/left_arm/joint_cmd', '/left_arm/target_ee_pose', '/left_gripper/joint_cmd',
            '/right_arm/joint_cmd', '/right_arm/target_ee_pose', '/right_gripper/joint_cmd',
        }
        self.all_topics_set = self.video_topics_set | self.state_topics_set | self.action_topics_set

        self.dataset = None
        self.num_xy_pairs = 0
        self._cur_state_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._cur_action_msg = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        self._transcoders: dict[str, CameraTranscoder] = {}

    def _get_camera_key_from_topic(self, topic: str) -> str | None:
        for key, t in self.video_topics.items():
            if t == topic:
                return key
        return None

    def discover_rosbags(self, MULTIBAG_FLAG):
        rosbags = []
        if MULTIBAG_FLAG:
            root = Path(self.input_directory).resolve()
            if not root.exists() or not root.is_dir():
                raise ValueError(f"Invalid path: {root}")
            for dirpath, dirnames, filenames in os.walk(root):
                current_dir = Path(dirpath)
                exts = {Path(f).suffix.lower() for f in filenames}
                has_bag = '.db3' in exts or '.mcap' in exts
                has_yaml = '.yaml' in exts or '.yml' in exts
                if has_bag and has_yaml:
                    bag_files = list(current_dir.glob("*.mcap")) or list(current_dir.glob("*.db3"))
                    for bf in bag_files:
                        rosbags.append({'name': current_dir.name, 'path': str(current_dir), 'bag_file': str(bf)})
                    dirnames.clear()
        else:
            bag_files = list(self.input_directory.glob("*.mcap")) or list(self.input_directory.glob("*.db3"))
            for i, bf in enumerate(sorted(bag_files)):
                rosbags.append({'name': f"episode_{i:03d}", 'path': str(bf.parent), 'bag_file': str(bf)})

        self.logger.info(f"Discovered {len(rosbags)} bags")
        return rosbags

    def setup_features(self):
        features = {}
        feature_names = [
            "left_joint1_position", "left_joint2_position", "left_joint3_position", "left_joint4_position",
            "left_joint5_position", "left_joint6_position", "left_joint7_position", "left_gripper_position",
            "right_joint1_position", "right_joint2_position", "right_joint3_position", "right_joint4_position",
            "right_joint5_position", "right_joint6_position", "right_joint7_position", "right_gripper_position",
            "left_joint1_velocity", "left_joint2_velocity", "left_joint3_velocity", "left_joint4_velocity",
            "left_joint5_velocity", "left_joint6_velocity", "left_joint7_velocity", "left_gripper_velocity",
            "right_joint1_velocity", "right_joint2_velocity", "right_joint3_velocity", "right_joint4_velocity",
            "right_joint5_velocity", "right_joint6_velocity", "right_joint7_velocity", "right_gripper_velocity",
            "left_joint1_effort", "left_joint2_effort", "left_joint3_effort", "left_joint4_effort",
            "left_joint5_effort", "left_joint6_effort", "left_joint7_effort", "left_gripper_effort",
            "right_joint1_effort", "right_joint2_effort", "right_joint3_effort", "right_joint4_effort",
            "right_joint5_effort", "right_joint6_effort", "right_joint7_effort", "right_gripper_effort",
            "left_ee_position_x", "left_ee_position_y", "left_ee_position_z",
            "left_ee_orientation_x", "left_ee_orientation_y", "left_ee_orientation_z", "left_ee_orientation_w",
            "right_ee_position_x", "right_ee_position_y", "right_ee_position_z",
            "right_ee_orientation_x", "right_ee_orientation_y", "right_ee_orientation_z", "right_ee_orientation_w",
        ]
        features["action"] = {"dtype": "float32", "shape": (STATE_ACTION_DIM,), "names": feature_names}
        features["observation.state"] = {"dtype": "float32", "shape": (STATE_ACTION_DIM,), "names": feature_names}
        for camera_key in self.video_topics.keys():
            tc = self._transcoders.get(camera_key)
            if tc and tc.initialized:
                h, w = tc.output_height, tc.output_width
            else:
                h, w = 1920, 3840
            features[f"observation.images.{camera_key}"] = {
                "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"],
            }
        return features

    def create_dataset_if_needed(self):
        if self.dataset is None:
            features = self.setup_features()
            self.dataset = LeRobotDataset.create(
                repo_id=self.output_repo_id, fps=self.fps, features=features,
                robot_type="dual_arm_robot",
            )

    def _ensure_transcoders(self):
        for camera_key in self.video_topics.keys():
            if camera_key not in self._transcoders:
                self._transcoders[camera_key] = CameraTranscoder(
                    camera_name=camera_key, fps=self.fps, root_dir=Path(""), eye=self.eye,
                )

    def _update_state(self, topic, msg):
        if topic == '/left_arm/joint_states':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            self._cur_state_msg[0:7] = pos[:7]
            self._cur_state_msg[16:23] = vel[:7]
            self._cur_state_msg[32:39] = eff[:7]
        elif topic == '/right_arm/joint_states':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            self._cur_state_msg[8:15] = pos[:7]
            self._cur_state_msg[24:31] = vel[:7]
            self._cur_state_msg[40:47] = eff[:7]
        elif topic == '/left_gripper/joint_states':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            if len(pos) > 0:
                self._cur_state_msg[7] = pos[0]
                self._cur_state_msg[23] = vel[0] if len(vel) > 0 else 0
                self._cur_state_msg[39] = eff[0] if len(eff) > 0 else 0
        elif topic == '/right_gripper/joint_states':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            if len(pos) > 0:
                self._cur_state_msg[15] = pos[0]
                self._cur_state_msg[31] = vel[0] if len(vel) > 0 else 0
                self._cur_state_msg[47] = eff[0] if len(eff) > 0 else 0
        elif topic == '/left_arm/current_ee_pose':
            self._cur_state_msg[48:55] = [
                msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            ]
        elif topic == '/right_arm/current_ee_pose':
            self._cur_state_msg[55:62] = [
                msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            ]

    def _update_action(self, topic, msg):
        if topic == '/left_arm/joint_cmd':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            self._cur_action_msg[0:7] = pos[:7]
            self._cur_action_msg[16:23] = vel[:7]
            self._cur_action_msg[32:39] = eff[:7]
        elif topic == '/right_arm/joint_cmd':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            self._cur_action_msg[8:15] = pos[:7]
            self._cur_action_msg[24:31] = vel[:7]
            self._cur_action_msg[40:47] = eff[:7]
        elif topic == '/left_gripper/joint_cmd':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            if len(pos) > 0:
                self._cur_action_msg[7] = pos[0]
                self._cur_action_msg[23] = vel[0] if len(vel) > 0 else 0
                self._cur_action_msg[39] = eff[0] if len(eff) > 0 else 0
        elif topic == '/right_gripper/joint_cmd':
            pos = list(msg.position); vel = list(msg.velocity); eff = list(msg.effort)
            if len(pos) > 0:
                self._cur_action_msg[15] = pos[0]
                self._cur_action_msg[31] = vel[0] if len(vel) > 0 else 0
                self._cur_action_msg[47] = eff[0] if len(eff) > 0 else 0
        elif topic == '/left_arm/target_ee_pose':
            self._cur_action_msg[48:55] = [
                msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            ]
        elif topic == '/right_arm/target_ee_pose':
            self._cur_action_msg[55:62] = [
                msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w,
            ]

    def convert_single_bag(self, rosbag, task_description, ENFORCE_FOUR_VIDEO_TOPICS_FLAG):
        self.logger.info(f"\n=== Processing {rosbag['name']} ===")
        bag_path = rosbag['path']

        self._ensure_transcoders()
        typestore = make_typestore()

        with BagReader(bag_path, typestore=typestore) as reader:
            topic_types = reader.topic_types

            if ENFORCE_FOUR_VIDEO_TOPICS_FLAG:
                for vt in self.video_topics_set:
                    if vt not in topic_types:
                        self.logger.warning(f"Missing video topic {vt}, skipping bag")
                        return

            x_button, y_button = 2, 3
            is_recording = False
            previous_buttons = None
            frame_data = dict()
            episode_state_target_t = None
            episode_action_target_t = None
            is_in_adding_phase = False
            start_time = None

            joy_topic = '/xr/left_hand_inputs'
            wanted_topics = self.all_topics_set | {joy_topic}

            for topic, msg, t_ns in reader.read(topics=wanted_topics):
                timestamp = t_ns / 1e9

                # Always feed video packets to the decoder (even outside recording)
                if topic in self.video_topics_set:
                    camera_key = self._get_camera_key_from_topic(topic)
                    if camera_key:
                        tc = self._transcoders[camera_key]
                        tc.feed(bytes(msg.data), msg.encoding, int(msg.width), int(msg.height))

                # Update state/action caches
                if topic in self.state_topics_set:
                    self._update_state(topic, msg)
                elif topic in self.action_topics_set:
                    self._update_action(topic, msg)

                # Joy button logic
                if topic == joy_topic:
                    try:
                        buttons = list(msg.buttons)
                        if len(buttons) > max(x_button, y_button):
                            if previous_buttons is not None:
                                # X press -> start/re-record
                                if previous_buttons[x_button] == 0 and buttons[x_button] == 1 and not is_recording:
                                    is_recording = True
                                    start_time = timestamp
                                    self.logger.info(f"Start Episode: {self.num_xy_pairs}")
                                    episode_state_target_t = start_time + self.frame_duration
                                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                                    is_in_adding_phase = False
                                    frame_data.clear()
                                    self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                                    for tc in self._transcoders.values():
                                        tc.begin_episode()

                                elif previous_buttons[x_button] == 0 and buttons[x_button] == 1 and is_recording:
                                    is_recording = True
                                    start_time = timestamp
                                    self.logger.info(f"Re-record Episode: {self.num_xy_pairs}")
                                    episode_state_target_t = start_time + self.frame_duration
                                    episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                                    self.dataset.clear_episode_buffer()
                                    for tc in self._transcoders.values():
                                        tc.discard()
                                        tc.begin_episode()
                                    self.dataset.episode_buffer = self.dataset.create_episode_buffer()
                                    is_in_adding_phase = False
                                    frame_data.clear()

                                # Y press -> stop
                                if previous_buttons[y_button] == 0 and buttons[y_button] == 1 and is_recording:
                                    is_recording = False
                                    end_time = timestamp
                                    duration = end_time - start_time
                                    self.logger.info(f"Stop. Duration: {duration:.3f}s")
                                    self.num_xy_pairs += 1

                                    ep_len = min(tc.length for tc in self._transcoders.values())
                                    state_len = self.dataset.episode_buffer["size"]

                                    if ep_len < MIN_EPISODE_LENGTH or state_len < MIN_EPISODE_LENGTH:
                                        self.dataset.clear_episode_buffer()
                                        for tc in self._transcoders.values():
                                            tc.discard()
                                    else:
                                        if ep_len > state_len:
                                            for tc in self._transcoders.values():
                                                tc.trim_final()
                                        elif ep_len < state_len:
                                            self.dataset.delete_final_frame()

                                        self._save_episode(task_description)

                                    episode_state_target_t = None
                                    episode_action_target_t = None

                        previous_buttons = list(buttons)
                    except Exception as e:
                        self.logger.error(f"Error processing Joy: {e}\n{traceback.format_exc()}")

                # Grid-rate recording logic
                if is_recording:
                    if timestamp < episode_state_target_t:
                        continue
                    elif episode_state_target_t <= timestamp <= episode_action_target_t and not is_in_adding_phase:
                        # Add state + commit video
                        frame_data["observation.state"] = self._cur_state_msg.copy()
                        for camera_key in self.video_topics.keys():
                            tc = self._transcoders[camera_key]
                            h = tc.output_height or 1920
                            w = tc.output_width or 3840
                            frame_data[f"observation.images.{camera_key}"] = np.zeros((h, w, 3), dtype=np.uint8)
                            tc.commit()
                        is_in_adding_phase = True
                    elif timestamp > episode_action_target_t and is_in_adding_phase:
                        frame_data['action'] = self._cur_action_msg.copy()
                        self.dataset.add_frame(frame_data, task_description,
                                               episode_state_target_t - start_time - self.frame_duration)
                        frame_data.clear()
                        time_gap = timestamp - episode_action_target_t
                        skipped_frames = time_gap // self.frame_duration
                        episode_state_target_t += (skipped_frames + 1) * self.frame_duration
                        episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                        is_in_adding_phase = False
                    elif timestamp > episode_action_target_t and not is_in_adding_phase:
                        time_gap = timestamp - episode_action_target_t
                        skipped_frames = time_gap // self.frame_duration + 1
                        episode_state_target_t += skipped_frames * self.frame_duration
                        episode_action_target_t = episode_state_target_t + ACTION_OFFSET_RATIO * self.frame_duration
                        frame_data.clear()

        shutil.rmtree(self.dataset.root / 'images', ignore_errors=True)

    def _save_episode(self, task_description):
        episode_index = self.dataset.num_episodes

        # Finalize video for each camera -> write MP4 to LeRobot path
        for camera_key, tc in self._transcoders.items():
            tc.root_dir = self.dataset.root
            video_path = tc.finalize(episode_index, self.dataset.meta)

            video_key = f"observation.images.{camera_key}"
            episode_length = tc.length
            try:
                sampled_frames = sample_frames_from_video(
                    video_path=video_path,
                    episode_length=episode_length,
                    fps=self.dataset.fps,
                    width=tc.output_width,
                    height=tc.output_height,
                )
                self.dataset.episode_buffer[video_key] = sampled_frames
                self.logger.info(f"Sampled {len(sampled_frames)} frames from {camera_key}")
            except Exception as e:
                self.logger.error(f"Failed to sample frames from {camera_key}: {e}")

        self.dataset.save_episode()
        self.logger.info(f"Saved episode {episode_index}")

    def convert_all(self, task_description, MULTIBAG_FLAG, ENFORCE_FOUR_VIDEO_TOPICS_FLAG):
        self.logger.info(f"Starting conversion: {self.input_directory}")
        rosbags = self.discover_rosbags(MULTIBAG_FLAG)
        if not rosbags:
            self.logger.error("No rosbags found!")
            return False

        # Do a quick pre-scan of the first bag to init decoders and get resolution
        self._ensure_transcoders()
        self._prescan_resolutions(rosbags[0])
        self.create_dataset_if_needed()

        for i, rosbag in enumerate(rosbags):
            self.convert_single_bag(rosbag, task_description, ENFORCE_FOUR_VIDEO_TOPICS_FLAG)
            self.logger.info(f"[{i+1}/{len(rosbags)}] Finished: {rosbag['name']}")

        self.logger.info(f"\nConversion complete! Episodes: {self.dataset.num_episodes}")
        return True

    def _prescan_resolutions(self, rosbag):
        """Read a few video packets to initialize decoders (get width/height)."""
        typestore = make_typestore()
        with BagReader(rosbag['path'], typestore=typestore) as reader:
            initialized = set()
            for topic, msg, t_ns in reader.read(topics=self.video_topics_set):
                camera_key = self._get_camera_key_from_topic(topic)
                if camera_key and camera_key not in initialized:
                    tc = self._transcoders[camera_key]
                    tc.feed(bytes(msg.data), msg.encoding, int(msg.width), int(msg.height))
                    if tc.initialized:
                        initialized.add(camera_key)
                if len(initialized) == len(self.video_topics):
                    break


def main():
    parser = argparse.ArgumentParser(description="[macOS] Convert ROS2 bags with markers to LeRobot dataset")
    parser.add_argument("--multibag", action="store_true")
    parser.add_argument("--enforce_four_video_topics", action="store_true")
    parser.add_argument("--input_directory", default="./data/rosbags")
    parser.add_argument("--output", "-o", default="output/dataset")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--task", default="task description")
    parser.add_argument("--stereo-eye", dest="stereo_eye", default="both", choices=STEREO_EYES,
                        help="Which eye to export from side-by-side stereo frames: "
                             "'both' (full frame, default), 'left', or 'right'.")
    args = parser.parse_args()

    if not os.path.exists(args.input_directory):
        print(f"Error: Input directory {args.input_directory} not found!")
        sys.exit(1)

    converter = MultiVideoRosBag2LeRobotConverter(args.input_directory, args.output, args.fps, eye=args.stereo_eye)
    converter.convert_all(args.task, args.multibag, args.enforce_four_video_topics)


if __name__ == "__main__":
    main()
