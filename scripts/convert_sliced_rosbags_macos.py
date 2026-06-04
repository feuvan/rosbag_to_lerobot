#!/usr/bin/env python3
# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
macOS (ROS2-free) port of convert_sliced_rosbags.py.

Converts multiple pre-sliced ROS2 bag files (each containing one episode) to a
LeRobot dataset. Uses pure-Python `rosbags` + streaming decode->re-encode video.

Usage:
    uv run python scripts/convert_sliced_rosbags_macos.py \
        --input_directory ./data/rosbags/segments \
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
from collections import defaultdict
from time import time

from rosbag_macos_io import BagReader, CameraTranscoder, make_typestore, STEREO_EYES

from lerobot.record import sample_frames_from_video
from lerobot.datasets.lerobot_dataset import LeRobotDataset

STATE_ACTION_DIM = 62


class MultiVideoRosBag2LeRobotConverter:

    def __init__(self, input_directory: str, output_repo_id: str, fps: int = 30, eye: str = "both"):
        self.input_directory = Path(input_directory)
        self.output_repo_id = output_repo_id
        self.fps = fps
        self.eye = eye
        self.frame_duration = 1.0 / self.fps

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
        self.current_episode_index = 0

    def _get_camera_key_from_topic(self, topic: str) -> str | None:
        for key, t in self.video_topics.items():
            if t == topic:
                return key
        return None

    def discover_bag_segments(self):
        segments = []
        if self.input_directory.is_dir():
            segment_dirs = sorted(self.input_directory.glob("rosbag_*"))
            if segment_dirs:
                for sd in segment_dirs:
                    bag_files = list(sd.glob("*.mcap")) or list(sd.glob("*.db3"))
                    if bag_files:
                        segments.append({'name': sd.name, 'path': str(sd), 'bag_file': str(bag_files[0])})
            else:
                bag_files = sorted(
                    list(self.input_directory.glob("*.mcap")) or list(self.input_directory.glob("*.db3"))
                )
                for i, bf in enumerate(bag_files):
                    segments.append({'name': f"episode_{i:03d}", 'path': str(bf.parent), 'bag_file': str(bf)})
        self.logger.info(f"Discovered {len(segments)} bag segments")
        return segments

    def setup_features(self, width_height_map: dict[str, tuple[int, int]]):
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
            h, w = width_height_map.get(camera_key, (1920, 3840))
            if self.eye != "both":
                w = w // 2
            features[f"observation.images.{camera_key}"] = {
                "dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"],
            }
        return features

    def create_dataset(self, width_height_map):
        features = self.setup_features(width_height_map)
        self.dataset = LeRobotDataset.create(
            repo_id=self.output_repo_id, fps=self.fps, features=features,
            robot_type="dual_arm_robot",
        )

    def _prescan_resolutions(self, segment) -> dict[str, tuple[int, int]]:
        """Read a few video packets from a bag to get camera resolutions."""
        wh_map = {}
        typestore = make_typestore()
        with BagReader(segment['path'], typestore=typestore) as reader:
            for topic, msg, t_ns in reader.read(topics=self.video_topics_set):
                camera_key = self._get_camera_key_from_topic(topic)
                if camera_key and camera_key not in wh_map:
                    wh_map[camera_key] = (int(msg.height), int(msg.width))
                if len(wh_map) == len(self.video_topics):
                    break
        return wh_map

    def read_bag_messages(self, bag_path: str):
        """Read all messages, return (state/action dict, video_packets dict)."""
        all_messages = defaultdict(list)
        video_packets = defaultdict(list)
        typestore = make_typestore()

        with BagReader(bag_path, typestore=typestore) as reader:
            for topic, msg, t_ns in reader.read(topics=self.all_topics_set):
                timestamp_s = t_ns / 1e9
                if topic in self.video_topics_set:
                    camera_key = self._get_camera_key_from_topic(topic)
                    if camera_key:
                        video_packets[camera_key].append({
                            'data': bytes(msg.data),
                            'encoding': msg.encoding,
                            'width': int(msg.width),
                            'height': int(msg.height),
                            'timestamp': timestamp_s,
                        })
                else:
                    all_messages[topic].append((timestamp_s, msg))

        return all_messages, video_packets

    def _find_closest_msg(self, topic_messages, target_time):
        best = None
        best_diff = float('inf')
        for ts, msg in topic_messages:
            diff = abs(ts - target_time)
            if diff < best_diff:
                best_diff = diff
                best = msg
        return best

    def _find_closest_video_idx(self, packets, target_time):
        best_idx = 0
        best_diff = float('inf')
        for i, p in enumerate(packets):
            diff = abs(p['timestamp'] - target_time)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
        return best_idx

    def _extract_joint_state(self, msg):
        return list(msg.position), list(msg.velocity), list(msg.effort)

    def _extract_ee_pose(self, msg):
        return [msg.position.x, msg.position.y, msg.position.z,
                msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]

    def create_frame_at_time(self, current_time, all_messages, video_packets):
        state_data = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        action_data = np.zeros(STATE_ACTION_DIM, dtype=np.float32)
        video_frame_idx = {}

        # State
        left_js = self._find_closest_msg(all_messages.get('/left_arm/joint_states', []), current_time)
        if left_js:
            pos, vel, eff = self._extract_joint_state(left_js)
            state_data[0:7] = pos[:7]; state_data[16:23] = vel[:7]; state_data[32:39] = eff[:7]

        right_js = self._find_closest_msg(all_messages.get('/right_arm/joint_states', []), current_time)
        if right_js:
            pos, vel, eff = self._extract_joint_state(right_js)
            state_data[8:15] = pos[:7]; state_data[24:31] = vel[:7]; state_data[40:47] = eff[:7]

        left_grip = self._find_closest_msg(all_messages.get('/left_gripper/joint_states', []), current_time)
        if left_grip:
            pos, vel, eff = self._extract_joint_state(left_grip)
            if len(pos) > 0:
                state_data[7] = pos[0]
                state_data[23] = vel[0] if len(vel) > 0 else 0
                state_data[39] = eff[0] if len(eff) > 0 else 0

        right_grip = self._find_closest_msg(all_messages.get('/right_gripper/joint_states', []), current_time)
        if right_grip:
            pos, vel, eff = self._extract_joint_state(right_grip)
            if len(pos) > 0:
                state_data[15] = pos[0]
                state_data[31] = vel[0] if len(vel) > 0 else 0
                state_data[47] = eff[0] if len(eff) > 0 else 0

        left_ee = self._find_closest_msg(all_messages.get('/left_arm/current_ee_pose', []), current_time)
        if left_ee:
            state_data[48:55] = self._extract_ee_pose(left_ee)

        right_ee = self._find_closest_msg(all_messages.get('/right_arm/current_ee_pose', []), current_time)
        if right_ee:
            state_data[55:62] = self._extract_ee_pose(right_ee)

        # Action
        left_cmd = self._find_closest_msg(all_messages.get('/left_arm/joint_cmd', []), current_time)
        if left_cmd:
            pos, vel, eff = self._extract_joint_state(left_cmd)
            action_data[0:7] = pos[:7]; action_data[16:23] = vel[:7]; action_data[32:39] = eff[:7]

        right_cmd = self._find_closest_msg(all_messages.get('/right_arm/joint_cmd', []), current_time)
        if right_cmd:
            pos, vel, eff = self._extract_joint_state(right_cmd)
            action_data[8:15] = pos[:7]; action_data[24:31] = vel[:7]; action_data[40:47] = eff[:7]

        left_grip_cmd = self._find_closest_msg(all_messages.get('/left_gripper/joint_cmd', []), current_time)
        if left_grip_cmd:
            pos, vel, eff = self._extract_joint_state(left_grip_cmd)
            if len(pos) > 0:
                action_data[7] = pos[0]
                action_data[23] = vel[0] if len(vel) > 0 else 0
                action_data[39] = eff[0] if len(eff) > 0 else 0

        right_grip_cmd = self._find_closest_msg(all_messages.get('/right_gripper/joint_cmd', []), current_time)
        if right_grip_cmd:
            pos, vel, eff = self._extract_joint_state(right_grip_cmd)
            if len(pos) > 0:
                action_data[15] = pos[0]
                action_data[31] = vel[0] if len(vel) > 0 else 0
                action_data[47] = eff[0] if len(eff) > 0 else 0

        left_target = self._find_closest_msg(all_messages.get('/left_arm/target_ee_pose', []), current_time)
        if left_target:
            action_data[48:55] = self._extract_ee_pose(left_target)

        right_target = self._find_closest_msg(all_messages.get('/right_arm/target_ee_pose', []), current_time)
        if right_target:
            action_data[55:62] = self._extract_ee_pose(right_target)

        # Video
        for camera_key, packets in video_packets.items():
            idx = self._find_closest_video_idx(packets, current_time)
            video_frame_idx[camera_key] = idx

        frame_data = {
            "observation.state": state_data,
            "action": action_data,
        }
        for camera_key in self.video_topics.keys():
            wh = video_packets.get(camera_key)
            if wh and len(wh) > 0:
                h, w = wh[0]['height'], wh[0]['width']
            else:
                h, w = 1920, 3840
            if self.eye != "both":
                w = w // 2
            frame_data[f"observation.images.{camera_key}"] = np.zeros((h, w, 3), dtype=np.uint8)

        return frame_data, video_frame_idx

    def convert_single_bag(self, segment, task_description):
        self.logger.info(f"\n=== Processing {segment['name']} ===")
        try:
            all_messages, video_packets = self.read_bag_messages(segment['path'])

            if not all_messages and not video_packets:
                self.logger.error(f"No messages in {segment['name']}")
                return False

            all_timestamps = []
            for topic_messages in all_messages.values():
                all_timestamps.extend([ts for ts, _ in topic_messages])
            if not all_timestamps:
                self.logger.error(f"No timestamps in {segment['name']}")
                return False

            start_time = min(all_timestamps)
            end_time = max(all_timestamps)
            self.logger.info(f"  Duration: {end_time - start_time:.3f}s")

            self.dataset.episode_buffer = self.dataset.create_episode_buffer(self.current_episode_index)

            current_time = start_time
            frame_count = 0
            video_frame_indices = defaultdict(list)

            while current_time <= end_time:
                frame_data, vfi = self.create_frame_at_time(current_time, all_messages, video_packets)
                for cam, idx in vfi.items():
                    video_frame_indices[cam].append(idx)

                if frame_data:
                    self.dataset.add_frame(frame_data, task_description,
                                           current_time - start_time - self.frame_duration)
                    frame_count += 1
                    if frame_count % 1000 == 0:
                        self.logger.info(f"    Processed {frame_count} frames")

                current_time += self.frame_duration

            # Encode video: decode selected packets -> re-encode to MP4
            for camera_key in self.video_topics.keys():
                packets = video_packets.get(camera_key, [])
                indices = video_frame_indices.get(camera_key, [])
                if not packets or not indices:
                    continue

                tc = CameraTranscoder(camera_key, self.fps, self.dataset.root, eye=self.eye)
                tc.begin_episode()

                # Feed ALL packets in order to decoder, commit only at selected indices
                selected_set = set(indices)
                for i, pkt in enumerate(packets):
                    tc.feed(pkt['data'], pkt['encoding'], pkt['width'], pkt['height'])
                    if i in selected_set:
                        tc.commit()

                video_path = tc.finalize(self.dataset.num_episodes, self.dataset.meta)
                episode_length = tc.length

                video_key = f"observation.images.{camera_key}"
                try:
                    sampled_frames = sample_frames_from_video(
                        video_path=video_path,
                        episode_length=episode_length,
                        fps=self.dataset.fps,
                        width=tc.output_width,
                        height=tc.output_height,
                    )
                    self.dataset.episode_buffer[video_key] = sampled_frames
                    self.logger.info(f"  Sampled {len(sampled_frames)} frames from {camera_key}")
                except Exception as e:
                    self.logger.error(f"  Failed to sample frames from {camera_key}: {e}")

            if frame_count > 0:
                self.dataset.save_episode()
                self.logger.info(f"  Episode {self.current_episode_index} complete: {frame_count} frames")
                self.current_episode_index += 1
                return True
            else:
                self.logger.error(f"  No valid frames for {segment['name']}")
                return False

        finally:
            shutil.rmtree(self.dataset.root / 'images', ignore_errors=True)

    def convert_all(self, task_description="task description"):
        self.logger.info(f"Starting multi-bag conversion: {self.input_directory}")
        segments = self.discover_bag_segments()
        if not segments:
            self.logger.error("No bag segments found!")
            return False

        wh_map = self._prescan_resolutions(segments[0])
        self.create_dataset(wh_map)

        successful = 0
        t0 = time()
        for segment in segments:
            if self.convert_single_bag(segment, task_description):
                successful += 1

        elapsed = time() - t0
        self.logger.info(f"\nConversion complete! {successful}/{len(segments)} episodes in {elapsed:.1f}s")
        return successful > 0


def main():
    parser = argparse.ArgumentParser(description="[macOS] Convert pre-sliced ROS2 bags to LeRobot dataset")
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
    success = converter.convert_all(args.task)
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
