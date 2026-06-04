# ROS2 Bag to LeRobot Dataset Converter — macOS

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![macOS](https://img.shields.io/badge/macOS-12%2B-lightgrey.svg)](https://www.apple.com/macos/)
[![LeRobot 0.3.3](https://img.shields.io/badge/LeRobot-0.3.3-orange.svg)](https://github.com/huggingface/lerobot)
[![Dataset v2.1](https://img.shields.io/badge/Dataset%20Format-v2.1-orange.svg)](https://github.com/huggingface/lerobot)

Pure-Python port of the ROS2 bag → LeRobot converters. Runs on macOS with **no ROS2 installation required**.

## 📑 Table of Contents

- [How it Differs from the Linux Version](#-how-it-differs-from-the-linux-version)
- [System Requirements](#-system-requirements)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Scenario 1: Single Rosbag with Multiple Episodes](#scenario-1-single-rosbag-with-multiple-episodes)
  - [Scenario 2: Each Rosbag Contains Single Episode](#scenario-2-each-rosbag-contains-single-episode)
  - [Stereo Eye Selection](#stereo-eye-selection)
- [Video Encoding Notes](#-video-encoding-notes)
- [Important Notes](#️-important-notes)
- [Project Structure](#-project-structure)

## 🔄 How it Differs from the Linux Version

| | Linux scripts | macOS scripts |
|---|---|---|
| Bag reading | `rosbag2_py` (ROS2) | `rosbags` (pure Python) |
| Message deserialization | `rclpy` / `rosidl` | `rosbags` typestore |
| Head cam output codec | HEVC (stream-copy) | HEVC (decode → re-encode) |
| Stereo cam output codec | MJPEG (stream-copy) | H.264 (decode → re-encode) |
| Stereo eye crop | not supported | `--stereo-eye left/right/both` |
| Environment | Conda + ROS2 Humble | uv (Python only) |

The Linux scripts use lossless stream-copy which cannot handle this robot's data (mixed codecs, HEVC streams that start mid-GOP). The macOS scripts decode and re-encode each camera to a valid, seekable MP4.

## 📋 System Requirements

- **OS**: macOS 12+ (Apple Silicon or Intel)
- **Python**: 3.11 (managed automatically by uv)
- **uv**: [Install here](https://docs.astral.sh/uv/getting-started/installation/)
- **ffmpeg**: `brew install ffmpeg` (required by the lerobot fork's stats step)

## 🚀 Quick Start

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install Dependencies

```bash
cd rosbag_to_lerobot
uv sync
```

This creates a `.venv` and installs all dependencies including the bundled LeRobot fork.

## 📖 Usage

### Scenario 1: Single Rosbag with Multiple Episodes

**Use case**: Rosbag contains operator start (X button) and end (Y button) markers.

**Script**: `convert_rosbag_with_markers_macos.py`

```bash
# Clear previous cache
rm -rf ~/.cache/huggingface/lerobot/username/dataset_name

# Convert dataset
uv run python scripts/convert_rosbag_with_markers_macos.py \
  --input_directory ./data/rosbags/multiepisode_rosbag \
  --output username/dataset_name \
  --fps 30 \
  --task "task description"
```

**Parameters**:
- `--input_directory`: Directory containing ROS2 bag files (`.db3` or `.mcap`)
- `--output`: Output dataset name (format: `username/dataset_name`)
- `--fps`: Target frame rate (default: 30)
- `--task`: Task description
- `--multibag`: Use if the directory contains multiple rosbag folders
- `--enforce_four_video_topics`: Skip bags that are missing any video topic
- `--stereo-eye`: Which eye to export — `both` (default), `left`, or `right`

### Scenario 2: Each Rosbag Contains Single Episode

**Use case**: Rosbags are pre-sliced, each containing one episode.

**Script**: `convert_sliced_rosbags_macos.py`

```bash
# Clear previous cache
rm -rf ~/.cache/huggingface/lerobot/username/dataset_name

# Convert dataset
uv run python scripts/convert_sliced_rosbags_macos.py \
  --input_directory ./data/rosbags/single_episode_segments \
  --output username/dataset_name \
  --fps 30 \
  --task "task description"
```

**Parameters**:
- `--input_directory`: Directory containing rosbag segment folders (`rosbag_*`) or bag files
- `--output`: Output dataset name
- `--fps`: Target frame rate (default: 30)
- `--task`: Task description
- `--stereo-eye`: Which eye to export — `both` (default), `left`, or `right`

### Stereo Eye Selection

All cameras on this robot publish **side-by-side stereo pairs** — two eye views stitched horizontally into a single frame:

| Camera | Full frame | Single eye |
|---|---|---|
| Left wrist / Right wrist | 2560 × 800 | 1280 × 800 |
| Head (XR) | 3840 × 1920 | 1920 × 1920 |

```bash
# Export left eye only (half-width, single view per camera)
uv run python scripts/convert_rosbag_with_markers_macos.py \
  --input_directory ./data/rosbags \
  --output username/dataset_name \
  --task "task" \
  --stereo-eye left

# Export right eye only
  --stereo-eye right

# Export full stereo frame (default)
  --stereo-eye both
```

## 🎬 Video Encoding Notes

- **Head camera** (HEVC source): re-encoded to HEVC (libx265, CRF 23)
- **Wrist cameras** (MJPEG source): re-encoded to H.264 (libx264, CRF 23)
- Re-encoding is necessary because the HEVC stream starts mid-GOP (no keyframe at recording start) and MJPEG cannot be stream-copied into MP4. All output MP4s are fully seekable.
- Output videos can be played in QuickTime, VLC, or any standard player.

## ⚠️ Important Notes

1. **Critical**: Always delete the dataset cache before re-running to avoid stale data.
2. Long recordings (> 10 min, 3 cameras at 4K) take significant time due to re-encoding. This is a one-time cost.
3. Decoder warnings like `PPS id out of range` and `unable to decode APP fields` are harmless — they occur during the initial mid-GOP frames before the first keyframe arrives.

## 📁 Project Structure

```
rosbag_to_lerobot/
├── scripts/
│   ├── rosbag_macos_io.py                      # Shared: ROS2-free bag reader + streaming transcoder
│   ├── convert_rosbag_with_markers_macos.py    # macOS: convert rosbags with X/Y episode markers
│   ├── convert_sliced_rosbags_macos.py         # macOS: convert pre-sliced rosbag segments
│   ├── convert_rosbag_with_markers.py          # Linux/ROS2: original marker-based converter
│   └── convert_sliced_rosbags.py               # Linux/ROS2: original sliced-bag converter
├── lerobot/                                    # Modified LeRobot library
├── pyproject.toml                              # uv project definition
├── uv.lock                                     # Locked dependencies
├── environment.yml                             # Linux Conda environment (ROS2 + CUDA)
├── environment-macos.yml                       # macOS Conda/Micromamba environment (optional)
├── README.md                                   # Linux / ROS2 usage
├── README_MACOS.md                             # This file
└── README_MACOS_ZH.md                          # 中文版本
```
