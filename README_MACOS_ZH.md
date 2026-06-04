# ROS2 Bag 转 LeRobot 数据集 — macOS 版

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![macOS](https://img.shields.io/badge/macOS-12%2B-lightgrey.svg)](https://www.apple.com/macos/)
[![LeRobot 0.3.3](https://img.shields.io/badge/LeRobot-0.3.3-orange.svg)](https://github.com/huggingface/lerobot)
[![Dataset v2.1](https://img.shields.io/badge/Dataset%20Format-v2.1-orange.svg)](https://github.com/huggingface/lerobot)

ROS2 bag → LeRobot 转换脚本的纯 Python 移植版，**无需安装 ROS2**，可直接在 macOS 上运行。

## 📑 目录

- [与 Linux 版本的区别](#-与-linux-版本的区别)
- [系统要求](#-系统要求)
- [快速开始](#-快速开始)
- [使用方法](#-使用方法)
  - [场景一：单个 Rosbag 含多个 Episode](#场景一单个-rosbag-含多个-episode)
  - [场景二：每个 Rosbag 包含单个 Episode](#场景二每个-rosbag-包含单个-episode)
  - [双目图像裁切](#双目图像裁切)
- [视频编码说明](#-视频编码说明)
- [注意事项](#️-注意事项)
- [项目结构](#-项目结构)

## 🔄 与 Linux 版本的区别

| | Linux 脚本 | macOS 脚本 |
|---|---|---|
| Bag 读取 | `rosbag2_py`（ROS2） | `rosbags`（纯 Python） |
| 消息反序列化 | `rclpy` / `rosidl` | `rosbags` typestore |
| 头部相机视频编码 | HEVC（流复制） | HEVC（解码 → 重编码） |
| 腕部相机视频编码 | MJPEG（流复制，在此机器人上失败） | H.264（解码 → 重编码） |
| 双目裁切 | 不支持 | `--stereo-eye left/right/both` |
| 运行环境 | Conda + ROS2 Humble | uv（仅 Python） |

Linux 脚本使用无损流复制，但无法处理本机器人的数据（混合编解码器、HEVC 流从 GOP 中间开始录制）。macOS 脚本对每路相机进行解码并重编码为合法的、可随机访问的 MP4 文件。

## 📋 系统要求

- **操作系统**：macOS 12 及以上（Apple Silicon 或 Intel）
- **Python**：3.11（由 uv 自动管理）
- **uv**：[安装方法](https://docs.astral.sh/uv/getting-started/installation/)
- **ffmpeg**：`brew install ffmpeg`（LeRobot 统计步骤需要）

## 🚀 快速开始

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 安装依赖

```bash
cd rosbag_to_lerobot
uv sync
```

该命令会自动创建 `.venv` 并安装所有依赖，包括项目内置的 LeRobot 分支。

## 📖 使用方法

### 场景一：单个 Rosbag 含多个 Episode

**适用场景**：Rosbag 中包含操作员按下 X 键（开始）和 Y 键（结束）的标记。

**脚本**：`convert_rosbag_with_markers_macos.py`

```bash
# 清除旧缓存
rm -rf ~/.cache/huggingface/lerobot/用户名/数据集名

# 转换数据集
uv run python scripts/convert_rosbag_with_markers_macos.py \
  --input_directory ./data/rosbags/multiepisode_rosbag \
  --output 用户名/数据集名 \
  --fps 30 \
  --task "任务描述"
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--input_directory` | 包含 ROS2 bag 文件（`.db3` 或 `.mcap`）的目录 | `./data/rosbags` |
| `--output` | 输出数据集名称，格式为 `用户名/数据集名` | `output/dataset` |
| `--fps` | 目标帧率 | `30` |
| `--task` | 任务描述文本 | — |
| `--multibag` | 若目录下包含多个 rosbag 子文件夹则加此参数 | — |
| `--enforce_four_video_topics` | 跳过缺少视频话题的 bag | — |
| `--stereo-eye` | 双目裁切：`both`（默认）、`left`、`right` | `both` |

### 场景二：每个 Rosbag 包含单个 Episode

**适用场景**：Rosbag 已预先切分，每个文件只包含一个 episode。

**脚本**：`convert_sliced_rosbags_macos.py`

```bash
# 清除旧缓存
rm -rf ~/.cache/huggingface/lerobot/用户名/数据集名

# 转换数据集
uv run python scripts/convert_sliced_rosbags_macos.py \
  --input_directory ./data/rosbags/single_episode_segments \
  --output 用户名/数据集名 \
  --fps 30 \
  --task "任务描述"
```

**参数说明**：

| 参数 | 说明 | 默认值 |
|---|---|---|
| `--input_directory` | 包含 `rosbag_*` 子目录或 bag 文件的目录 | `./data/rosbags` |
| `--output` | 输出数据集名称 | `output/dataset` |
| `--fps` | 目标帧率 | `30` |
| `--task` | 任务描述文本 | — |
| `--stereo-eye` | 双目裁切：`both`（默认）、`left`、`right` | `both` |

### 双目图像裁切

本机器人的所有相机均以**左右拼接（side-by-side）双目格式**输出图像：两个视角水平拼接在同一帧中。

| 相机 | 完整帧分辨率 | 单眼分辨率 |
|---|---|---|
| 左腕 / 右腕相机 | 2560 × 800 | 1280 × 800 |
| 头部相机（XR） | 3840 × 1920 | 1920 × 1920 |

```bash
# 仅导出左眼（单视角，宽度减半）
uv run python scripts/convert_rosbag_with_markers_macos.py \
  --input_directory ./data/rosbags \
  --output 用户名/数据集名 \
  --task "任务" \
  --stereo-eye left

# 仅导出右眼
  --stereo-eye right

# 导出完整双目帧（默认）
  --stereo-eye both
```

## 🎬 视频编码说明

- **头部相机**（HEVC 源）：重编码为 HEVC（libx265，CRF 23）
- **腕部相机**（MJPEG 源）：重编码为 H.264（libx264，CRF 23）
- 重编码是必要的：HEVC 流从录制中间的帧开始（无关键帧），MJPEG 无法直接封装进 MP4。所有输出 MP4 均为合法、可随机访问的文件，可用 QuickTime、VLC 等播放器直接播放。

## ⚠️ 注意事项

1. **重要**：重新运行转换前，务必删除旧的数据集缓存，否则会产生脏数据。
2. 长时间录制（> 10 分钟，3 路 4K 相机）因重编码耗时较长，属正常现象，仅需一次。
3. 解码器警告 `PPS id out of range` 和 `unable to decode APP fields` 属无害信息，来自 HEVC 流开头的非关键帧，不影响输出结果。

## 📁 项目结构

```
rosbag_to_lerobot/
├── scripts/
│   ├── rosbag_macos_io.py                      # 共享模块：ROS2-free 读包 + 流式转码器
│   ├── convert_rosbag_with_markers_macos.py    # macOS：带 X/Y 标记的 rosbag 转换脚本
│   ├── convert_sliced_rosbags_macos.py         # macOS：预切分 rosbag 转换脚本
│   ├── convert_rosbag_with_markers.py          # Linux/ROS2：原始标记版转换脚本
│   └── convert_sliced_rosbags.py               # Linux/ROS2：原始切分版转换脚本
├── lerobot/                                    # 修改版 LeRobot 库
├── pyproject.toml                              # uv 项目定义
├── uv.lock                                     # 锁定的依赖版本
├── environment.yml                             # Linux Conda 环境（含 ROS2 + CUDA）
├── environment-macos.yml                       # macOS Conda/Micromamba 环境（可选）
├── README.md                                   # Linux / ROS2 使用说明
├── README_MACOS.md                             # macOS 英文说明
└── README_MACOS_ZH.md                          # 本文件
```
