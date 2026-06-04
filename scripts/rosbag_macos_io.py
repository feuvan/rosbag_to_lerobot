# Copyright (c) 2026 Dexteleop Intelligence (灵御智能)
# Licensed under the MIT License.
# See LICENSE file in the project root for full license information.

"""
macOS / ROS2-free I/O helpers for the rosbag -> LeRobot converters.

This module replaces the two things the original scripts used ROS2 for:
  1. Reading rosbag2 files (.db3 / .mcap) -> rosbags.highlevel.AnyReader
  2. Deserializing CDR messages           -> rosbags typestore

It also replaces the lossless stream-copy video path (which cannot handle this
robot's data: mixed codecs across cameras, and HEVC streams that start mid-GOP)
with a streaming decode -> re-encode pipeline that produces valid, seekable MP4s:
  - cameras recorded as HEVC are re-encoded to HEVC (libx265)
  - cameras recorded as MJPEG / H.264 are re-encoded to H.264 (libx264)

Only `av` (PyAV), `numpy` and `rosbags` are required - no ROS2.
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

logger = logging.getLogger(__name__)

# The only message type not shipped with a default ROS2 distribution.
FFMPEG_PACKET_MSG = """
std_msgs/Header header
int32 width
int32 height
string encoding
uint64 pts
uint8 flags
bool is_bigendian
uint8[] data
"""
FFMPEG_PACKET_TYPE = "ffmpeg_image_transport_msgs/msg/FFMPEGPacket"


def make_typestore():
    """Return a ROS2 Humble typestore with the FFMPEGPacket type registered."""
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    typestore.register(get_types_from_msg(FFMPEG_PACKET_MSG, FFMPEG_PACKET_TYPE))
    return typestore


def normalize_src_codec(encoding: str) -> str:
    """Map a FFMPEGPacket encoding string to a PyAV decoder name."""
    token = encoding.split(";")[0].lower()
    if token in ("hevc", "h265", "x265", "hevc_vaapi"):
        return "hevc"
    if token in ("h264", "avc", "x264"):
        return "h264"
    return token  # e.g. "mjpeg"


def output_codec_for(src_codec: str) -> str:
    """Choose the re-encode target: keep HEVC sources as HEVC, everything else H.264."""
    return "libx265" if src_codec == "hevc" else "libx264"


# Valid choices for the side-by-side stereo crop.
STEREO_EYES = ("both", "left", "right")


def crop_eye(rgb: np.ndarray, eye: str) -> np.ndarray:
    """
    Crop a side-by-side stereo frame to a single eye.

    All cameras on this robot publish a horizontally-stacked stereo pair in one
    frame (width = 2 * eye_width). `eye="both"` returns the frame unchanged;
    `"left"`/`"right"` return the corresponding half.
    """
    if eye == "both":
        return rgb
    w = rgb.shape[1]
    half = w // 2
    if eye == "left":
        return rgb[:, :half, :]
    if eye == "right":
        return rgb[:, half:, :]
    raise ValueError(f"invalid eye '{eye}', expected one of {STEREO_EYES}")


def eye_output_size(width: int, height: int, eye: str) -> tuple[int, int]:
    """Return (width, height) of the encoded frame for the chosen eye."""
    if eye == "both":
        return width, height
    return width // 2, height


@dataclass
class TopicInfo:
    name: str
    msgtype: str


class BagReader:
    """
    ROS2-free replacement for rosbag2_py.SequentialReader.

    Opens a single rosbag2 file (.db3 or .mcap) and yields messages in the bag's
    global timestamp order, matching the behavior the original scripts relied on.

    Usage:
        with BagReader(bag_path) as reader:
            topic_types = reader.topic_types  # {topic_name: msgtype_str}
            for topic, msg, t_ns in reader.read():
                ...
    """

    def __init__(self, bag_path: str | Path, typestore=None):
        # AnyReader takes the directory containing the bag + metadata.yaml.
        self.bag_path = Path(bag_path)
        self.bag_dir = self.bag_path if self.bag_path.is_dir() else self.bag_path.parent
        self.typestore = typestore or make_typestore()
        self._reader = AnyReader([self.bag_dir], default_typestore=self.typestore)
        self._opened = False

    def __enter__(self):
        self._reader.open()
        self._opened = True
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._opened:
            self._reader.close()
            self._opened = False
        return False

    @property
    def topic_types(self) -> dict[str, str]:
        return {c.topic: c.msgtype for c in self._reader.connections}

    def read(self, topics: set[str] | None = None):
        """Yield (topic, deserialized_msg, timestamp_ns) in global time order."""
        if topics is None:
            conns = list(self._reader.connections)
        else:
            conns = [c for c in self._reader.connections if c.topic in topics]
        for conn, t_ns, raw in self._reader.messages(connections=conns):
            msg = self._reader.deserialize(raw, conn.msgtype)
            yield conn.topic, msg, t_ns


class _ContinuousDecoder:
    """
    Per-camera decoder fed EVERY packet in arrival order.

    Inter-frame codecs (HEVC here) require the full packet stream; you cannot
    feed a subsampled stream. Call feed() for every packet, then snapshot()
    at each output-grid step to get the most recently decoded frame.
    """

    def __init__(self):
        self._dec = None
        self.latest = None  # most recent decoded av.VideoFrame, or None
        self.width = None
        self.height = None
        self.src_codec = None

    def feed(self, data: bytes, encoding: str, width: int, height: int) -> None:
        if self._dec is None:
            self.src_codec = normalize_src_codec(encoding)
            self._dec = av.codec.CodecContext.create(self.src_codec, "r")
            self.width = int(width)
            self.height = int(height)
        for frame in self._dec.decode(av.packet.Packet(data)):
            self.latest = frame

    def snapshot(self) -> np.ndarray:
        """Return the latest decoded frame as HxWx3 uint8 RGB, or black if none yet."""
        if self.latest is None:
            return np.zeros((self.height, self.width, 3), dtype=np.uint8)
        return self.latest.to_ndarray(format="rgb24")


class _StreamingMp4Encoder:
    """Streaming H.264/HEVC MP4 encoder. Push RGB frames one at a time."""

    def __init__(self, out_path: Path, width: int, height: int, fps: int, out_codec: str):
        self.out_path = Path(out_path)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.width = width
        self.height = height
        self.fps = fps
        self.out_codec = out_codec
        self.n_written = 0
        self._container = av.open(str(self.out_path), "w")
        self._stream = self._container.add_stream(out_codec, rate=fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = "yuv420p"
        self._stream.codec_context.time_base = Fraction(1, fps)
        self._stream.options = {"crf": "23"}

    def push(self, rgb: np.ndarray) -> None:
        arr = np.ascontiguousarray(rgb)
        frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
        frame = frame.reformat(width=self.width, height=self.height, format="yuv420p")
        frame.pts = self.n_written
        frame.time_base = Fraction(1, self.fps)
        for pkt in self._stream.encode(frame):
            self._container.mux(pkt)
        self.n_written += 1

    def close(self) -> None:
        for pkt in self._stream.encode(None):
            self._container.mux(pkt)
        self._container.close()
        if not self.out_path.exists():
            raise IOError(f"Video file was not created: {self.out_path}")


class CameraTranscoder:
    """
    One camera's video pipeline for ROS2-free conversion.

    Holds a PERSISTENT decoder (fed every packet, in order, so inter-frame codecs
    like HEVC decode correctly and stay warm across episodes) plus a PER-EPISODE
    streaming MP4 encoder. A one-frame lag is kept so the final frame can be
    dropped cheaply (`trim_final`) to match the state/action frame count, mirroring
    the original VideoPacketBuffer.delete_final_packet semantics.

    Memory stays bounded: at most one decoded frame is held; everything else is
    streamed straight into the encoder.

    Lifecycle per episode:
        begin_episode()
        feed(...)        # for every video packet (also call between episodes)
        commit()         # once per output-grid step while recording
        trim_final()     # optional, drop the last committed frame
        finalize(...)    # flush + write MP4 to the LeRobot path
      or discard()       # throw the episode away
    """

    def __init__(self, camera_name: str, fps: int, root_dir: Path, eye: str = "both"):
        assert eye in STEREO_EYES, f"eye must be one of {STEREO_EYES}"
        self.camera_name = camera_name
        self.fps = fps
        self.root_dir = Path(root_dir)
        self.eye = eye
        self._decoder = _ContinuousDecoder()
        self._encoder: _StreamingMp4Encoder | None = None
        self._tmp_path: Path | None = None
        self._held: np.ndarray | None = None
        self._count = 0  # committed frames not yet trimmed (held frame included)

    def feed(self, data: bytes, encoding: str, width: int, height: int) -> None:
        """Feed one compressed packet to the decoder. Call for EVERY packet."""
        self._decoder.feed(data, encoding, width, height)

    @property
    def initialized(self) -> bool:
        return self._decoder.width is not None

    def begin_episode(self) -> None:
        self._discard_encoder()
        self._held = None
        self._count = 0

    @property
    def output_width(self) -> int | None:
        if self._decoder.width is None:
            return None
        return eye_output_size(self._decoder.width, self._decoder.height, self.eye)[0]

    @property
    def output_height(self) -> int | None:
        return self._decoder.height

    def commit(self) -> None:
        """Snapshot the latest decoded frame as one output-grid frame."""
        snap = crop_eye(self._decoder.snapshot(), self.eye)
        if self._held is not None:
            self._push(self._held)
        self._held = snap
        self._count += 1

    def trim_final(self) -> None:
        if self._held is not None:
            self._held = None
            self._count -= 1

    @property
    def length(self) -> int:
        return self._count

    def discard(self) -> None:
        self._discard_encoder()
        self._held = None
        self._count = 0

    def finalize(self, episode_index: int, dataset_meta) -> Path:
        """Flush held frame, close encoder, move MP4 to the LeRobot video path."""
        if self._held is not None:
            self._push(self._held)
            self._held = None
        if self._encoder is not None:
            self._encoder.close()
        video_key = f"observation.images.{self.camera_name}"
        final_path = self.root_dir / dataset_meta.get_video_file_path(episode_index, video_key)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        if self._tmp_path and self._tmp_path.exists():
            shutil.move(str(self._tmp_path), str(final_path))
        self._encoder = None
        self._tmp_path = None
        logger.info(
            "Wrote %d frames -> %s (%s)",
            self._count, final_path, self._decoder.src_codec,
        )
        return final_path

    # --- internal ---

    def _push(self, rgb: np.ndarray) -> None:
        if self._encoder is None:
            self._tmp_path = Path(tempfile.mktemp(suffix=".mp4", prefix=f"{self.camera_name}_"))
            out_codec = output_codec_for(self._decoder.src_codec)
            h, w = rgb.shape[:2]  # use actual (possibly cropped) dimensions
            self._encoder = _StreamingMp4Encoder(
                self._tmp_path, w, h, self.fps, out_codec,
            )
        self._encoder.push(rgb)

    def _discard_encoder(self) -> None:
        if self._encoder is not None:
            try:
                self._encoder.close()
            except Exception:
                pass
            self._encoder = None
        if self._tmp_path and self._tmp_path.exists():
            self._tmp_path.unlink()
            self._tmp_path = None
