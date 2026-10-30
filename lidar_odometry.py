"""Small, dependency-free LiDAR odometry bridge for the reproduced policy.

The odometry process sends one newline-delimited JSON object per pose over UDP:
``{"t": 1710000000.1, "position": [x,y,z],
"quaternion": [w,x,y,z], "frame": "lidar"}``.

``frame=lidar`` is converted with the calibrated ``lidar->pelvis`` extrinsic;
``frame=pelvis`` is already a root pose.  The policy consumes map->pelvis pose.
"""
from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass

import numpy as np


def qnorm(q):
    q = np.asarray(q, np.float32)
    return q / max(float(np.linalg.norm(q)), 1e-8)


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw-ax*bx-ay*by-az*bz,
                     aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx,
                     aw*bz+ax*by-ay*bx+az*bw], np.float32)


def qapply(q, v):
    q = qnorm(q)
    return qmul(qmul(q, np.r_[0., np.asarray(v, np.float32)]),
                np.array([q[0], -q[1], -q[2], -q[3]], np.float32))[1:]


@dataclass(frozen=True)
class PoseSample:
    stamp: float
    position: np.ndarray
    quaternion: np.ndarray
    frame: str


class UdpOdometry:
    """Latest-pose UDP receiver with freshness and finite-value checks."""
    def __init__(self, bind: str, port: int, max_age: float = 0.15,
                 lidar_to_pelvis_pos=(0., 0., 0.),
                 lidar_to_pelvis_quat=(1., 0., 0., 0.)):
        self.bind, self.port, self.max_age = bind, int(port), float(max_age)
        self.lp = np.asarray(lidar_to_pelvis_pos, np.float32)
        self.lq = qnorm(lidar_to_pelvis_quat)
        self._lock = threading.Lock()
        self._latest: PoseSample | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="lidar-odometry", daemon=True)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((bind, self.port))
        self._sock.settimeout(0.2)
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                raw, _ = self._sock.recvfrom(8192)
                obj = json.loads(raw.decode("utf-8"))
                p = np.asarray(obj["position"], np.float32).reshape(3)
                q = qnorm(np.asarray(obj["quaternion"], np.float32).reshape(4))
                frame = str(obj.get("frame", "lidar")).lower()
                # Freshness is measured locally; the sender's timestamp is
                # retained only as metadata by the sender and may use Unix time.
                stamp = time.monotonic()
                if frame not in ("lidar", "pelvis", "base") or not np.isfinite(p).all() or not np.isfinite(q).all():
                    continue
                with self._lock:
                    self._latest = PoseSample(stamp, p, q, frame)
            except (socket.timeout, ValueError, KeyError, json.JSONDecodeError):
                continue
            except OSError:
                break

    def latest(self) -> tuple[np.ndarray, np.ndarray, float] | None:
        with self._lock:
            s = self._latest
        if s is None:
            return None
        age = time.monotonic() - s.stamp
        if age > self.max_age:
            return None
        if s.frame in ("pelvis", "base"):
            return s.position.copy(), s.quaternion.copy(), max(age, 0.0)
        # T_map_pelvis = T_map_lidar * T_lidar_pelvis.
        p = s.position + qapply(s.quaternion, self.lp)
        q = qmul(s.quaternion, self.lq)
        return p.astype(np.float32), qnorm(q), max(age, 0.0)

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        finally:
            self._thread.join(timeout=1.0)
