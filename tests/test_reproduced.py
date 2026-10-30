import json
import socket
import time
import unittest
from pathlib import Path

import numpy as np

from lidar_odometry import UdpOdometry
from reproduced_controller import A, START, ReproducedController, State, gravity

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / 'assets'
POLICY = ASSETS / 'reproduced_policy_lafan1_anchor_5080.onnx'


class ReproducedPipelineTests(unittest.TestCase):
    def test_policy_shapes_and_motion_fields(self):
        p = ReproducedController(ASSETS, POLICY, 'walk1_subject1', 8.0)
        self.assertEqual(p.goal_dim, 689)
        with np.load(ASSETS / 'reproduced_motions' / 'walk1_subject1.npz') as d:
            for key in ('goal', 'joint_pos', 'joint_vel', 'root_pos', 'root_quat',
                        'root_ang_vel_b', 'anchor_pos_w', 'anchor_quat_w'):
                self.assertIn(key, d.files)
        self.assertEqual(len(p.joint_names), 29)
        np.testing.assert_allclose(gravity([1, 0, 0, 0]), [0, 0, -1])

    def test_start_alignment_and_first_action(self):
        p = ReproducedController(ASSETS, POLICY, 'walk1_subject1', 1.0)
        with np.load(ASSETS / 'reproduced_motions' / 'walk1_subject1.npz') as d:
            q = d['joint_pos'][0].astype(np.float32)
            dq = np.zeros(29, np.float32)
            root_pos = d['root_pos'][0]
            root_quat = d['root_quat'][0]
        p.set_root(root_pos, root_quat)
        self.assertTrue(np.isfinite(p.tick(0.0, q, dq, root_quat, np.zeros(3), START)).all())
        target = p.tick(1.1, q, dq, root_quat, np.zeros(3), 0)
        self.assertEqual(p.state, State.READY)
        self.assertTrue(np.isfinite(target).all())
        action = p.tick(1.12, q, dq, root_quat, np.zeros(3), A)
        self.assertEqual(p.state, State.POLICY)
        self.assertEqual(action.shape, (29,))
        self.assertTrue(np.isfinite(action).all())

    def test_udp_lidar_pose_conversion(self):
        port = 18765
        odom = UdpOdometry('127.0.0.1', port, 0.5, (1, 0, 0), (1, 0, 0, 0))
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.sendto(json.dumps({'position': [2, 3, 4], 'quaternion': [1, 0, 0, 0], 'frame': 'lidar'}).encode(), ('127.0.0.1', port))
            sample = None
            for _ in range(20):
                sample = odom.latest()
                if sample is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(sample)
            np.testing.assert_allclose(sample[0], [3, 3, 4])
        finally:
            odom.close()


if __name__ == '__main__':
    unittest.main()
