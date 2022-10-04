#!/usr/bin/env python3
import time
from dataclasses import dataclass

import numpy as np
import ros_numpy
import rospy
import tf
from geometry_msgs.msg import Pose, PoseStamped
from mohou_ros_utils.utils import CoordinateTransform
from scipy.optimize import minimize
from sensor_msgs.msg import PointCloud2


@dataclass
class Circle:
    center: np.ndarray
    radius: float


def find_circle(X: np.ndarray) -> Circle:
    assert X.ndim == 2
    assert X.shape[1] == 2

    def fun(arg):
        c1, c2, r = arg
        C = np.array([c1, c2])
        diffs = np.sqrt(np.sum((X - C) ** 2, axis=1)) - r
        cost = np.sum(diffs**2)
        return cost

    pcloud_center = np.mean(X, axis=0)
    r = 0.1
    x0 = np.hstack([pcloud_center, r])
    time.time()
    res = minimize(fun, x0=x0, method="BFGS")
    return Circle(res.x[:2], res.x[2])


class BowlCenterFinder:
    listener: tf.TransformListener
    publisher: rospy.Publisher
    n_window: int
    center_list = []

    def __init__(self, n_window: int = 10):
        self.listener = tf.TransformListener()
        self.publisher = rospy.Publisher("bowl_center", PoseStamped)
        self.n_window = n_window
        rospy.Subscriber("/hsi_filter/output", PointCloud2, self.callback)

    def callback(self, msg: PointCloud2):
        target = "base_link"
        source = "head_mount_kinect_rgb_optical_frame"
        while True:
            try:
                (trans, rot) = self.listener.lookupTransform(target, source, rospy.Time(0))
                break

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                pass

        pose = Pose()
        pose.position.x, pose.position.y, pose.position.z = trans
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = rot
        X_source = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(msg)  # type: ignore
        transform = CoordinateTransform.from_ros_pose(pose)
        X = transform(X_source)
        center = np.mean(X, axis=0)

        r_filter = 0.15
        is_inside = np.sum((X[:, :2] - center[:2]) ** 2, axis=1) < r_filter**2
        X_bowl = X[is_inside]
        h_bowl_max = np.max(X_bowl[:, 2])

        is_upper = X_bowl[:, 2] > (h_bowl_max - 0.02)
        X_bowl_upper = X_bowl[is_upper]
        circle = find_circle(X_bowl_upper[:, :2])

        self.center_list.append(circle.center)
        self.center_list = self.center_list[-self.n_window :]
        center_mean = sum(self.center_list) / len(self.center_list)

        ret_center = np.hstack([center_mean, h_bowl_max])

        pose = PoseStamped()
        pos = pose.pose.position
        pos.x, pos.y, pos.z = ret_center

        ori = pose.pose.orientation
        ori.x, ori.y, ori.z, ori.w = (0, 0, 0, 1.0)

        pose.header = msg.header
        pose.header.frame_id = target
        self.publisher.publish(pose)
        rospy.loginfo("publish")


if __name__ == "__main__":
    rospy.init_node("bowl_center_finder")
    finder = BowlCenterFinder()
    rospy.spin()
