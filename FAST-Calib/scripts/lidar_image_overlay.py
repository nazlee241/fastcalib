#!/usr/bin/env python3
import numpy as np
import rospy
import cv2

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import message_filters
import ros_numpy

def make_K(cam_info: CameraInfo):
    # K is row-major in CameraInfo.K
    K = np.array(cam_info.K, dtype=np.float64).reshape(3, 3)
    return K

def project_points(points_lidar_xyz, Rcl, tcl):
    # points: (N,3) in LiDAR frame
    # p_cam = Rcl * p_lidar + tcl
    return (points_lidar_xyz @ Rcl.T) + tcl.reshape(1, 3)

class LidarImageOverlay:
    def __init__(self):
        self.bridge = CvBridge()

        # ====== Parameters (edit if needed) ======
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_overlay/image")

        # FAST-Calib result: T_cam_lidar
        self.Rcl = np.array([
            [-0.130379, -0.991454,   0.004591],
            [-0.076312,  0.005418,  -0.997069],
            [0.988523,  -0.130348,  -0.076366]
        ], dtype=np.float64)

        self.tcl = np.array([ 0.388071,   0.364550,  -0.125440], dtype=np.float64)  # meters

        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=1)

        # Sync image + pointcloud + camera_info
        sub_img = message_filters.Subscriber(self.image_topic, Image)
        sub_pc  = message_filters.Subscriber(self.lidar_topic, PointCloud2)
        sub_ci  = message_filters.Subscriber(self.cam_info_topic, CameraInfo)

        # Approximate sync is more robust
        ats = message_filters.ApproximateTimeSynchronizer([sub_img, sub_pc, sub_ci],
                                                          queue_size=10, slop=0.05)
        ats.registerCallback(self.cb)

        rospy.loginfo("Overlay node started.")
        rospy.loginfo(f"Image: {self.image_topic}")
        rospy.loginfo(f"CameraInfo: {self.cam_info_topic}")
        rospy.loginfo(f"LiDAR: {self.lidar_topic}")
        rospy.loginfo(f"Output: {self.output_topic}")

    def cb(self, img_msg, pc_msg, cam_info_msg):
        # Convert image
        try:
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            rospy.logwarn(f"cv_bridge failed: {e}")
            return

        H, W = img.shape[:2]
        K = make_K(cam_info_msg)
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]

        # Convert pointcloud to numpy
        try:
            pc = ros_numpy.point_cloud2.pointcloud2_to_array(pc_msg)
            # Ensure fields exist
            x = pc['x'].astype(np.float64)
            y = pc['y'].astype(np.float64)
            z = pc['z'].astype(np.float64)
            pts = np.vstack((x, y, z)).T
        except Exception as e:
            rospy.logwarn(f"PointCloud2 conversion failed: {e}")
            return

        # Remove NaNs / inf
        mask = np.isfinite(pts).all(axis=1)
        pts = pts[mask]
        if pts.shape[0] == 0:
            return

        # Transform to camera frame
        pts_cam = project_points(pts, self.Rcl, self.tcl)

        # Keep only points in front of camera (Zc > 0)
        Z = pts_cam[:, 2]
        valid = Z > 0.1
        pts_cam = pts_cam[valid]
        if pts_cam.shape[0] == 0:
            return

        X = pts_cam[:, 0]
        Y = pts_cam[:, 1]
        Z = pts_cam[:, 2]

        # Project to pixels
        u = (fx * (X / Z) + cx).astype(np.int32)
        v = (fy * (Y / Z) + cy).astype(np.int32)

        in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        u = u[in_img]
        v = v[in_img]
        z_vis = Z[in_img]

        # Draw points (color by depth)
        # Simple depth coloring: near=red, far=blue-ish
        if u.size > 0:
            zmin, zmax = 0.5, 20.0
            z_norm = np.clip((z_vis - zmin) / (zmax - zmin), 0.0, 1.0)
            # Build BGR colors
            # (No fixed palette required; this is just a simple gradient)
            b = (255 * z_norm).astype(np.uint8)
            g = (255 * (1.0 - np.abs(z_norm - 0.5) * 2.0)).astype(np.uint8)
            r = (255 * (1.0 - z_norm)).astype(np.uint8)

            for ui, vi, bi, gi, ri in zip(u, v, b, g, r):
                cv2.circle(img, (int(ui), int(vi)), 1, (int(bi), int(gi), int(ri)), -1)

        # Publish overlay image
        out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        out_msg.header = img_msg.header
        self.pub.publish(out_msg)

def main():
    rospy.init_node("lidar_image_overlay", anonymous=False)
    LidarImageOverlay()
    rospy.spin()

if __name__ == "__main__":
    main()

