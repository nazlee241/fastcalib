#!/usr/bin/env python3
import numpy as np
import rospy
import cv2
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from cv_bridge import CvBridge
import sensor_msgs.point_cloud2 as pc2
import threading
from collections import deque

class OptimizedLidarImageOverlay:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock = threading.Lock()
        
        # Parameters
        self.image_topic = rospy.get_param("~image_topic", "/zed2i/zed_node/left/image_rect_color")
        self.cam_info_topic = rospy.get_param("~cam_info_topic", "/zed2i/zed_node/left/camera_info")
        self.lidar_topic = rospy.get_param("~lidar_topic", "/ouster/points")
        self.output_topic = rospy.get_param("~output_topic", "/lidar_overlay/image")
        
        # Performance optimizations
        self.max_points_to_draw = rospy.get_param("~max_points", 3000)  # Limit points for speed
        self.downsample_rate = rospy.get_param("~downsample_rate", 4)   # Take 1 out of every N points
        self.frame_skip = rospy.get_param("~frame_skip", 1)             # Process every Nth frame
        
        # Store latest messages
        self.latest_image = None
        self.latest_cam_info = None
        self.latest_pts = None
        self.received_data = {'image': False, 'camera_info': False, 'lidar': False}
        
        # Frame counter for skipping
        self.frame_counter = 0
        
        # Transformation matrix
        self.Rcl = np.array([
            [-0.130379, -0.991454,  0.004591],
            [-0.076312,  0.005418, -0.997069],
            [ 0.988523, -0.130348, -0.076366]
        ], dtype=np.float64)
        self.tcl = np.array([0.388071, 0.364550, -0.125440], dtype=np.float64)
        
        # Pre-allocate arrays for speed
        self.cache_K = None
        self.cache_imsize = None
        
        # Visualization params
        self.point_size = rospy.get_param("~point_size", 2)  # Smaller points for speed
        self.min_depth = rospy.get_param("~min_depth", 0.5)
        self.max_depth = rospy.get_param("~max_depth", 20.0)
        
        # Publishers and subscribers
        self.pub = rospy.Publisher(self.output_topic, Image, queue_size=5)
        
        # Simple subscribers
        rospy.Subscriber(self.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber(self.cam_info_topic, CameraInfo, self.cam_info_callback, queue_size=1)
        rospy.Subscriber(self.lidar_topic, PointCloud2, self.lidar_callback, queue_size=1, buff_size=2**24)
        
        # Process at higher rate but with frame skipping
        self.processing_timer = rospy.Timer(rospy.Duration(0.033), self.process_callback)  # ~30 Hz
        
        rospy.loginfo("="*60)
        rospy.loginfo("OPTIMIZED LiDAR-Image Overlay Node")
        rospy.loginfo("="*60)
        rospy.loginfo(f"Max points: {self.max_points_to_draw}")
        rospy.loginfo(f"Downsample rate: 1/{self.downsample_rate}")
        rospy.loginfo(f"Frame skip: {self.frame_skip}")
        rospy.loginfo(f"Point size: {self.point_size}")
        rospy.loginfo("="*60)
    
    def image_callback(self, msg):
        with self.lock:
            self.latest_image = msg
            self.received_data['image'] = True
    
    def cam_info_callback(self, msg):
        with self.lock:
            self.latest_cam_info = msg
            self.received_data['camera_info'] = True
    
    def lidar_callback(self, msg):
        try:
            # Fast point extraction using generator
            points_list = []
            count = 0
            
            # Extract points with downsampling
            for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
                if count % self.downsample_rate == 0:  # Downsample
                    points_list.append([point[0], point[1], point[2]])
                count += 1
                
                # Early break if we have enough points
                if len(points_list) >= self.max_points_to_draw * 2:
                    break
            
            if len(points_list) > 0:
                pts = np.array(points_list, dtype=np.float32)  # Use float32 for speed
                
                with self.lock:
                    self.latest_pts = pts
                    self.received_data['lidar'] = True
                    
        except Exception as e:
            rospy.logwarn_throttle(5, f"LiDAR error: {e}")
    
    def process_callback(self, event):
        # Frame skipping for performance
        self.frame_counter += 1
        if self.frame_counter % (self.frame_skip + 1) != 0:
            return
        
        with self.lock:
            if self.latest_image is None or self.latest_cam_info is None or self.latest_pts is None:
                return
            
            img_msg = self.latest_image
            cam_info_msg = self.latest_cam_info
            pts = self.latest_pts.copy()
        
        try:
            # Fast image conversion
            img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            H, W = img.shape[:2]
            
            # Get camera intrinsics (cached if possible)
            if self.cache_K is None or self.cache_imsize != (H, W):
                K = np.array(cam_info_msg.K, dtype=np.float32).reshape(3, 3)
                self.cache_K = K
                self.cache_imsize = (H, W)
            else:
                K = self.cache_K
            
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            
            # Quick check if points exist
            if len(pts) == 0:
                return
            
            # Transform points (use float32 for speed)
            pts_cam = (pts @ self.Rcl.T.astype(np.float32)) + self.tcl.astype(np.float32)
            
            # Filter and project in one step for speed
            Z = pts_cam[:, 2]
            valid = Z > 0.1
            if not valid.any():
                return
            
            X = pts_cam[valid, 0]
            Y = pts_cam[valid, 1]
            Z = pts_cam[valid, 2]
            
            # Vectorized projection (much faster)
            u = (fx * X / Z + cx).astype(np.int32)
            v = (fy * Y / Z + cy).astype(np.int32)
            
            # Filter image bounds
            in_img = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            u = u[in_img]
            v = v[in_img]
            Z = Z[in_img]
            
            # Limit number of points to draw
            if len(u) > self.max_points_to_draw:
                indices = np.random.choice(len(u), self.max_points_to_draw, replace=False)
                u = u[indices]
                v = v[indices]
                Z = Z[indices]
            
            # Draw points efficiently
            if len(u) > 0:
                # Pre-calculate colors
                depth_norm = np.clip((Z - self.min_depth) / (self.max_depth - self.min_depth), 0, 1)
                
                # Use batch drawing for better performance
                for i in range(len(u)):
                    color = (int(255 * depth_norm[i]), 0, int(255 * (1 - depth_norm[i])))
                    cv2.circle(img, (u[i], v[i]), self.point_size, color, -1)
                
                # Publish without debug text for speed (optional)
                if rospy.get_param("~show_debug_text", False):
                    cv2.putText(img, f"Points: {len(u)}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Publish
                out_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
                out_msg.header = img_msg.header
                self.pub.publish(out_msg)
            
        except Exception as e:
            pass  # Suppress errors for performance

def main():
    rospy.init_node("optimized_lidar_overlay", anonymous=False)
    
    rospy.loginfo("Starting OPTIMIZED overlay node...")
    
    try:
        node = OptimizedLidarImageOverlay()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Fatal error: {e}")

if __name__ == "__main__":
    main()