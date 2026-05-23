#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Features:
1) Automatically detect LiDAR point cloud type in rosbag:
   - sensor_msgs/PointCloud2 (e.g., /hesai/pandar)
   - livox_ros_driver/CustomMsg (e.g., /livox/lidar)
2) Export point cloud with intensity to PCD file (x y z intensity, ASCII)
3) Use Open3D for interactive point selection (Shift+Click)
4) Coordinates appear in TERMINAL when clicking (no hover, but printed)
5) Automatically saves selected points and bounding range to .txt file
"""

import os
import sys
import numpy as np
import rosbag
import sensor_msgs.point_cloud2 as pc2
import open3d as o3d

# ===================== General: Save PCD =====================

def save_pcd_with_intensity(points, intensities, output_path):
    """
    Save point cloud as PCD file with intensity field (ASCII format)
    points: list/ndarray of [x, y, z]
    intensities: list/ndarray of intensity
    """
    N = len(points)
    header = f"""# .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH {N}
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS {N}
DATA ascii
"""
    with open(output_path, 'w') as f:
        f.write(header)
        for (x, y, z), inten in zip(points, intensities):
            f.write(f"{x} {y} {z} {inten}\n")
    print(f"[PCD] Saved point cloud with intensity field to: {output_path}")

# ===================== Case 1: PointCloud2 =====================

def find_intensity_field(msg):
    """Auto-detect intensity field name in PointCloud2 fields"""
    candidates = ["intensity", "reflectivity", "i", "ref"]
    for field in msg.fields:
        if field.name.lower() in candidates:
            return field.name
    return None

def convert_pointcloud2_bag_to_pcd(
    bag_file,
    output_dir,
    topic_name=None,
    pcd_name="sensor_PointCloud2_inten_ascii.pcd"
):
    """
    Merge and export PointCloud2 point clouds from rosbag to a single PCD file.
    Keep original LiDAR coordinates without transformation.
    """
    print(f"[Bag] Opening rosbag: {bag_file}")
    bag = rosbag.Bag(bag_file, "r")

    # If topic not specified, auto-detect first PointCloud2 topic
    if topic_name is None:
        for topic, msg, t in bag.read_messages():
            if msg._type == "sensor_msgs/PointCloud2":
                topic_name = topic
                print(f"[Bag] Auto-detected topic: {topic_name}")
                break
    
    if topic_name is None:
        print("[ERROR] No PointCloud2 topic found!", file=sys.stderr)
        bag.close()
        return None

    # 1) Detect intensity field first
    intensity_field = None
    for topic, msg, t in bag.read_messages():
        if msg._type == "sensor_msgs/PointCloud2":
            intensity_field = find_intensity_field(msg)
            if intensity_field:
                print(f"[Bag] Detected intensity field: {intensity_field}")
            break

    if not intensity_field:
        print("[ERROR] Intensity field not found! Exiting PointCloud2 conversion.", file=sys.stderr)
        bag.close()
        return None

    # 2) Read all point clouds from specified topic
    all_points = []
    all_intensities = []

    print(f"[Bag] Starting to read PointCloud2 point clouds from topic '{topic_name}'...")

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if msg._type == "sensor_msgs/PointCloud2":
            try:
                field_names = ["x", "y", "z", intensity_field]
                for point in pc2.read_points(msg, field_names=field_names, skip_nans=True):
                    all_points.append([point[0], point[1], point[2]])
                    all_intensities.append(point[3])
            except Exception as e:
                print(f"[ERROR] Read error: {str(e)}", file=sys.stderr)
                continue

    bag.close()

    if not all_points:
        print("[ERROR] No PointCloud2 point cloud data found!", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    save_pcd_with_intensity(all_points, all_intensities, output_path)
    return output_path

# ===================== Case 2: Livox CustomMsg =====================

def parse_livox_custom_msg(msg):
    """
    Parse x, y, z, reflectivity from livox_ros_driver/CustomMsg
    Assumes msg.points is a list of CustomPoint objects with fields x, y, z, reflectivity
    """
    points = []
    intensities = []

    for pt in msg.points:
        points.append([pt.x, pt.y, pt.z])
        intensities.append(pt.reflectivity)

    return points, intensities

def convert_livox_custom_bag_to_pcd(
    bag_file,
    output_dir,
    topic_name="/livox/lidar",
    pcd_name="livox_CustomMsg_inten_ascii.pcd"
):
    """
    Merge and export Livox CustomMsg point clouds from rosbag to a single PCD file.
    Keep original LiDAR coordinates without transformation.
    """
    print(f"[Bag] Opening rosbag: {bag_file}")
    bag = rosbag.Bag(bag_file, "r")

    all_points = []
    all_intensities = []

    print(f"[Bag] Starting to read CustomMsg point clouds from topic '{topic_name}'...")

    for topic, msg, t in bag.read_messages(topics=[topic_name]):
        if msg._type == "livox_ros_driver/CustomMsg":
            pts, intens = parse_livox_custom_msg(msg)
            all_points.extend(pts)
            all_intensities.extend(intens)

    bag.close()

    if not all_points:
        print("[ERROR] No Livox CustomMsg point cloud data found!", file=sys.stderr)
        return None

    output_path = os.path.join(output_dir, pcd_name)
    save_pcd_with_intensity(all_points, all_intensities, output_path)
    return output_path

# ===================== Auto-detection: Which message type is in this bag? =====================

def detect_lidar_msg_type(bag_file):
    """
    Scan the bag to detect PointCloud2 or Livox CustomMsg.
    Returns:
        "PointCloud2", "CustomMsg", or None
    If both exist, prioritize PointCloud2 and print a message.
    """
    has_pc2 = False
    has_livox = False

    print(f"[Detect] Scanning bag: {bag_file}")
    bag = rosbag.Bag(bag_file, "r")

    for topic, msg, t in bag.read_messages():
        if msg._type == "sensor_msgs/PointCloud2":
            has_pc2 = True
        elif msg._type == "livox_ros_driver/CustomMsg":
            has_livox = True

        if has_pc2 and has_livox:
            break

    bag.close()

    if has_pc2 and has_livox:
        print("[Detect] Both PointCloud2 and Livox CustomMsg detected, defaulting to PointCloud2.")
        return "PointCloud2"
    elif has_pc2:
        print("[Detect] PointCloud2 point cloud detected.")
        return "PointCloud2"
    elif has_livox:
        print("[Detect] Livox CustomMsg point cloud detected.")
        return "CustomMsg"
    else:
        print("[Detect] No PointCloud2 or Livox CustomMsg point cloud detected.")
        return None

# ===================== Point Picker with Terminal Feedback =====================

def pick_points_with_terminal_feedback(pcd, window_title="Select Points"):
    """
    Click points and see coordinates printed in terminal (not hover)
    Returns selected point indices and coordinates
    """
    print("\n" + "="*70)
    print("POINT PICKER INSTRUCTIONS:")
    print("  - Hold Shift + left click to select points")
    print("  - Selected point coordinates will appear in THIS TERMINAL")
    print("  - Press 'Q' when finished selecting")
    print("  - Need at least 4 points for calibration board")
    print("="*70)
    
    # Create visualizer with vertex selection
    vis = o3d.visualization.VisualizerWithVertexSelection()
    vis.create_window(window_name=window_title, width=1280, height=720)
    vis.add_geometry(pcd)
    
    # Set rendering options for distant LiDAR
    opt = vis.get_render_option()
    opt.point_size = 2.0  # Small points to avoid occlusion
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.show_coordinate_frame = True
    
    # Run visualization - coordinates will print to terminal as you click
    vis.run()
    vis.destroy_window()
    
    # Get all selected points
    picked = vis.get_picked_points()
    return picked

def select_and_save_points_terminal(pcd_folder, target_pcd_name):
    """
    Interactive point selection with terminal feedback
    Coordinates appear in terminal when Shift+clicking
    """
    pcd_path = os.path.join(pcd_folder, target_pcd_name)
    if not os.path.isfile(pcd_path):
        print(f"[ERROR] Specified PCD file does not exist: {pcd_path}", file=sys.stderr)
        return False

    # Read point cloud
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"[ERROR] {target_pcd_name} has no point cloud data, skipping", file=sys.stderr)
        return False

    all_points = np.asarray(pcd.points)
    
    # Print point cloud statistics
    print("\n" + "="*70)
    print(f"PROCESSING: {target_pcd_name}")
    print("="*70)
    print(f"Total points in cloud: {len(all_points):,}")
    print(f"\nPoint cloud boundaries:")
    print(f"  X range: [{all_points[:,0].min():.3f}, {all_points[:,0].max():.3f}]")
    print(f"  Y range: [{all_points[:,1].min():.3f}, {all_points[:,1].max():.3f}]")
    print(f"  Z range: [{all_points[:,2].min():.3f}, {all_points[:,2].max():.3f}]")
    
    # Calculate center of point cloud for reference
    center = all_points.mean(axis=0)
    print(f"\nApproximate center of point cloud: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f})")
    
    # Launch point picker
    print("\nLaunching point picker...")
    print("NOTE: Coordinates will appear in THIS terminal window as you click!")
    print("Watch the terminal for coordinate output.\n")
    
    selected_indices_obj = pick_points_with_terminal_feedback(
        pcd, 
        window_title=f"Shift+Click to select points - {target_pcd_name}"
    )
    
    if not selected_indices_obj:
        print("[ERROR] No points selected!", file=sys.stderr)
        return False
    
    # Extract indices and coordinates
    selected_indices = [p.index for p in selected_indices_obj]
    selected_points = np.array([p.coord for p in selected_indices_obj])
    
    print("\n" + "="*70)
    print(f"Total points selected: {len(selected_indices)}")
    
    if len(selected_indices) < 4:
        print(f"[ERROR] Only selected {len(selected_indices)} points, need at least 4!", file=sys.stderr)
        return False
    
    # Take only first 4 points (or let user choose which 4 to use)
    if len(selected_indices) > 4:
        print(f"\nYou selected {len(selected_indices)} points. Using first 4 for calibration.")
        print("(You can re-run if you want different points)")
        selected_indices = selected_indices[:4]
        selected_points = selected_points[:4]
    
    # Display all selected points
    print("\nSELECTED POINTS (will be used for calibration):")
    for i, (idx, point) in enumerate(zip(selected_indices, selected_points)):
        print(f"  Point {i+1}: Index={idx}, Coord=({point[0]:.6f}, {point[1]:.6f}, {point[2]:.6f})")
    
    # Ask for confirmation
    print("\n" + "="*70)
    confirm = input("Use these 4 points for calibration? (y/n): ").lower()
    if confirm != 'y':
        print("Selection cancelled. No file saved.")
        return False
    
    # Calculate min and max on each axis from the 4 points
    mins = selected_points.min(axis=0)
    maxs = selected_points.max(axis=0)
    
    # Expand by 0.2m as defined
    x_min = mins[0] - 0.2
    x_max = maxs[0] + 0.2
    y_min = mins[1] - 0.2
    y_max = maxs[1] + 0.2
    z_min = mins[2] - 0.2
    z_max = maxs[2] + 0.2
    
    # Generate save filename (same as PCD file but with .txt extension)
    base_name = os.path.splitext(target_pcd_name)[0]
    save_file = os.path.join(pcd_folder, f"{base_name}.txt")
    
    # Save to file
    with open(save_file, 'w') as f:
        f.write("# 4 selected points (x y z)\n")
        f.write("# Format: x y z\n")
        f.write("# Point index (in original cloud) and coordinates:\n")
        for idx, point in zip(selected_indices, selected_points):
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}  # index: {idx}\n")
        
        f.write("\n# Range values with 0.2m expansion:\n")
        f.write(f"x_min: {x_min:.6f}\n")
        f.write(f"x_max: {x_max:.6f}\n")
        f.write(f"y_min: {y_min:.6f}\n")
        f.write(f"y_max: {y_max:.6f}\n")
        f.write(f"z_min: {z_min:.6f}\n")
        f.write(f"z_max: {z_max:.6f}\n")
    
    print("\n" + "="*70)
    print("SAVE RESULTS")
    print("="*70)
    print(f"  Selected points saved to: {save_file}")
    print(f"\n  Calculated bounding box (with 0.2m expansion):")
    print(f"    X: [{x_min:.3f}, {x_max:.3f}]")
    print(f"    Y: [{y_min:.3f}, {y_max:.3f}]")
    print(f"    Z: [{z_min:.3f}, {z_max:.3f}]")
    print("\n  Point cloud processing completed successfully!")
    print("="*70)
    
    return True

# ===================== Main =====================

if __name__ == "__main__":
    # 1) Parse command line arguments: bag path & output directory
    if len(sys.argv) > 1:
        bag_file = sys.argv[1]
    else:
        bag_file = os.path.join(os.getcwd(), "all_2025-11-17-18-22-27.bag")
        print(f"No bag file specified, using default: {bag_file}")

    if len(sys.argv) > 2:
        output_dir = sys.argv[2]
    else:
        output_dir = os.getcwd()
        print(f"No output directory specified, using current directory: {output_dir}")

    if not os.path.isfile(bag_file):
        print(f"[ERROR] Bag file '{bag_file}' does not exist", file=sys.stderr)
        sys.exit(1)

    if not os.path.isdir(output_dir):
        print(f"[INFO] Creating output directory: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)

    # 2) Auto-detect point cloud type in bag
    msg_type = detect_lidar_msg_type(bag_file)
    if msg_type is None:
        print("[ERROR] No supported LiDAR message type detected, exiting.", file=sys.stderr)
        sys.exit(1)

    # 3) Convert to PCD based on type
    if msg_type == "PointCloud2":
        pcd_path = convert_pointcloud2_bag_to_pcd(
            bag_file=bag_file,
            output_dir=output_dir,
            topic_name=None,  # Auto-detect
            pcd_name="pointcloud_intensity.pcd"
        )
    else:  # "CustomMsg"
        pcd_path = convert_livox_custom_bag_to_pcd(
            bag_file=bag_file,
            output_dir=output_dir,
            topic_name=None,  # Will use default "/livox/lidar"
            pcd_name="pointcloud_intensity.pcd"
        )

    if pcd_path is None:
        print("[ERROR] PCD generation failed, exiting.", file=sys.stderr)
        sys.exit(1)

    # 4) Interactive point selection with terminal feedback
    success = select_and_save_points_terminal(
        pcd_folder=output_dir,
        target_pcd_name=os.path.basename(pcd_path)
    )
    
    if success:
        print("\nScript completed successfully!")
    else:
        print("\nScript completed with errors!")
        sys.exit(1)