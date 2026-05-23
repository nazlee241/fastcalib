# After launching the zed_wrapper, you can echo the camera info topic
rostopic echo /zed2i/zed_node/left/camera_info -n 1 | grep -A 9 "K:"

# Look for the K: matrix, which is the 3x3 intrinsic matrix. It will look like this:
K: [fx, 0.0, cx,
    0.0, fy, cy,
    0.0, 0.0, 1.0]

# the usage of distance_filter_tool.py
the script is for manually assign the circle of the checkerboard, then it will gave us the value to put in the qr_params.yaml
STEP to run
1) $ python3 distance_filter_tool.py /home/amt4/Documents/master_nazlee/checkerboard_calib_bag/sample_2/right.bag /home/amt4/fastcalib_ws/src/FAST-Calib/scripts/sample_2/right
2) shift + click at the center point circle
3) press "q" once done

