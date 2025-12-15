import cv2
import os
import numpy as np
import glob 

THRESHOLD = 0.80                    # Extent threshold for rule-based perspective correction of side-view image

# Function to extract 20 video frames
def extract_20_frames(video_path, output_folder):
    """
    Extract 20 frames (first and last 5, and frame 40 to 49) from the input video and output to a destination folder.

    Args:
        video_path: Path of input videos.
        output_folder: Path of output folder.

    Returns:
        None
    """
    TARGET_FRAMES_PER_END = 5           # N for first N and last N frames
    START_FRAME_INDEX = 40              # Start of the fixed middle range
    END_FRAME_INDEX = 49                # End of the fixed middle range
    WHITE_BACKGROUND_THRESHOLD = 240    # Average BGR pixel value threshold for the corner regions
    CORNER_SAMPLE_SIZE = 50             # The side length in pixels of the square region to sample in each corner
    BORDER_COLOR = (255, 255, 255) 
    PADDING_THICKNESS = 20 

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file at path: {video_path}. Check file existence and OpenCV/FFmpeg codecs.")

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Processing video: {os.path.basename(video_path)} | Total Frames: {total_frames}")

    # --- 1. Calculate Target Frame Indices ---
    if total_frames > TARGET_FRAMES_PER_END:
        first_frames = list(range(TARGET_FRAMES_PER_END))
    else:
        first_frames = list(range(total_frames)) 
    
    if total_frames > END_FRAME_INDEX:
        fixed_frames = list(range(START_FRAME_INDEX, END_FRAME_INDEX + 1))
    elif total_frames > START_FRAME_INDEX:
        fixed_frames = list(range(START_FRAME_INDEX, total_frames))
        print(f"  Warning: Video ends before {END_FRAME_INDEX}. Fixed set truncated at frame {total_frames - 1}.")
    else:
        fixed_frames = []
        print(f"  Warning: Video is too short for the fixed range starting at {START_FRAME_INDEX}. Fixed set skipped.")
        
    last_frames = []
    if total_frames > 0:
        last_frames_start_index = max(0, total_frames - TARGET_FRAMES_PER_END)
        last_frames = list(range(last_frames_start_index, total_frames))
    
    combined_indices = sorted(list(set(
        first_frames + fixed_frames + last_frames
    )))
    
    frame_indices = [idx for idx in combined_indices if idx < total_frames]

    print(f"  Total frames to attempt extraction: {len(frame_indices)}")

    # --- 2. Define Corner Sample Boxes (BL and BR) ---
    corner_boxes = [
        (0, frame_height - CORNER_SAMPLE_SIZE, CORNER_SAMPLE_SIZE, frame_height),
        (frame_width - CORNER_SAMPLE_SIZE, frame_height - CORNER_SAMPLE_SIZE, frame_width, frame_height)
    ]
    
    # --- 3. Extract and Process Frames ---
    frames_saved_count = 0
    
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = cap.read()
        if not ret:
            print(f"  Error: Failed to read frame {frame_index}. Skipping.")
            continue

        corner_intensities = []
        for x_start, y_start, x_end, y_end in corner_boxes:
            corner_region = frame[y_start:y_end, x_start:x_end]
            corner_intensities.append(np.mean(corner_region))

        overall_corner_average = np.mean(corner_intensities) if corner_intensities else 0 

        if overall_corner_average >= WHITE_BACKGROUND_THRESHOLD:
            cv2.rectangle(
                img=frame,
                pt1=(0, 0),
                pt2=(frame_width - 1, frame_height - 1),
                color=BORDER_COLOR,
                thickness=PADDING_THICKNESS
            )

            frame_filename = os.path.join(output_folder, f"frame_{frame_index:04d}.jpg")
            cv2.imwrite(frame_filename, frame)
            frames_saved_count += 1

    cap.release()
    print(f" Completed: {frames_saved_count} frames saved.")


# Helper function to extract binary mask
def morphological_gradient_detection(image):
    """
    Extract binary masks from all 180 frames from the input video and output to a destination folder.

    Args:
        image: The input image.

    Returns:
        image: The processed image.
    """

    # 1. Blur slightly to reduce camera noise, but keep structure
    blurred = cv2.GaussianBlur(image, (5, 5), 0)

    # 2. Compute Morphological Gradient
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_gradient = cv2.morphologyEx(blurred, cv2.MORPH_GRADIENT, kernel)

    # 3. Binarize the Gradient
    _, binary_edges = cv2.threshold(morph_gradient, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)

    # 4. Fill the Gaps (Morphological Closing)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    closed_mask = cv2.morphologyEx(binary_edges, cv2.MORPH_CLOSE, close_kernel)

    # 5. Find the Largest Contour to remove noise
    contours, _ = cv2.findContours(closed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    final_mask = np.zeros_like(image)
    
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fill the largest contour (The Gem)
        cv2.drawContours(final_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)

    return final_mask


# Helper function to extract binary mask
def apply_convex_hull(image):
    """
    Robustly finds the gemstone shape using Convex Hull.
    Returns a black mask if no object is found, preventing 'all white' errors.

    Args:
        image: The input image.

    Returns:
        image: The processed image.

    """
    # 1. Ensure binary
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    h, w = binary.shape[:2]
    image_area = h * w
    final_mask = np.zeros((h, w), dtype=np.uint8)

    # 2. AUTO-DETECT BACKGROUND COLOR
    corners = [binary[0,0], binary[0, w-1], binary[h-1, 0], binary[h-1, w-1]]
    avg_corner = sum(corners) / 4

    if avg_corner > 127:
        binary = cv2.bitwise_not(binary)

    # 3. Find Contours (Use RETR_EXTERNAL to ignore internal holes/noise)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("   -> No contours found. Returning empty black mask.")
        return final_mask

    # 4. Filter Contours
    valid_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # Ignore tiny noise
        if area < 100: 
            continue
        
        # IGNORE FRAME: Only ignore if it is literally the border of the image
        # (Increased threshold from 0.95 to 0.99 to allow large gems)
        if area > 0.99 * image_area:
            continue
            
        valid_contours.append(cnt)

    if not valid_contours:
        if contours:
            largest_raw = max(contours, key=cv2.contourArea)
            x, y, cw, ch = cv2.boundingRect(largest_raw)
            if cw < w or ch < h: 
                valid_contours.append(largest_raw)
            else:
                print("   -> Only found image frame/border. Returning empty mask.")
                return final_mask
        else:
            return final_mask

    # 5. Find the largest remaining contour (The Gemstone)
    largest_contour = max(valid_contours, key=cv2.contourArea)

    # 6. Apply Convex Hull
    hull = cv2.convexHull(largest_contour)

    # 7. Draw on black canvas
    cv2.drawContours(final_mask, [hull], -1, 255, thickness=cv2.FILLED)

    return final_mask


# Function to extract binary mask
def extract_binary_masks(input_folder, output_folder):
    """
    Extract the binary mask from the extracted video frames using the following steps:
    Step 1: Extract binary mask using morphological gradient detection.
    Step 2: Fill holes in the extracted binary masks using convex hull.

    Args:
        input_folder: The input folder containing 20 video frames.
        output_folder: The output folder to save the binary masks.

    Returns:
        None
    """

    image_extensions = ('.jpg')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' not found.")
    else:
        image_files = os.listdir(input_folder)
        image_files = [i for i in image_files if i.lower().endswith('.jpg')]
        image_files.sort()

        if not image_files:
            print(f"No images found in the directory: {input_folder}")
        else:
            for filename in image_files:
                if filename.lower().endswith(image_extensions):
                    input_path = os.path.join(input_folder, filename)
                    image = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
                    
                    if image is None:
                        print(f"Warning: Could not read image {filename}. Skipping.")
                        continue
                    
                    # 1. Apply morphological gradient detection
                    processed_mask = morphological_gradient_detection(image)

                    # 2. Apply convex hull
                    closed_mask = apply_convex_hull(processed_mask)

                    mask_filename = os.path.splitext(filename)[0] + ".png"
                    output_path = os.path.join(output_folder, mask_filename)
                    cv2.imwrite(output_path, closed_mask)
                    print(f"Processed '{filename}' and saved mask to '{mask_filename}'")

            print("\nAll images processed successfully.")


# Function to calculate maximum length of binary mask to extract top-view image
def calculate_maximum_length(image):
    """
    Finds the maximum width (length) of the largest WHITE object (contour)

    Args:
        image: The input image.

    Returns:
        tuple: A tuple containing the maximum length, start and end point of the maximum length

    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No white object contours found in the mask.")
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), 0.0

    # 1. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 2. Extract all coordinates (x, y) from the largest contour
    all_points = largest_contour.reshape(-1, 2)
    
    # 3. Find the minimum and maximum X values
    extreme_left_x = np.min(all_points[:, 0])
    extreme_right_x = np.max(all_points[:, 0])
    maximum_length = extreme_right_x - extreme_left_x
    
    # 4. Find the actual (x, y) boundary points (Needed to calculate a reference Y-coordinate)
    left_points = all_points[all_points[:, 0] == extreme_left_x]
    right_points = all_points[all_points[:, 0] == extreme_right_x]
    start_point_hypotenuse = tuple(left_points[-1])
    end_point_hypotenuse = tuple(right_points[0])

    return float(maximum_length), start_point_hypotenuse, end_point_hypotenuse


# Function to extract index of top-view image filename
def extract_top_view_image(input_folder):
    """
    Finds the top-view image filename based on maximum length from binary masks.

    Args:
        input folder: The input folder containing all the 20 processed binary masks.

    Returns:
        string: A string containing the path of the top-view image filename.
    """
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    SAMPLE_SIZE = 5 
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    image_paths.sort()
    image_paths = [i for i in image_paths if i.split(".")[-1] != "jpg"]
    
    if not image_paths:
        print(f"Error: No images found in the directory: {input_folder}")
        exit()
    
    all_lengths = []
    processed_count = 0
    
    for input_path in image_paths:
        binary_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is not None:
            length, _, _ = calculate_maximum_length(binary_mask)
            all_lengths.append(length)
            processed_count += 1

    total_files = len(image_paths)
    first_sample_paths = image_paths[:SAMPLE_SIZE]
    last_sample_paths = image_paths[max(0, total_files - SAMPLE_SIZE):]
    paths_to_analyze = first_sample_paths + last_sample_paths
    
    longest_length = -1.0
    longest_mask_path = None 
    
    for path in paths_to_analyze:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        length, _, _ = calculate_maximum_length(mask) 
        if length > longest_length:
            longest_length = length
            longest_mask_path = path

    if longest_mask_path and longest_length > 0:
        global_filename = os.path.basename(longest_mask_path)
        global_filename = global_filename.replace(".jpg", ".png")
        print(f"global_filename: {global_filename}")
        
        return global_filename
    else:
        print(f"Could not find a valid frame with length data in the first {SAMPLE_SIZE} frames.")


# Function to extract index of side-view image filename
def extract_side_image(global_filename):
    """
    Finds the side-view image filename based on the top-view image filename.

    Args:
        global filename: The filename of the top-view image.

    Returns:
        string: A string containing the path of the side-view image filename.
    """
    # Extract top-view image frame index
    top_view_image = global_filename
    top_view_image_index = top_view_image.split("_")[1]
    top_view_image_index = int(top_view_image_index.split(".")[0])

    # Extract front view image frame index
    side_view_image_index = top_view_image_index + 45
    side_view_image = "frame_00" + str(side_view_image_index) + ".png"
    print(f"side view image filename: {side_view_image}") 

    return side_view_image


# Function to calculate maximum length and width from top-view image
def calculate_max_dimensions_combined(image):
    """
    Calculates length and width of top-view image using maximum length and width.

    Args:
        image: The input top-view image to process the maximum length and width.

    Returns:
        tuple: A tuple containing the length and width the top-view image.
    """

    # Check the number of dimensions/channels
    if len(image.shape) == 3:
        # If it has 3 dimensions (channels), assume it's color (BGR) and convert to grayscale.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif len(image.shape) == 2:
        # If it has 2 dimensions, it is already grayscale (1 channel).
        gray = image.copy() 
    else:
        # Handle unexpected number of dimensions (e.g., 4 channels for BGRA)
        raise ValueError(f"Unexpected number of image dimensions: {len(image.shape)}")

    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximum_width = 0.0
    maximum_length = 0.0
    conversion_factor = 0.00274 * 2

    if not contours:
        print("No black object contours found in the mask.")
        return maximum_width, maximum_length

    largest_contour = max(contours, key=cv2.contourArea)
    all_points = largest_contour.reshape(-1, 2)
    
    # Y-Extremes (Width)
    extreme_top_y = np.min(all_points[:, 1])
    extreme_bottom_y = np.max(all_points[:, 1])
    maximum_width = extreme_bottom_y - extreme_top_y 

    # X-Extremes (Length)
    extreme_left_x = np.min(all_points[:, 0])
    extreme_right_x = np.max(all_points[:, 0])
    maximum_length = extreme_right_x - extreme_left_x

    length = 0
    width = 0

    if maximum_length > maximum_width:
        length = maximum_length * conversion_factor
        width = maximum_width * conversion_factor
    else:
        length = maximum_width * conversion_factor
        width = maximum_length * conversion_factor

    length = round(length, 2)
    width = round(width, 2)

    print(f"Length: {length} mm")
    print(f"Width: {width} mm")

    return float(length), float(width)


# Helper function to apply perspective correction for rectangular shapes
def extract_min_bounding_box(global_filename):
    """
    Finds the minimum area bounding box for the main object in an image.

    Args:
        global filename: The filename of the top-view image which is rectangular.

    Returns:
        tuple: A tuple containing the image, width, height and angle of rotation or 
               (None, None, None, None) if no image was found.
    """
    image_path = "/temp/" + global_filename 
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return None, None, None, None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        min_area_rect = cv2.minAreaRect(largest_contour)
        box_points = cv2.boxPoints(min_area_rect)
        box_points = np.intp(box_points)
        
        (x, y), (width, height), angle = min_area_rect
        print(f"Processing {os.path.basename(image_path)}:")
        print(f"   Minimum Bounding Box: Size: ({width:.2f} x {height:.2f}), Angle: {angle:.2f}°")

        return image, width, height, angle
    else:
        print(f"No contours found in {os.path.basename(image_path)}")
        return None, None, None, None


# Helper function to apply perspective correction for rectangular shapes
def correct_perspective_min_bounding_box_top_view(image, angle_deg, center=None):
    """
    Applies rotation based on the angle outputted from extract_min_bounding_box.
    
    Args:
        image: The top-view image which is rectangular.
        angle_deg: The angle of rotation from the minimum area bounding box.

    Returns:
        tuple: A tuple containing the rotated image and angle of rotation.
    """
    if image is None:
        return None

    # 1. Calculate the rotation angle (Using the logic provided by the user)
    rotation_angle = 0
    if angle_deg > 10:
        rotation_angle = angle_deg - 90
    else:
        rotation_angle = angle_deg

    # 2. Define the center of rotation
    if center is None:
        (h, w) = image.shape[:2] if len(image.shape) == 3 else image.shape
        center = (w // 2, h // 2)

    # 3. Create and apply the rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    (h, w) = image.shape[:2] if len(image.shape) == 3 else image.shape 
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image, rotation_angle


# Helper function to apply iterative horizontal perspective correction
def correct_perspective_horizontal(image, A, B):
    """
    Calculates the rotation angle needed to make the line connecting A and B horizontal,
    and then applies that rotation to the image.

    Args:
        image (np.array): The input image.
        A (tuple): (x, y) coordinates of point A.
        B (tuple): (x, y) coordinates of point B.

    Returns:
        tuple: The tuple containing the corrected (rotated) image and angle of rotation.
    """
    # Ensure image is not None/empty
    if image is None:
        print("Error: Input image is None.")
        return None

    # 1. Calculate the angle of the line connecting A and B
    angle_rad = np.arctan2(B[1] - A[1], B[0] - A[0])
    angle_deg = np.degrees(angle_rad)
    print(f"Calculated Angle (degrees): {angle_deg:.2f}")
    rotation_angle = angle_deg

    # 2. Define the center of rotation
    if len(image.shape) == 3:
        (h, w) = image.shape[:2]
    else:
        (h, w) = image.shape
    
    center = (w // 2, h // 2)

    # 3. Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # 4. Apply the rotation to the image
    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return rotated_image, rotation_angle


# Function to apply iterative horizontal perspective correction
def horizontal_iterative_correction(global_filename):
    """" 
    Apply iterative horizontal perspective correction till convergence.
    
    Args:
        global filename: The filename of the input top- or side-view image.

    Returns:
        image (np.array): The rotated (corrected) image.
    """
    image_path = "/temp/" + global_filename 
    binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    MAX_ITERATIONS = 10
    CONVERGENCE_TOLERANCE = 0.5 # Degrees
    DIVERGENCE_THRESHOLD = 0.1 # Stop if correction angle increases by more than this (e.g., 0.1 deg)

    # Calculate initial angle and points BEFORE loop starts
    _, A, B = calculate_maximum_length(binary_mask)
    
    # Calculate the exact angle required for the initial rotation
    initial_correction_angle = np.degrees(np.arctan2(B[1] - A[1], B[0] - A[0]))
    
    rotation_angle = initial_correction_angle # Start with the initial angle
    previous_required_angle = initial_correction_angle * 10 # Initialize high to pass the first check
    iteration = 0
    
    # Variables to store the last stable (non-divergent) result
    best_stable_mask = binary_mask.copy()
    best_stable_angle = 0.0
    print(f"Starting Iterative Horizontal Perspective Correction (Tolerance: {CONVERGENCE_TOLERANCE} deg)...")
    
    # Get the initial drawn mask for plotting "Before Convergence"
    _, A, B = calculate_maximum_length(binary_mask)
    
    while iteration < MAX_ITERATIONS:
        if abs(rotation_angle) <= CONVERGENCE_TOLERANCE and iteration > 0:
            print(f"\n--- SUCCESS: CONVERGED ---")
            print(f"Final Angle: {rotation_angle:.4f}° at Iteration {iteration}.")
            break
            
        # A. Find the new extreme points (A and B) on the current mask
        _, A, B = calculate_maximum_length(binary_mask)

        # B. Calculate the rotation needed and apply it
        binary_mask, rotation_angle = correct_perspective_horizontal(binary_mask, A, B)
        
        # --- DIVERGENCE CHECK (After 1st iteration) ---
        if iteration > 0:
            if abs(rotation_angle) > abs(previous_required_angle) + DIVERGENCE_THRESHOLD:
                print(f"\n--- WARNING: DIVERGENCE DETECTED ---")
                print(f"Current angle ({rotation_angle:.4f}°) is > Previous angle ({previous_required_angle:.4f}°).")
                print(f"Stopping and reverting to last stable result (Iteration {iteration - 1}).")
                
                # Revert mask and angle to the last stable state
                binary_mask = best_stable_mask
                rotation_angle = best_stable_angle
                break

        # Store the current mask and angle as the 'best stable' result
        best_stable_mask = binary_mask.copy()
        best_stable_angle = rotation_angle
        previous_required_angle = rotation_angle 
        print(f"Iteration {iteration + 1}: Required Correction: {rotation_angle:.4f} deg")

        if binary_mask is None:
            print("Error during rotation. Stopping loop.")
            break
            
        iteration += 1
    
    else: 
        print(f"\n--- WARNING: MAX ITERATIONS REACHED ---")
        print(f"Final Angle: {rotation_angle:.4f} deg (Reverting to best stable result if not converged).")
        binary_mask = best_stable_mask
        rotation_angle = best_stable_angle

    return binary_mask


# Helper function to apply iterative vertical perspective correction
def correct_perspective_vertical(image, A, B, center=None):
    """
    Calculates and applies the rotation needed to make the line connecting A and B vertical.
    
    Returns: rotated_image, required_rotation_angle
    """
    if image is None:
        return None, 0.0
    
    # 1. Calculate the angle of the line A->B (Theta)
    angle_rad = np.arctan2(B[1] - A[1], B[0] - A[0])
    angle_deg = np.degrees(angle_rad)
    
    # Target angle for the line is 90 degrees (downwards on the image plane)
    rotation_angle = angle_deg - 90.0

    # 2. Define the center of rotation
    if center is None:
        (h, w) = image.shape[:2] if len(image.shape) > 1 else image.shape
        center = (w // 2, h // 2)

    # 3. Create and apply the rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    (h, w) = image.shape[:2] if len(image.shape) > 1 else image.shape
    rotated = cv2.warpAffine(
        image, 
        M, 
        (w, h), 
        flags=cv2.INTER_CUBIC, 
        borderMode=cv2.BORDER_CONSTANT, 
        borderValue=0
    )

    return rotated, rotation_angle


# Function to apply iterative vertical perspective correction
def vertical_iterative_correction(original_mask):
    """
    Applies vertical iterative perspective correction until convergence or divergence.
    Returns: final_stable_mask, final_angle_used
    """
    current_mask = original_mask.copy()
    
    (h, w) = original_mask.shape[:2]
    center = (w // 2, h // 2)

    # 2. ITERATIVE CORRECTION PARAMETERS
    tolerance = 0.5 # Degrees
    max_iterations = 20
    rotation_angle = tolerance + 1.0 # Initialize above tolerance to start the loop
    iteration = 0
    
    print(f"Starting Iterative Vertical Perspective Correction (Tolerance: <{tolerance} deg)...")
    
    while abs(rotation_angle) > tolerance and iteration < max_iterations:
        # A. Find the new extreme points (A and B) on the current mask
        _, _, A, B = calculate_maximum_height(current_mask)
        
        # B. Calculate the rotation needed and apply it
        current_mask, rotation_angle = correct_perspective_vertical(current_mask, A, B, center)
        
        print(f"Iteration {iteration + 1}: Required Rotation: {rotation_angle:.4f} deg")

        if current_mask is None:
            print("Error during rotation. Stopping loop.")
            break
            
        iteration += 1

    return current_mask, rotation_angle


# Function to determine criteria for rule-based perspective for side-view image
def calculate_extent(global_filename):
    image_path = "/temp/" + global_filename 
    image = cv2.imread(image_path)

    if len(image.shape) == 3:
        # Convert BGR (3-channel) to Grayscale (1-channel)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    else:
        # Assume it's already Grayscale (1-channel)
        gray = image

    # Find contours in the grayscale image
    # RETR_EXTERNAL retrieves only the outer contours
    # CHAIN_APPROX_SIMPLE compresses horizontal, vertical, and diagonal segments
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if not contours:
        print("No contours found.")
        return None

    # Select the largest contour (assumes the shape is the main object)
    # This converts the list of contours into a single contour array 'c'
    c = max(contours, key=cv2.contourArea)

    # Get Min Area Rect (Rotated Bounding Box)
    rect = cv2.minAreaRect(c)
    (_, _), (width, height), _ = rect
    box_area = width * height
    shape_area = cv2.contourArea(c)

    # Extent (How much the shape fills the bounding box)
    extent = shape_area / box_area if box_area > 0 else 0

    return extent


# Function to apply perspective correction using minimum bounding box for side-view image
def correct_perspective_min_bounding_box_side_view(image, width_rect, height_rect, angle_deg, center=None):
    """
    Calculates the rotation angle needed to align the *longest* side (the length) 
    horizontally, and applies that rotation to the image.
    
    Args:
        image (np.ndarray): The image to be rotated (can be BGR or single-channel).
        width_rect (float): The width of the minAreaRect.
        height_rect (float): The height of the minAreaRect.
        angle_deg (float): The rotation angle of the minAreaRect (-90 to 0).
        center (tuple): Center of rotation (default is image center).

    Returns: 
        rotated_image (np.ndarray): The perspective-corrected image.
    """
    if image is None:
        return None
    
    if width_rect >= height_rect:
        rotation_angle = angle_deg
    else:
        rotation_angle = angle_deg + 90
    print(f"   Perspective Correction Angle Applied: {rotation_angle:.2f} degrees")

    # Define the center of rotation and dimensions
    (h, w) = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)

    # Create the rotation matrix
    M = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)

    # Apply the rotation
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return rotated

# Function to calculate maximum height from side-view image
def calculate_maximum_height(image):
    """
    Finds the maximum height of the largest WHITE object (contour).

     Args:
        image (np.array): The input side-view image.

    Returns:
        float: The maximum height of the side-view image.
    """
    maximum_height = 0.0
    conversion_factor = 0.00274 * 2
    
    if len(image.shape) == 3:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contour_image = binary_mask
    else:
        contour_image = image

    contours, _ = cv2.findContours(contour_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No black object contours found in the mask.")
        return maximum_height

    # 1. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 2. Extract all coordinates (x, y) from the largest contour
    all_points = largest_contour.reshape(-1, 2)
    
    # 3. Find the minimum and maximum Y values (Height is difference between min/max Y)
    extreme_top_y = np.min(all_points[:, 1])    
    extreme_bottom_y = np.max(all_points[:, 1]) 
    
    maximum_height = extreme_bottom_y - extreme_top_y
    height = maximum_height * conversion_factor
    height = round(height, 2)
    print(f"Thickness: {height} mm")

    return float(height)

# Function to process dimensions
def process_dimensions(video_path, input_folder, output_folder, shape):
    """
    Finds the length, width and thickness of the input video.

     Args:
        video: The input video.
        input_folder: The input folder containing the extracted 180 video frames.
        output_folder: The output folder to save the binary masks.
        
    Returns:
        tuple: The tuple containing the length, width and thickness of the input video.
    """
    
    # 1. Extract 20 video frames
    extract_20_frames(video_path, output_folder)

    # 2. Extract binary mask
    extract_binary_masks(input_folder, output_folder)

    # 3. Extract index of top-view image filename
    top_global_filename = extract_top_view_image(input_folder)
    
    # 4. Apply minimum bounding box perspective correction for rectangular shapes
    if shape == "Radiant" or shape == "Emerald" or shape == "Cushion" or shape == "Rectangular Cushion" or shape == "Rectangular" or shape == "Square Cushion" or shape == "Square":
        image, _, _, angle = extract_min_bounding_box(top_global_filename)
        corrected_image, _ = correct_perspective_min_bounding_box_top_view(image, angle, center=None)

    # 5. Apply iterative horizontal perspective correcion for round shape
    elif shape == "Oval" or shape == "Round" or shape == "Pear" or shape == "Other" or shape == "Marquise":
        corrected_image = horizontal_iterative_correction(top_global_filename)

    # 6. Apply iterative vertical perspective correction for heart or triangular shapes
    else:
        corrected_image = vertical_iterative_correction(top_global_filename)

    # 7. Calculate maximum length and width from top-view image     
    length, width = calculate_max_dimensions_combined(corrected_image)

    # 8. Extract index of side-view image filename
    side_global_filename = extract_side_image(top_global_filename)

    # 9. Calculate extent of side-view image for rule-based perspective correction
    extent = calculate_extent(side_global_filename)

    if extent < THRESHOLD:
        # 10. Apply iterative horizontal perspective correction to side-view image
        corrected_image_side = horizontal_iterative_correction(side_global_filename)
    else:
        # 11. Apply minimum bounding box perspective correction to side-view image
        image, width_rect, height_rect, angle_rect = extract_min_bounding_box(side_global_filename)

        # 12. Apply perspective correction to side-view image
        corrected_image_side = correct_perspective_min_bounding_box_side_view(image, width_rect, height_rect, angle_rect)

    # 13. Calculate maximum height from side-view image
    thickness = calculate_maximum_height(corrected_image_side)

    return length, width, thickness