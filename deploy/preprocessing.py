import cv2
import os
import numpy as np
import glob 


# Function to extract 180 video frames
def extract_180_frames(video_path, output_folder):
    """
    Extract all 180 frames from the input video and output to a destination folder.

    Args:
        video_path: Path of input videos.
        output_folder: Path of output folder.

    Returns:
        None
    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    vidcap = cv2.VideoCapture(video_path)
    if not vidcap.isOpened():
        raise RuntimeError(f"Could not open video file at path: {video_path}. Check file existence and OpenCV/FFmpeg codecs.")

    TOTAL_FRAMES_TO_PROCESS = 180
    frames_processed = 0

    while frames_processed < TOTAL_FRAMES_TO_PROCESS:
        success, image = vidcap.read()

        if not success:
            print(f"Warning: Video ended early at frame {frames_processed}.")
            break

        frame_filename = os.path.join(output_folder, f"frame_{frames_processed:04d}.jpg")
        cv2.imwrite(frame_filename, image)
        print(f"Extracted Frame {frames_processed} (Saved as {os.path.basename(frame_filename)})")
        frames_processed += 1 

    vidcap.release()
    print(f"Total frames processed: {frames_processed}")


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
        input_folder: The input folder containing 180 video frames.
        output_folder: The output folder to save the binary masks.

    Returns:
        None
    """

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' not found.")
    else:
        image_files = os.listdir(input_folder)
        image_files = [i for i in image_files if i.split(".")[-1] !="png"]
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


# Function to calculate maximum length of binary mask to extract frontal image
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


# Function to extract index of frontal image filename
def extract_frontal_image(input_folder):
    """
    Finds the frontal image filename based on maximum length from binary masks.

    Args:
        input folder: The input folder containing all the 180 processed binary masks.

    Returns:
        string: A string containing the path of the frontal image filename.
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

    analysis_paths = image_paths[:SAMPLE_SIZE]
    max_first_five = {'path': None, 'length': float('-inf')}
    
    for path in analysis_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        length, _, _ = calculate_maximum_length(mask) 
        if length > max_first_five['length']:
            max_first_five['length'] = length
            max_first_five['path'] = path

    if max_first_five['path']:
        global_filename = os.path.basename(max_first_five['path'])
        global_filename = global_filename.replace(".jpg", ".png")
        print(f"global_filename: {global_filename}")
        
        return global_filename
    else:
        print(f"Could not find a valid frame with length data in the first {SAMPLE_SIZE} frames.")


# Function to extract index of side image filename
def extract_side_image(global_filename):
    """
    Finds the side-view image filename based on the frontal image filename.

    Args:
        global filename: The filename of the frontal image.

    Returns:
        string: A string containing the path of the side-view image filename.
    """
    # Extract front view image frame index
    front_view_image = global_filename
    front_view_image_index = front_view_image.split("_")[1]
    front_view_image_index = int(front_view_image_index.split(".")[0])

    # Extract front view image frame index
    side_view_image_index = front_view_image_index + 45
    side_view_image = "frame_00" + str(side_view_image_index) + ".png"
    print(f"side view image filename: {side_view_image}") 

    return side_view_image


# Function to calculate maximum length and width from frontal image
def calculate_max_dimensions_combined(image):
    """
    Calculates length and width of frontal image using maximum length and width.

    Args:
        image: The input frontal image to process the maximum length and width.

    Returns:
        tuple: A tuple containing the length and width the frontal image.
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
        global filename: The filename of the frontal image which is rectangular.

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
def correct_perspective_horizontal_mab(image, angle_deg, center=None):
    """
    Applies rotation based on the angle outputted from extract_min_bounding_box.
    
    Args:
        image: The frontal image which is rectangular.
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


# Helper function to detect rectangular shapes
def calculate_extent(global_filename):
    """
    Detect rectangular shape from the input frontal image.
    
    Args:
        global filename: The frontal image to be checked.

    Returns:
        float: A float containing the extent of the object.
    """
    image_path = "/temp/" + global_filename 
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 1. Find contours in the grayscale image
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 2. Check if any contours were found
    if not contours:
        print("No contours found.")
        return None

    # 3. Select the largest contour (assumes the shape is the main object)
    c = max(contours, key=cv2.contourArea)

    # 4. Get Min Area Rect (Rotated Bounding Box)
    rect = cv2.minAreaRect(c)
    (_, _), (width, height), _ = rect
    box_area = width * height
    shape_area = cv2.contourArea(c)

    # 5. Extent (How much the shape fills the bounding box)
    extent = shape_area / box_area if box_area > 0 else 0

    return extent


# Function to check for rectangular shape
def check_rectangular(extent):
    """
    Checks for rectangular shape based on custom criteria.
    
    Args:
        extent: The extent of the object in the frontal image to be checked.

    Returns:
        bool: A boolean whether the shape is rectangular or not.
    """
    is_rectangular = (extent >= 0.85)
    if is_rectangular:
        print("✅ Shape detected as RECTANGULAR!")
    else:
        print(f"❌ Shape is NOT rectangular.")
    return is_rectangular


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


# Function to apply iterative horizontal perspective correction for non-rectangular shapes
def iterative_horizontal_perspection_correction(global_filename):
    """" 
    Apply iterative horizontal perspective correction till convergence for non-rectangular shapes.
    
    Args:
        global filename: The filename of the input frontal image.

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


# Function to calculate maximum height from side view image
def calculate_maximum_height(image):
    """
    Finds the maximum height of the largest WHITE object (contour).

     Args:
        image (np.array): The input side view image.

    Returns:
        float: The maximum height of the side view image.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    maximum_height = 0.0
    conversion_factor = 0.00274 * 2
    
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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
def process_dimensions(video_path, input_folder, output_folder):
    """
    Finds the length, width and thickness of the input video.

     Args:
        video: The input video.
        input_folder: The input folder containing the extracted 180 video frames.
        output_folder: The output folder to save the binary masks.
        
    Returns:
        tuple: The tuple containing the length, width and thickness of the input video.
    """
    # 1. Extract 180 video frames
    extract_180_frames(video_path, output_folder)

    # 2. Extract binary mask
    extract_binary_masks(input_folder, output_folder)

    # 3. Extract index of frontal image filename
    front_global_filename = extract_frontal_image(input_folder)

    # 4. Check for rectangular shape
    extent = calculate_extent(front_global_filename)
    is_rectangular = check_rectangular(extent)
    
    # 5. Calculate length and width for rectangular shapes
    if is_rectangular:
        image, _, _, angle = extract_min_bounding_box(front_global_filename)
        corrected_image, _ = correct_perspective_horizontal_mab(image, angle, center=None)
        length, width = calculate_max_dimensions_combined(corrected_image)

    # 6. Calculate length and width for non-rectangular shapes
    else:
        corrected_image = iterative_horizontal_perspection_correction(front_global_filename)
        length, width = calculate_max_dimensions_combined(corrected_image)

    # 7. Extract index of side image filename
    side_global_filename = extract_side_image(front_global_filename)

    # 8. Apply iterative horizontal perspective correction to side view image
    corrected_mask = iterative_horizontal_perspection_correction(side_global_filename)

    # 9. Calculate maximum height from side view image
    thickness = calculate_maximum_height(corrected_mask)

    return length, width, thickness