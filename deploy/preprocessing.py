import cv2
import os
import numpy as np
import glob 


# Function to extract 180 video frames
def extract_180_frames(video_path, output_folder):
    """
    Extract all 180 frames from the input video and output to a destination folder.
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
def fill_holes_for_specific_mask(binary_mask):
    """
    Fill holes in the extracted binary mask using external contours.
    """

    inverted_mask = cv2.bitwise_not(binary_mask)
    contours, hierarchy = cv2.findContours(inverted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("No contours found in the inverted mask. Returning original.")
        return binary_mask

    largest_contour = max(contours, key=cv2.contourArea)
    filled_region = np.zeros_like(binary_mask)
    cv2.drawContours(filled_region, [largest_contour], 0, 255, thickness=cv2.FILLED)
    final_filled_mask = filled_region 

    return final_filled_mask


# Helper function to extract binary mask
def repair_mask_edges(binary_mask, kernel_size=7, iterations=2):
    # Define the kernel for morphological closing
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Apply morphological closing (dilation followed by erosion)
    closed_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    return closed_mask


# Function to extract binary mask
def extract_binary_masks(input_folder, output_folder):
    """
    Extract the binary mask from the extracted video frames using the following steps:
    Step 1: Extract binary mask using Otsu's Thresholding.
    Step 2: Fill holes in the extracted binary masks using external contours.
    Step 3: Repair the edges of extracted binary masks using smoothing with morphological closing.
    """

    image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output directory: {output_folder}")

    if not os.path.exists(input_folder):
        print(f"Error: Input directory '{input_folder}' not found.")
    else:
        image_files = os.listdir(input_folder)

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
                    
                    # 1. Apply Otsu's Binarization
                    retVal, binary_mask = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # 2. Apply the hole-filling function to the image array
                    processed_mask = fill_holes_for_specific_mask(binary_mask)

                    # 3. Apply morphological closing (dilation followed by erosion)
                    closed_mask = repair_mask_edges(processed_mask, kernel_size=7, iterations=2)

                    mask_filename = os.path.splitext(filename)[0] + ".png"
                    output_path = os.path.join(output_folder, mask_filename)
                    cv2.imwrite(output_path, closed_mask)
                    print(f"Processed '{filename}' and saved mask to '{mask_filename}'")

            print("\nAll images processed successfully.")


# Function to calculate maximum length of binary mask to extract frontal image
def calculate_maximum_length(binary_mask):
    """
    Finds the maximum width (length) of the largest BLACK object (contour), 
    and draws a purely HORIZONTAL line connecting the extreme left/right X-coordinates 
    at a central Y-reference point.
    """
    
    # We find contours directly on the binary mask to target the black regions.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR), 0.0

    # 1. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 2. Extract all coordinates (x, y) from the largest contour
    all_points = largest_contour.reshape(-1, 2)
    
    # 3. Find the minimum and maximum X values
    extreme_left_x = np.min(all_points[:, 0])
    extreme_right_x = np.max(all_points[:, 0])
    
    maximum_length = extreme_right_x - extreme_left_x
    
    return float(maximum_length)


# Function to extract index of frontal image filename
def extract_frontal_image(input_folder):
    image_extensions = ('*.png', '*.jpg', '*.jpeg')
    SAMPLE_SIZE = 5 
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_folder, ext)))
    
    image_paths.sort()
    
    if not image_paths:
        print(f"Error: No images found in the directory: {input_folder}")
        exit()
    
    all_lengths = []
    processed_count = 0
    
    for input_path in image_paths:
        binary_mask = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if binary_mask is not None:
            length = calculate_maximum_length(binary_mask)
            all_lengths.append(length)
            processed_count += 1

    analysis_paths = image_paths[:SAMPLE_SIZE]
    max_first_five = {'path': None, 'length': float('-inf')}
    
    for path in analysis_paths:
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        
        if mask is None:
            continue
            
        length = calculate_maximum_length(mask) 
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
    # Extract front view image frame index
    front_view_image = global_filename
    print(f"front_view_image: {front_view_image}")
    front_view_image_index = front_view_image.split("_")[1]
    front_view_image_index = int(front_view_image_index.split(".")[0])
    print(f"front_view_image_index: {front_view_image_index}")

    # Extract front view image frame index
    side_view_image_index = front_view_image_index + 45
    side_view_image = "frame_00" + str(side_view_image_index) + ".png"
    print(f"side view image filename: {side_view_image}") 

    return side_view_image


# Function to calculate maximum length and width from frontal image
def calculate_max_dimensions_combined(global_filename):
    """
    Calculates length and width of frontal image using maximum length and width
    """
    image_path = "/temp/" + global_filename 
    binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    maximum_width = 0.0
    maximum_length = 0.0
    conversion_factor = 0.00274 * 2

    if not contours:
        print("No black object contours found in the mask.")
        return drawn_mask, maximum_width, maximum_length

    largest_contour = max(contours, key=cv2.contourArea)
    all_points = largest_contour.reshape(-1, 2)
    
    # Y-Extremes (Width)
    extreme_top_y = np.min(all_points[:, 1])
    extreme_bottom_y = np.max(all_points[:, 1])
    maximum_width = extreme_bottom_y - extreme_top_y 
    print(f"maximum_width: {maximum_width}")

    # X-Extremes (Length)
    extreme_left_x = np.min(all_points[:, 0])
    extreme_right_x = np.max(all_points[:, 0])
    maximum_length = extreme_right_x - extreme_left_x
    print(f"maximum_length: {maximum_length}")

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

    print(f"Length: {length:.2f} mm")
    print(f"Width: {width:.2f} mm")

    return float(length), float(width)


# Function to calculate maximum height from side view image
def calculate_maximum_height(global_filename):
    """
    Finds the maximum height of the largest black object (contour), 
    and draws a purely VERTICAL line connecting the extreme top/bottom Y-coordinates 
    at a central X-reference point.
    """
    image_path = "/temp/" + global_filename 
    binary_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_mask = cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR)
    maximum_height = 0.0
    conversion_factor = 0.00274 * 2
    
    # We find contours directly on the binary mask to target the black regions.
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("No black object contours found in the mask.")
        return cv2.cvtColor(drawn_mask, cv2.COLOR_GRAY2BGR), 0.0

    # 1. Find the largest contour
    largest_contour = max(contours, key=cv2.contourArea)

    # 2. Extract all coordinates (x, y) from the largest contour
    all_points = largest_contour.reshape(-1, 2)
    
    # 3. Find the minimum and maximum Y values (Height is difference between min/max Y)
    extreme_top_y = np.min(all_points[:, 1])    # Top edge has smallest Y
    extreme_bottom_y = np.max(all_points[:, 1]) # Bottom edge has largest Y
    
    maximum_height = extreme_bottom_y - extreme_top_y
    height = maximum_height * conversion_factor
    height = round(height, 2)
    print(f"maximum_height: {height}")

    return float(height)


# Function to process dimensions
def process_dimensions(video_path, input_folder, output_folder):
    # 1. Extract 180 video frames
    extract_180_frames(video_path, output_folder)

    # 2. Extract binary mask
    extract_binary_masks(input_folder, output_folder)

    # 3. Extract index of frontal image filename
    front_global_filename = extract_frontal_image(input_folder)

    # 4. Calculate maximum length and width from frontal image
    length, width = calculate_max_dimensions_combined(front_global_filename)

    # 5. Extract index of side image filename
    side_global_filename = extract_side_image(front_global_filename)

    # 6. Calculate maximum height from side view image
    height = calculate_maximum_height(side_global_filename)

    return length, width, height