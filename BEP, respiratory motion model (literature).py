from skimage import morphology, transform
import matplotlib.pyplot as plt
import numpy as np  # Fix the import statement
import os
import cv2
import xml.etree.ElementTree as ET

max_pixel_r = 4 # define maximum size in pixels of artery
down_sample_rate = 4 # how much do we downsample the imagey

# the morphological disk that is used for filtering
footprint = morphology.disk(max_pixel_r)

# Locate dataset directory
base_dir = os.path.dirname(__file__)
dataset_dir = os.path.join(base_dir, "Data", "Stenosis detection", "dataset")

# Debug: Verify dataset directory
if not os.path.exists(dataset_dir):
    print(f"Dataset directory does not exist: {dataset_dir}", flush=True)
    exit()

# Load BMP images and corresponding XML files
bmp_files = [f for f in os.listdir(dataset_dir) if f.endswith('.bmp')]
patients = list(set('_'.join(f.split('_')[1:3]) for f in bmp_files))

# Calculate the number of frames for each patient
patient_frame_counts = {patient: len([f for f in bmp_files if f"_{patient}_" in f]) for patient in patients}

# Select the top 6 patients with the most frames
top_6_patients = sorted(patient_frame_counts, key=patient_frame_counts.get, reverse=True)[:6]

# Debug: Log the top 6 patients and their frame counts
print("Top 6 patients with the most frames:")
for patient in top_6_patients:
    print(f"{patient}: {patient_frame_counts[patient]} frames", flush=True)

# Create output directory
output_dir = os.path.join(dataset_dir, "Output")
os.makedirs(output_dir, exist_ok=True)

# Create full_videos directory
full_videos_dir = os.path.join(dataset_dir, "full_videos")
os.makedirs(full_videos_dir, exist_ok=True)

# Process each of the top 6 patients
for idx, selected_patient in enumerate(top_6_patients):
    print(f"Processing patient: {selected_patient} with {patient_frame_counts[selected_patient]} frames", flush=True)

    # Filter files for the selected patient
    patient_files = [f for f in bmp_files if f"_{selected_patient}_" in f]
    nb_frames = len(patient_files)
    pixel_array = []

    # Debug: Log the number of files for selected patient
    print(f"Number of files for selected patient: {len(patient_files)}", flush=True)

    # Ensure there are files to process
    if not patient_files:
        print("No files found for the selected patient. Skipping.", flush=True)
        continue

    # Sort patient files numerically by frame number
    def extract_frame_number(filename):
        # Extract the numeric part of the frame from the filename (e.g., "frame_0001.bmp")
        parts = filename.split('_')
        if len(parts) > 1 and parts[-1].endswith('.bmp'):
            return int(parts[-1].replace('.bmp', ''))
        return 0

    patient_files = sorted(patient_files, key=extract_frame_number)

    # Debug: Log sorted patient files
    print("Sorted patient files:", flush=True)
    for file in patient_files:
        print(file, flush=True)

    # Define a consistent size for all cropped images
    target_size = (128, 128)  # Example size, adjust as needed

    # Prepare video writers
    image_video_path = os.path.join(output_dir, f"Patient_{selected_patient}_images.avi")
    plot_video_path = os.path.join(output_dir, f"Patient_{selected_patient}_plots.avi")
    combined_video_path = os.path.join(full_videos_dir, f"Patient_{selected_patient}_full.avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    image_video_writer = None
    plot_video_writer = None
    combined_video_writer = None

    # List to store plot images
    plot_images = []

    # List to store bounding box center coordinates
    bounding_box_centers = []

    # Process frames for the selected patient
    for frame_idx, file in enumerate(patient_files):
        # Log the file being processed
        print(f"Processing file: {file}", flush=True)

        # Debug: Check if file exists
        bmp_path = os.path.join(dataset_dir, file)
        if not os.path.exists(bmp_path):
            print(f"File does not exist: {bmp_path}", flush=True)
            continue

        # Load BMP image
        bmp_path = os.path.join(dataset_dir, file)
        image = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)

        # Check if the image is valid
        if image is None:
            print(f"Failed to load image: {file}", flush=True)
            continue

        # Parse corresponding XML file for bounding box
        xml_path = bmp_path.replace('.bmp', '.xml')
        if not os.path.exists(xml_path):
            print(f"XML file does not exist for {file}: {xml_path}", flush=True)
            continue

        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            bndbox = root.find(".//bndbox")
            if bndbox is None:
                print(f"No bounding box found in XML file for {file}: {xml_path}", flush=True)
                continue

            xmin, ymin = int(bndbox.find("xmin").text), int(bndbox.find("ymin").text)
            xmax, ymax = int(bndbox.find("xmax").text), int(bndbox.find("ymax").text)

            # Validate bounding box dimensions
            if xmin >= xmax or ymin >= ymax:
                print(f"Invalid bounding box dimensions for {file}: xmin={xmin}, ymin={ymin}, xmax={xmax}, ymax={ymax}", flush=True)
                continue

            # Calculate bounding box center and store it
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            bounding_box_centers.append((center_x, center_y))

        except Exception as e:
            print(f"Error parsing XML file for {file}: {e}", flush=True)
            continue

        # Draw bounding box and center on the original image
        image_with_bndbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(image_with_bndbox, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.circle(image_with_bndbox, (int(center_x), int(center_y)), 5, (0, 0, 255), -1)  # Draw center as a red dot

        # Initialize video writer if not already initialized
        if image_video_writer is None:
            height, width = image_with_bndbox.shape[:2]
            image_video_writer = cv2.VideoWriter(image_video_path, fourcc, 10, (width, height))

        # Write frame with bounding box and center to video
        image_video_writer.write(image_with_bndbox)

        # Resize the full image to the target size
        try:
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Skipping file {file} due to resizing error: {e}", flush=True)
            continue

        # Log the shape of the resized image
        print(f"File {file}: Resized image shape = {resized_image.shape}", flush=True)

        # Append the resized image to the pixel array
        pixel_array.append(resized_image)

    # Normalize the bounding box centers for plotting
    if bounding_box_centers:
        bounding_box_centers = np.array(bounding_box_centers)
        normalized_centers = (bounding_box_centers - np.min(bounding_box_centers, axis=0)) / (np.max(bounding_box_centers, axis=0) - np.min(bounding_box_centers, axis=0))
        xs = np.arange(0, len(normalized_centers))

    # Process frames for the selected patient
    for frame_idx, file in enumerate(patient_files):
        # Log the file being processed
        print(f"Processing file: {file}", flush=True)

        # Debug: Check if file exists
        bmp_path = os.path.join(dataset_dir, file)
        if not os.path.exists(bmp_path):
            print(f"File does not exist: {bmp_path}", flush=True)
            continue

        # Load BMP image
        bmp_path = os.path.join(dataset_dir, file)
        image = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)

        # Draw bounding box on the original image (optional, can be skipped)
        image_with_bndbox = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Initialize video writer if not already initialized
        if image_video_writer is None:
            height, width = image_with_bndbox.shape[:2]
            image_video_writer = cv2.VideoWriter(image_video_path, fourcc, 10, (width, height))

        # Write frame to video
        image_video_writer.write(image_with_bndbox)

        # Resize the full image to the target size
        try:
            resized_image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            print(f"Skipping file {file} due to resizing error: {e}")
            continue

        # Log the shape of the resized image
        print(f"File {file}: Resized image shape = {resized_image.shape}")

        # Append the resized image to the pixel array
        pixel_array.append(resized_image)

        # Create a plot for the current frame
        plt.figure(figsize=(8, 5))
        plt.plot(xs, normalized_centers[:, 0], label="Center X", color="blue")
        plt.plot(xs, normalized_centers[:, 1], label="Center Y", color="green")
        plt.scatter(frame_idx, normalized_centers[frame_idx, 0], color='red', label="Current Frame X")
        plt.scatter(frame_idx, normalized_centers[frame_idx, 1], color='orange', label="Current Frame Y")
        plt.title(f"Bounding Box Center Changes for Patient {selected_patient}")
        plt.xlabel("Frame Index")
        plt.ylabel("Normalized Center Coordinates")
        plt.legend()
        plt.tight_layout()

        # Save the plot to a temporary file
        plot_image_path = os.path.join(output_dir, f"Patient_{selected_patient}_Frame_{frame_idx}.png")
        plt.savefig(plot_image_path)
        plt.close()

        # Load the saved plot image and store it in the list
        plot_image = cv2.imread(plot_image_path)
        plot_images.append(plot_image)

    # Update width and height based on the first image in pixel_array
    if pixel_array:
        height, width = pixel_array[0].shape
    else:
        print(f"No valid images in pixel_array for patient {selected_patient}. Skipping video generation.", flush=True)
        continue

    # Create a video from the plot images
    if plot_images:
        height, width = plot_images[0].shape[:2]
        plot_video_writer = cv2.VideoWriter(plot_video_path, fourcc, 10, (width, height))
        for plot_image in plot_images:
            plot_video_writer.write(plot_image)
        plot_video_writer.release()

    # Combine the image and plot videos side by side
    if image_video_writer is not None:
        image_video_writer.release()
    if os.path.exists(image_video_path) and os.path.exists(plot_video_path):
        cap_image = cv2.VideoCapture(image_video_path)
        cap_plot = cv2.VideoCapture(plot_video_path)

        # Get video properties
        frame_width = int(cap_image.get(cv2.CAP_PROP_FRAME_WIDTH) + cap_plot.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap_image.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap_image.get(cv2.CAP_PROP_FPS))

        # Initialize combined video writer
        combined_video_writer = cv2.VideoWriter(combined_video_path, fourcc, fps, (frame_width, frame_height))

        while True:
            ret_image, frame_image = cap_image.read()
            ret_plot, frame_plot = cap_plot.read()

            if not ret_image or not ret_plot:
                break

            # Resize frames to have the same height
            if frame_image.shape[0] != frame_plot.shape[0]:
                if frame_image.shape[0] > frame_plot.shape[0]:
                    frame_plot = cv2.resize(frame_plot, (frame_plot.shape[1], frame_image.shape[0]))
                else:
                    frame_image = cv2.resize(frame_image, (frame_image.shape[1], frame_plot.shape[0]))

            # Combine frames side by side
            combined_frame = np.hstack((frame_image, frame_plot))
            combined_video_writer.write(combined_frame)

        cap_image.release()
        cap_plot.release()
        combined_video_writer.release()

    print(f"Saved combined video for patient {selected_patient} in {full_videos_dir}", flush=True)

    # Plot the bounding box center changes
    plt.figure(figsize=(8, 5))
    plt.plot(xs, normalized_centers[:, 0], label="Center X", color="blue")
    plt.plot(xs, normalized_centers[:, 1], label="Center Y", color="green")
    plt.title(f"Bounding Box Center Changes for Patient {selected_patient}")
    plt.xlabel("Frame Index")
    plt.ylabel("Normalized Center Coordinates")
    plt.legend()
    plt_path = os.path.join(output_dir, f"Patient_{selected_patient}_Bounding_Box_Center.png")
    plt.savefig(plt_path)
    plt.close()

    print(f"Saved bounding box center plot and video for patient {selected_patient} in {output_dir}", flush=True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()