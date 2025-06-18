# Importing libraries  
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import random
import xml.etree.ElementTree as ET
import re

# Get the current file's directory
base_dir = os.path.dirname(__file__)

# Path to dataset
Stenosis_path = os.path.join(base_dir, "Data", "Stenosis detection", "dataset")

# List all XML files in the dataset directory
xml_files = [f for f in os.listdir(Stenosis_path) if f.endswith('.xml')]

# Extract the sequence number from the filename
def extract_sequence_number(filename):
    match = re.search(r'_(\d+)\.xml$', filename)
    return int(match.group(1)) if match else None

# Group XML files by sequence
sequences = {}
for xml_file in xml_files:
    sequence_number = extract_sequence_number(xml_file)
    if sequence_number is not None:
        base_name = xml_file.rsplit('_', 1)[0]
        if base_name not in sequences:
            sequences[base_name] = []
        sequences[base_name].append((sequence_number, xml_file))

# Create a VideoWriter object
output_video_path = os.path.join(Stenosis_path, 'output_all_sequences.avi')
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video_writer = None

# Batch size for processing frames
batch_size = 50

# Iterate through all sequences
for base_name, sequence in sequences.items():
    sequence.sort()
    patient_number = base_name
    print(f"Processing patient: {patient_number}")

    frames = []  # List to store frames for the current batch

    for _, xml_file in sequence:
        xml_file_path = os.path.join(Stenosis_path, xml_file)
        print(f"Processing file: {xml_file_path}")

        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Extract the image filename
        image_filename = root.find('filename').text

        # Construct the correct image path
        image_path = os.path.join(Stenosis_path, image_filename)
        print(f"Image path: {image_path}")

        # Check if the image file exists
        if not os.path.exists(image_path):
            print(f"Error: The image file '{image_path}' does not exist.")
            continue

        # Extract bounding box coordinates
        bndbox = root.find('.//bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)
        print(f"Bounding box: ({xmin}, {ymin}), ({xmax}, {ymax})")

        # Load the image
        image = cv2.imread(image_path)

        # Check if the image was loaded successfully
        if image is None:
            print(f"Error: Failed to load the image '{image_path}'.")
            continue

        # Draw the bounding box
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

        # Add patient number text at the top of the image
        cv2.putText(image, f"Patient: {patient_number}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Append the processed frame to the list
        frames.append(image)
        print(f"Frame appended for patient: {patient_number}")

        # If the batch size is reached, write the frames to the video and clear the list
        if len(frames) >= batch_size:
            if video_writer is None:
                height, width, _ = frames[0].shape
                video_writer = cv2.VideoWriter(output_video_path, fourcc, 15, (width, height))  # Set to 15 fps
                print(f"VideoWriter initialized: {output_video_path}")

            for frame in frames:
                video_writer.write(frame)
                print(f"Frame written to video")

            frames.clear()  # Clear the list of frames

# Write any remaining frames to the video
if frames:
    if video_writer is None:
        height, width, _ = frames[0].shape
        video_writer = cv2.VideoWriter(output_video_path, fourcc, 15, (width, height))  # Set to 15 fps
        print(f"VideoWriter initialized: {output_video_path}")

    for frame in frames:
        video_writer.write(frame)
        print(f"Frame written to video")

# Release the VideoWriter object
if video_writer is not None:
    video_writer.release()
print("VideoWriter released")

# Display the video
for i, frame in enumerate(frames):
    cv2.imshow('Image with Bounding Box', frame)
    cv2.waitKey(67)  # Display each frame for 67 ms
    print(f"Frame {i+1}/{len(frames)} displayed")

cv2.destroyAllWindows()
print("All windows destroyed")
