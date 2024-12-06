import cv2 as cv
import numpy as np
from pathlib import Path
import os

def imageGenerator(class_label, output_dir='DATASET', max_images=1000, skip_frames=5): # Captures images using the webcam to build a dataset for a specific action class
    # Create the directory for the specific class
    class_path = os.path.join(output_dir, class_label)
    Path(class_path).mkdir(parents=True, exist_ok=True)

    # Get the starting index for image numbering
    existing_images = [
        int(file_name.split('.')[0]) for file_name in os.listdir(class_path)
        if file_name.endswith(('.png', '.jpg'))
    ]
    start_index = max(existing_images, default=0) + 1

    # Initialize the webcam
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return

    print(f"Starting image capture for class '{class_label}'.")
    print(f"Images will be saved in: {class_path}")
    print(f"Press 'q' to quit at any time.")

    frame_counter = 0  # To keep track of frames
    saved_images = 0  # To count saved images

    while saved_images < max_images:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        frame_counter += 1
        if frame_counter % skip_frames == 0:
            # Save the current frame
            image_path = os.path.join(class_path, f'{start_index + saved_images}.png')
            cv.imwrite(image_path, frame)
            saved_images += 1

            # Provide real-time feedback to the user
            cv.putText(frame, f"Saved: {saved_images}/{max_images}", 
                       (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the current frame in a window
        cv.imshow(f"Capturing {class_label}", frame)

        # Exit on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            print("Exiting capture...")
            break

    # Release the webcam and close all windows
    cap.release()
    cv.destroyAllWindows()

    print(f"Capture completed. Total images saved: {saved_images}/{max_images}")


if __name__ == '__main__':
    # Prompt the user to enter the class label for the dataset
    action_class = input("Enter the action for the dataset: ")
    imageGenerator(action_class)
