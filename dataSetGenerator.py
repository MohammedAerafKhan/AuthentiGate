import cv2
import mediapipe as mp
import os
import numpy as np

def process_image(file_path):#Processes a static image to extract hand landmark data using MediaPipe.

    # Load the image
    image = cv2.imread(file_path)

    # Preprocess the image: Convert BGR to RGB and flip along Y-axis
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    flipped_image = cv2.flip(image_rgb, 1)

    # Setup MediaPipe Hands for hand landmark detection
    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

    # Process the flipped image to detect hand landmarks
    results = hands_detector.process(flipped_image)

    # Clean up the hands detector
    hands_detector.close()

    try:
        # Extract hand landmark data
        landmarks = results.multi_hand_landmarks[0]

        # Convert landmarks to string and split into lines
        landmarks_data = str(landmarks).strip().split('\n')

        # Remove unwanted lines
        excluded_lines = {'landmark {', '  visibility: 0.0', '  presence: 0.0', '}'}
        filtered_data = [line.strip()[2:] for line in landmarks_data if line not in excluded_lines]

        # Convert valid landmark data to floats
        landmarks_floats = [float(value) for value in filtered_data]
        return landmarks_floats
    except:
        # Return a zeroed array if no hand landmarks are found
        return np.zeros([1, 63], dtype=int)[0]

def generate_csv_dataset(): #    Generates a CSV dataset of hand landmark data with associated labels.

    # Define the folder containing action data and the output CSV file
    dataset_folder = 'Action_Data'
    output_file = 'hand_dataset.csv'

    with open(output_file, 'a') as csv_file:
        # Iterate over each folder in the dataset
        for label_folder in os.listdir(dataset_folder):
            if label_folder.startswith('._'):
                continue

            folder_path = os.path.join(dataset_folder, label_folder)
            for image_file in os.listdir(folder_path):
                if image_file.startswith('._'):
                    continue

                # Full path to the image file
                image_path = os.path.join(folder_path, image_file)
                # Extract hand landmark data
                landmarks = process_image(image_path)

                try:
                    # Write the landmarks and label to the CSV file
                    csv_file.write(','.join(map(str, landmarks)))
                    csv_file.write(f',{label_folder}\n')
                except:
                    # Handle any errors during writing
                    csv_file.write('0,' * 63 + 'None\n')

    print('Dataset generation completed successfully!')

if __name__ == "__main__":
    generate_csv_dataset()
