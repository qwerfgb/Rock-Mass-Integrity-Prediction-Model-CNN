import os
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import random


def extract_image_features_and_save(input_directory, output_excel_path=None):
    """
    Extract image features from the specified directory and save to an Excel file

    Parameters:
        input_directory (str): Input image directory path
        output_excel_path (str, optional): Output Excel file path, if None uses default path

    Returns:
        pandas.DataFrame: DataFrame containing all image features
    """
    # If output path is not specified, use default name
    if output_excel_path is None:
        output_excel_path = 'output_excel.xlsx'

    # Define image feature extraction function
    def extract_image_features(image_path):
        """
        Extract features from a single image

        Parameters:
            image_path (str): Image file path

        Returns:
            dict: Dictionary containing extracted features
        """
        try:
            # Load image
            image = Image.open(image_path)
            image_np = np.array(image)

            # Resize image to fixed size (224x224) for consistency
            resized_image = cv2.resize(image_np, (224, 224))

            # Feature 1: Color Histogram (normalized)
            color_hist = cv2.calcHist([resized_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
            color_hist = cv2.normalize(color_hist, color_hist).flatten()

            # Feature 2: Edge Detection using Sobel operator
            gray_image = cv2.cvtColor(resized_image, cv2.COLOR_RGB2GRAY)
            edges = cv2.Sobel(gray_image, cv2.CV_64F, 1, 1, ksize=5)
            edges_hist = np.histogram(edges, bins=64, range=(-4, 4))[0]

            # Feature 3: Texture Features using Gray Level Co-occurrence Matrix (GLCM)
            glcm = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray_image)
            texture_hist = np.histogram(glcm, bins=64, range=(0, 256))[0]

            return {
                'color_histogram': color_hist,
                'edges_histogram': edges_hist,
                'texture_histogram': texture_hist
            }
        except Exception as e:
            print(f"Error processing image {image_path}: {str(e)}")
            return None

    # Function to process all images in a directory
    def process_directory(directory_path):
        """
        Process all image files in the directory

        Parameters:
            directory_path (str): Directory path

        Returns:
            pandas.DataFrame: DataFrame containing all image features
        """
        data = []
        total_files = 0
        processed_files = 0

        # Count total number of files
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
                    total_files += 1

        # Process each image file
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.lower().endswith(('jpg', 'jpeg', 'png', 'bmp', 'gif')):
                    file_path = os.path.join(root, file)

                    processed_files += 1

                    features = extract_image_features(file_path)
                    if features:
                        row = {
                            'file_name': file,
                            # Record folder name as output label
                            'output': os.path.basename(root)
                        }
                        # Add feature values to data row
                        for key, values in features.items():
                            for i, value in enumerate(values):
                                row[f'{key}_bin_{i + 1}'] = value
                        data.append(row)

        return pd.DataFrame(data)

    # Process directory
    df = process_directory(input_directory)

    # Save DataFrame to Excel file
    df.to_excel(output_excel_path, index=False)

    print(f"Feature extraction completed and saved to {output_excel_path}")
    return df


def shuffle_excel_data(file_path, save_path=None, random_seed=None):
    """
    Read Excel file, randomly shuffle data rows and save

    Parameters:
        file_path (str): Input Excel file path
        save_path (str, optional): Path to save results, if None overwrites original file
        random_seed (int, optional): Random seed, if None uses random value

    Returns:
        pandas.DataFrame: Shuffled DataFrame
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        # Read Excel file
        df = pd.read_excel(file_path)

        if len(df) <= 1:
            print("Warning: Not enough data rows in Excel file, no need to shuffle")
            return df

        # Extract header and data parts
        header = df.iloc[0]  # First row as header
        data = df.iloc[1:]  # Second row and beyond as data

        # Set random seed
        if random_seed is None:
            random_seed = random.randint(1, 100000)

        # Shuffle data part
        data_shuffled = data.sample(frac=1, random_state=random_seed)

        # Recombine DataFrame
        df_shuffled = pd.concat([header.to_frame().T, data_shuffled], ignore_index=True)

        # Determine save path
        if save_path is None:
            save_path = file_path

        # Save to Excel file
        df_shuffled.to_excel(save_path, index=False)

        return df_shuffled

    except Exception as e:
        print(f"Error during processing: {str(e)}")
        return None


def process_and_shuffle_images(input_directory, output_excel_path=None, shuffle_output_path=None, random_seed=None):
    """
    Integrate image feature extraction and data shuffling into one workflow

    Parameters:
        input_directory (str): Input image directory path
        output_excel_path (str, optional): Output Excel file path
        shuffle_output_path (str, optional): Path to save shuffled Excel, if None overwrites original file
        random_seed (int, optional): Random seed for shuffling

    Returns:
        pandas.DataFrame: Shuffled DataFrame
    """
    # 1. Extract image features
    df = extract_image_features_and_save(input_directory, output_excel_path)

    if df is None or len(df) <= 1:
        print("Feature extraction failed or insufficient data, cannot shuffle")
        return None

    # 2. Shuffle data
    if shuffle_output_path is None:
        shuffle_output_path = output_excel_path

    shuffled_df = shuffle_excel_data(output_excel_path, shuffle_output_path, random_seed)

    return shuffled_df


if __name__ == "__main__":
    # Image directory path (categorized by folder name)
    input_directory = 'graded_images'
    # Feature extraction result save path
    output_excel_path = 'excel/test.xlsx'
    # Set to None to overwrite original file
    shuffle_output_path = None
    # Fixed random seed, set to None to use random value
    random_seed = 42

    # Execute complete workflow
    result_df = process_and_shuffle_images(
        input_directory=input_directory,
        output_excel_path=output_excel_path,
        shuffle_output_path=shuffle_output_path,
        random_seed=random_seed
    )

    if result_df is not None:
        print("The entire process has been successfully completed!")
    else:
        print("An error occurred during processing, please check the logs")
