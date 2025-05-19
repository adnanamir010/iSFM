import os
import cv2
import natsort

def load_images_from_folder(folder_path, extensions=None):
    """
    Loads images sequentially from a specified folder.

    Args:
        folder_path (str): The path to the folder containing images.
        extensions (list, optional): A list of allowed image file extensions
                                     (e.g., ['.jpg', '.png']).
                                     Defaults to ['.jpg', '.jpeg', '.png', '.tiff'].

    Yields:
        tuple: A tuple containing:
            - filename (str): The name of the loaded image file.
            - image (numpy.ndarray): The loaded image as a NumPy array (BGR format).
                                     Returns None for the image if loading fails.

    Raises:
        FileNotFoundError: If the specified folder_path does not exist.
        ValueError: If no images are found in the specified folder.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Error: The folder '{folder_path}' was not found.")

    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']

    image_files = []
    for filename in os.listdir(folder_path):
        if any(filename.lower().endswith(ext) for ext in extensions):
            image_files.append(filename)

    if not image_files:
        raise ValueError(f"No images with supported extensions found in '{folder_path}'.")

    # Sort files alphanumerically.
    image_files = natsort.natsorted(image_files) # Natural sort

    print(f"Found {len(image_files)} images. Processing order:")
    for i, filename in enumerate(image_files):
        print(f"  {i+1}. {filename}")

    for filename in image_files:
        file_path = os.path.join(folder_path, filename)
        try:
            image = cv2.imread(file_path)
            if image is None:
                print(f"Warning: Could not load image {filename}. Skipping.")
                yield filename, None
            else:
                yield filename, image
        except Exception as e:
            print(f"Error loading image {filename}: {e}. Skipping.")
            yield filename, None