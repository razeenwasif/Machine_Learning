import cv2
import numpy as np
import os

def convert_image_to_white_background(image_path, output_path):
    # Load the image with transparency (if it exists)
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Load with unchanged flag to keep alpha channel

    if img.shape[2] == 4:  # Check if image has 4 channels (RGBA)
        # Separate the color channels and the alpha channel
        r, g, b, a = cv2.split(img)

        # Create a white background (all 255)
        white_background = np.ones_like(a) * 255

        # Combine the RGB channels with a white background where alpha is 0 (transparent)
        alpha_mask = a / 255.0
        r = r * alpha_mask + white_background * (1 - alpha_mask)
        g = g * alpha_mask + white_background * (1 - alpha_mask)
        b = b * alpha_mask + white_background * (1 - alpha_mask)

        # Merge the RGB channels back
        img = cv2.merge([r, g, b])

    # Save the image without transparency
    cv2.imwrite(output_path, img)

# Example usage for converting all images in a folder
input_folder = "Dataset/"
output_folder = "dataset/"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        convert_image_to_white_background(input_path, output_path)
        print(f"Converted {filename} to have a white background")

