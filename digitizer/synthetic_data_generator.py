import os
import random
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import cv2
from scipy.ndimage import gaussian_filter, map_coordinates
import matplotlib.pyplot as plt 

# Configuration
FONT_PATH = "./fonts/"
OUTPUT_DIR = "./synthetic_dataset/"
NUM_SAMPLES = 10000 # number of word images to generate
IMAGE_SIZE = (512, 128) # fixed size for generated images 

# Load word list
with open("./wordlist/words_dictionary.json", "r") as f:
    WORD_LIST = json.load(f)

WORD_LIST = list(WORD_LIST.keys())

# filter out empty strings
WORD_LIST = [word for word in WORD_LIST if word.strip()]

# debug 
print(f"Loaded {len(WORD_LIST)} words.")

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load available fonts 
def load_fonts():
    return [os.path.join(FONT_PATH, f) for f in os.listdir(FONT_PATH) if f.endswith(".ttf")]

FONTS = load_fonts()

# Apply elastic distortion
def elastic_transform(image, alpha, sigma):
    random_state = np.random.RandomState(None)
    shape = image.shape 
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha 
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, indices, order=1).reshape(shape)

# generate a synthethic word image 
def generate_word_image(word, font_path):
    font = ImageFont.truetype(font_path, 28) # adjust size 
    image = Image.new("L", IMAGE_SIZE, 255) # white background
    draw = ImageDraw.Draw(image)
    text_size = draw.textbbox((0,0), word, font=font)
    text_x = (IMAGE_SIZE[0] - text_size[0]) // 2  
    text_y = (IMAGE_SIZE[1] - text_size[1]) // 2
    draw.text((text_x, text_y), word, font=font, fill=0)

    # convert to numpy array for augmentation
    img_array = np.array(image)

    # Apply augmentations
    if random.random() > 0.5:
        img_array = elastic_transform(img_array, alpha=4, sigma=1)
        img_array = np.rot90(img_array, k=1 if random.random() > 0.5 else 3) 
        
    return Image.fromarray(img_array)

#======================== Testing ==============================
# generate image and display
word = random.choice(WORD_LIST)
font = random.choice(FONTS)
test_image = generate_word_image(word, font)
# display the image 
plt.imshow(test_image, cmap='gray')
plt.title(f"Generated Word: {word}", fontsize=14)
plt.axis('off')
plt.show()
#===============================================================

# generate dataset 
for i in range(NUM_SAMPLES):
    word = random.choice(WORD_LIST)
    font = random.choice(FONTS)
    image = generate_word_image(word, font)
    image.save(os.path.join(OUTPUT_DIR, f"{word}_{i}.png"))

print(f"Generated {NUM_SAMPLES} synthetic handwritten word images in {OUTPUT_DIR}")
