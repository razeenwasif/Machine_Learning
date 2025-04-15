import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont, ImageOps
import os
import random
from pathlib import Path
import sys
import traceback

def generate_character_images(characters, font_paths, size=(50, 50), variations_per_font=10):
    """
    Generate synthetic character images using different fonts and variations.
    
    Args:
        characters: Dictionary mapping character labels to actual characters
        font_paths: List of paths to font files
        size: Target size of the output images (default: 50x50)
        variations_per_font: Number of variations to generate per font
        
    Returns:
        Tuple of (images, labels) as numpy arrays
    """
    images = []
    labels = []
    
    # Create a temporary directory for debugging if needed
    # temp_dir = Path("temp_images")
    # temp_dir.mkdir(exist_ok=True)
    
    print(f"Generating synthetic data for {len(characters)} characters using {len(font_paths)} fonts")
    
    # Try the standard method first
    try:
        for char_label, char in characters.items():
            print(f"Processing character: {char} ({char_label})")
            for font_path in font_paths:
                try:
                    # Try different font sizes
                    for font_size in range(30, 41, 5):
                        # Create a blank image (larger than final size to allow for transformations)
                        img = Image.new('L', (100, 100), color=0)
                        draw = ImageDraw.Draw(img)
                        
                        try:
                            # Load font
                            font = ImageFont.truetype(font_path, font_size)
                            
                            # Calculate text position to center it
                            # Handle different Pillow versions
                            try:
                                # For older Pillow versions
                                text_width, text_height = draw.textsize(char, font=font)
                            except AttributeError:
                                try:
                                    # For newer Pillow versions (>=8.0.0)
                                    text_width, text_height = font.getsize(char)
                                except AttributeError:
                                    # For Pillow 9.0.0+
                                    left, top, right, bottom = font.getbbox(char)
                                    text_width = right - left
                                    text_height = bottom - top
                            
                            position = ((100 - text_width) // 2, (100 - text_height) // 2)
                            
                            # Draw the character
                            draw.text(position, char, fill=255, font=font)
                            
                            # Apply variations
                            for i in range(variations_per_font):
                                # Apply random transformations
                                img_variation = img.copy()
                                
                                # Random rotation (-10 to 10 degrees)
                                angle = random.uniform(-10, 10)
                                img_variation = img_variation.rotate(angle, resample=Image.BILINEAR, expand=False)
                                
                                # Random shift (-5 to 5 pixels)
                                shift_x = random.randint(-5, 6)
                                shift_y = random.randint(-5, 6)
                                img_variation = img_variation.transform(
                                    img_variation.size, 
                                    Image.AFFINE, 
                                    (1, 0, shift_x, 0, 1, shift_y), 
                                    resample=Image.BILINEAR
                                )
                                
                                # Resize to target size
                                img_variation = img_variation.resize(size, Image.LANCZOS)
                                
                                # Convert to numpy array
                                img_array = np.array(img_variation) / 255.0
                                
                                # Add noise
                                noise = np.random.normal(0, 0.01, img_array.shape)
                                img_array = np.clip(img_array + noise, 0, 1)
                                
                                # Save image to array
                                images.append(img_array)
                                labels.append(char_label)
                                
                                # Optionally save to disk for debugging
                                # img_variation.save(temp_dir / f"{char_label}_{i}.png")
                        except Exception as e:
                            print(f"Error with font size {font_size}: {e}")
                            continue
                            
                except Exception as e:
                    print(f"Error processing {char} with font {font_path}: {e}")
    except Exception as e:
        print(f"Standard method failed: {e}")
    
    # If no images were generated, try the fallback method
    if not images:
        print("Standard method failed to generate images. Trying fallback method...")
        try:
            for char_label, char in characters.items():
                print(f"Processing character (fallback): {char} ({char_label})")
                
                # Create simple images with text centered
                for _ in range(variations_per_font * len(font_paths)):
                    # Create a blank image
                    img = Image.new('L', size, color=0)
                    draw = ImageDraw.Draw(img)
                    
                    # Draw the character in the center
                    # Use a simple approach without font metrics
                    draw.text((size[0]//2 - 10, size[1]//2 - 10), char, fill=255)
                    
                    # Convert to numpy array
                    img_array = np.array(img) / 255.0
                    
                    # Add noise for variation
                    noise = np.random.normal(0, 0.05, img_array.shape)
                    img_array = np.clip(img_array + noise, 0, 1)
                    
                    # Save image to array
                    images.append(img_array)
                    labels.append(char_label)
        except Exception as e:
            print(f"Fallback method also failed: {e}")
    
    # Convert to numpy arrays
    if not images:
        raise ValueError("No images were generated. Check font paths and character compatibility.")
    
    images_array = np.array(images)
    labels_array = np.array(labels)
    
    print(f"Generated {len(images_array)} synthetic images")
    return images_array, labels_array

def find_system_fonts():
    """Find available font files for character generation."""
    font_paths = []
    
    # Project fonts directory (primary source)
    project_font_dirs = [
        "./fonts/",  # Root fonts directory
        "./fonts/korean/",  # Language-specific subdirectories
        "./fonts/japanese/",
        "./fonts/chinese/",
        "./fonts/greek/",
    ]
    
    # Check project fonts directory first
    for font_dir in project_font_dirs:
        if os.path.exists(font_dir):
            print(f"Checking for fonts in: {font_dir}")
            for file in os.listdir(font_dir):
                if file.lower().endswith(('.ttf', '.otf')):
                    font_path = os.path.join(font_dir, file)
                    font_paths.append(font_path)
                    print(f"Found font: {file}")
    
    # If we found fonts in the project directory, return them
    if font_paths:
        print(f"Found {len(font_paths)} fonts in the project fonts directory")
        return font_paths
    
    # Fallback to system fonts if no project fonts found
    print("No fonts found in project directory, checking system fonts...")
    system_font_dirs = [
        # macOS
        "/Library/Fonts/",
        "/System/Library/Fonts/",
        os.path.expanduser("~/Library/Fonts/"),
        
        # Windows
        "C:\\Windows\\Fonts\\",
        
        # Linux
        "/usr/share/fonts/",
        "/usr/local/share/fonts/",
        os.path.expanduser("~/.fonts/"),
        
        # Downloads folder (cross-platform)
        os.path.expanduser("~/Downloads/"),
    ]
    
    # Check each system directory as fallback
    for font_dir in system_font_dirs:
        if os.path.exists(font_dir):
            for root, _, files in os.walk(font_dir):
                for file in files:
                    if file.lower().endswith(('.ttf', '.otf')):
                        font_paths.append(os.path.join(root, file))
    
    print(f"Found {len(font_paths)} system fonts as fallback")
    return font_paths

# Language-specific character mappings
KOREAN_CHARS = {
    'GA': 'ㄱ', 'NA': 'ㄴ', 'DA': 'ㄷ', 'RA': 'ㄹ', 'MA': 'ㅁ',
    'BA': 'ㅂ', 'SA': 'ㅅ', 'A': 'ㅇ', 'JA': 'ㅈ', 'CHA': 'ㅊ',
    'KA': 'ㅋ', 'TA': 'ㅌ', 'PA': 'ㅍ', 'HA': 'ㅎ',
    # Basic vowels
    'AH': 'ㅏ', 'AE': 'ㅐ', 'YA': 'ㅑ', 'YAE': 'ㅒ', 'EO': 'ㅓ',
    'E': 'ㅔ', 'YEO': 'ㅕ', 'YE': 'ㅖ', 'O': 'ㅗ', 'WA': 'ㅘ',
    'WAE': 'ㅙ', 'OE': 'ㅚ', 'YO': 'ㅛ', 'U': 'ㅜ', 'WO': 'ㅝ',
    'WE': 'ㅞ', 'WI': 'ㅟ', 'YU': 'ㅠ', 'EU': 'ㅡ', 'YI': 'ㅢ',
    'I': 'ㅣ'
}

# Basic set of simplified Chinese characters (most common)
CHINESE_CHARS = {
    'YI': '一', 'ER': '二', 'SAN': '三', 'SI': '四', 'WU': '五',
    'LIU': '六', 'QI': '七', 'BA': '八', 'JIU': '九', 'SHI': '十',
    'REN': '人', 'RI': '日', 'YUE': '月', 'SHUI': '水', 'HUO': '火',
    'MU': '木', 'JIN': '金', 'TU': '土', 'SHAN': '山', 'DA': '大',
    'XIAO': '小', 'ZHONG': '中', 'WANG': '王', 'TIAN': '天', 'DI': '地'
}

# English characters (uppercase and lowercase)
ENGLISH_CHARS_UPPER = {
    'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F',
    'G': 'G', 'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L',
    'M': 'M', 'N': 'N', 'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R',
    'S': 'S', 'T': 'T', 'U': 'U', 'V': 'V', 'W': 'W', 'X': 'X',
    'Y': 'Y', 'Z': 'Z'
}

ENGLISH_CHARS_LOWER = {
    'a': 'a', 'b': 'b', 'c': 'c', 'd': 'd', 'e': 'e', 'f': 'f',
    'g': 'g', 'h': 'h', 'i': 'i', 'j': 'j', 'k': 'k', 'l': 'l',
    'm': 'm', 'n': 'n', 'o': 'o', 'p': 'p', 'q': 'q', 'r': 'r',
    's': 's', 't': 't', 'u': 'u', 'v': 'v', 'w': 'w', 'x': 'x',
    'y': 'y', 'z': 'z'
}

# Greek characters (uppercase and lowercase)
GREEK_CHARS_UPPER = {
    'ALPHA': 'Α', 'BETA': 'Β', 'GAMMA': 'Γ', 'DELTA': 'Δ', 'EPSILON': 'Ε',
    'ZETA': 'Ζ', 'ETA': 'Η', 'THETA': 'Θ', 'IOTA': 'Ι', 'KAPPA': 'Κ',
    'LAMBDA': 'Λ', 'MU': 'Μ', 'NU': 'Ν', 'XI': 'Ξ', 'OMICRON': 'Ο',
    'PI': 'Π', 'RHO': 'Ρ', 'SIGMA': 'Σ', 'TAU': 'Τ', 'UPSILON': 'Υ',
    'PHI': 'Φ', 'CHI': 'Χ', 'PSI': 'Ψ', 'OMEGA': 'Ω'
}

GREEK_CHARS_LOWER = {
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
    'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω'
}

def filter_fonts_for_language(font_paths, language):
    """Filter fonts that support a specific language."""
    supported_fonts = []
    
    # Test characters for each language
    test_chars = {
        'kr': 'ㄱㅏ',
        'zh': '中文',
        'en': 'ABCabc',
        'en_upper': 'ABCDEFG',
        'en_lower': 'abcdefg',
        'jp': 'あいう',
        'gr': 'αβγΑΒΓ',
        'gr_upper': 'ΑΒΓΔΕ',
        'gr_lower': 'αβγδε'
    }
    
    if language not in test_chars:
        print(f"Warning: Language '{language}' not recognized. Using all fonts.")
        return font_paths  # Return all fonts if language not recognized
    
    test_char = test_chars[language]
    print(f"Testing {len(font_paths)} fonts for {language} support using test characters: {test_char}")
    
    for font_path in font_paths:
        try:
            # Try to create a test image with this font
            img = Image.new('L', (50, 50), color=0)
            draw = ImageDraw.Draw(img)
            font = ImageFont.truetype(font_path, 24)
            
            # Test if the font can render the character
            # Just creating the font isn't enough - we need to try to render text
            try:
                # For older Pillow versions
                draw.textsize(test_char, font=font)
            except AttributeError:
                try:
                    # For newer Pillow versions (>=8.0.0)
                    font.getsize(test_char)
                except AttributeError:
                    # For Pillow 9.0.0+
                    font.getbbox(test_char)
            
            # If we get here without an exception, try to actually draw the text
            draw.text((10, 10), test_char, fill=255, font=font)
            
            # If we get here, the font supports the language
            font_name = os.path.basename(font_path)
            print(f"Font supports {language}: {font_name}")
            supported_fonts.append(font_path)
        except Exception as e:
            if 'verbose' in os.environ and os.environ['verbose'].lower() == 'true':
                print(f"Font does not support {language}: {os.path.basename(font_path)}")
                print(f"  Error: {str(e)}")
    
    # If we didn't find any fonts, try to use the project fonts directory
    if not supported_fonts:
        print(f"No system fonts found for {language}. Checking project fonts directory...")
        
        # Map language code to directory name
        lang_dir_map = {
            'kr': 'korean',
            'zh': 'chinese',
            'jp': 'japanese',
            'gr': 'greek',
            'gr_upper': 'greek',
            'gr_lower': 'greek',
            'en': 'english',
            'en_upper': 'english',
            'en_lower': 'english'
        }
        
        # Check if we have a language-specific directory
        if language in lang_dir_map:
            lang_dir = f"./fonts/{lang_dir_map[language]}/"
            if os.path.exists(lang_dir):
                for file in os.listdir(lang_dir):
                    if file.lower().endswith(('.ttf', '.otf')):
                        font_path = os.path.join(lang_dir, file)
                        supported_fonts.append(font_path)
                        print(f"Using project font for {language}: {file}")
    
    print(f"Found {len(supported_fonts)} fonts supporting {language}")
    return supported_fonts 