from flask import Flask, render_template, request, redirect, session, url_for, jsonify
from random import choice
from bidict import bidict
import numpy as np
from tensorflow import keras
import os
import traceback
from src.data_generator import generate_character_images, find_system_fonts, filter_fonts_for_language
from src.data_generator import KOREAN_CHARS, CHINESE_CHARS, ENGLISH_CHARS_UPPER, ENGLISH_CHARS_LOWER
from src.data_generator import GREEK_CHARS_UPPER, GREEK_CHARS_LOWER

# -------------------------------------------------------------------------
# 1. Greek mapping/model 
mappingGr = bidict({
    'alpha': 0,  'beta': 1,  'gamma': 2,  'delta': 3,  'epsilon': 4,
    'zeta': 5,  'eta': 6,  'theta': 7,  'iota': 8,  'kappa': 9,
    'lambda': 10, 'mu': 11, 'nu': 12, 'xi': 13, 'omicron': 14,
    'pi': 15, 'rho': 16, 'sigma': 17, 'tau': 18, 'upsilon': 19,
    'phi': 20, 'chi': 21, 'psi': 22, 'omega': 23
})
greekSymbolMap = bidict({
    'α': 'alpha',  'β': 'beta',  'γ': 'gamma',  'δ': 'delta',  'ε': 'epsilon',
    'ζ': 'zeta',  'η': 'eta',  'θ': 'theta',  'ι': 'iota',  'κ': 'kappa',
    'λ': 'lambda', 'μ': 'mu', 'ν': 'nu', 'ξ': 'xi', 'ο': 'omicron',
    'π': 'pi', 'ρ': 'rho', 'σ': 'sigma', 'τ': 'tau', 'υ': 'upsilon',
    'φ': 'phi', 'χ': 'chi', 'ψ': 'psi', 'ω': 'omega'
})

#    'Α': 0,  'Β': 1,  'Γ': 2,  'Δ': 3,  'Ε': 4,
#    'Ζ': 5,  'Η': 6,  'Θ': 7,  'Ι': 8,  'Κ': 9,
#    'Λ': 10, 'Μ': 11, 'Ν': 12, 'Ξ': 13, 'Ο': 14,
#    'Π': 15, 'Ρ': 16, 'Σ': 17, 'Τ': 18, 'Υ': 19,
#    'Φ': 20, 'Χ': 21, 'Ψ': 22, 'Ω': 23

modelGr = keras.models.load_model('./src/grCharacters.keras')

# Greek uppercase and lowercase models
try:
    modelGrUpper = keras.models.load_model('./src/grUpperCharacters.keras')
    print("Greek uppercase model loaded successfully")
except:
    modelGrUpper = None
    print("Greek uppercase model not found (will be created when generating data)")

try:
    modelGrLower = keras.models.load_model('./src/grLowerCharacters.keras')
    print("Greek lowercase model loaded successfully")
except:
    modelGrLower = None
    print("Greek lowercase model not found (will be created when generating data)")

# -------------------------------------------------------------------------
# 3. Hiragana mapping/model 
mappingJp = bidict({
    'A': 1, 'I': 2, 'U': 3, 'E': 4, 'O': 5,
    'KA': 6, 'KI': 7, 'KU': 8, 'KE': 9, 'KO': 10,
    'SA': 11, 'SHI': 12, 'SU': 13, 'SE': 14, 'SO': 15,
    'TA': 16, 'CHI': 17, 'TSU': 18, 'TE': 19, 'TO': 20,
    'NA': 21, 'NI': 22, 'NU': 23, 'NE': 24, 'NO': 25,
    'HA': 26, 'HI': 27, 'FU': 28, 'HE': 29, 'HO': 30,
    'MA': 31, 'MI': 32, 'MU': 33, 'ME': 34, 'MO': 35,
    'YA': 36, 'YU': 37, 'YO': 38,
    'RA': 39, 'RI': 40, 'RU': 41, 'RE': 42, 'RO': 43,
    'WA': 44, 'WO': 45, 'N': 46
})
kanaMap = bidict({
    'あ': 'A', 'い': 'I', 'う': 'U', 'え': 'E', 'お': 'O',
    'か': 'KA', 'き': 'KI', 'く': 'KU', 'け': 'KE', 'こ': 'KO',
    'さ': 'SA', 'し': 'SHI', 'す': 'SU', 'せ': 'SE', 'そ': 'SO',
    'た': 'TA', 'ち': 'CHI', 'つ': 'TSU', 'て': 'TE', 'と': 'TO',
    'な': 'NA', 'に': 'NI', 'ぬ': 'NU', 'ね': 'NE', 'の': 'NO',
    'は': 'HA', 'ひ': 'HI', 'ふ': 'FU', 'へ': 'HE', 'ほ': 'HO',
    'ま': 'MA', 'み': 'MI', 'む': 'MU', 'め': 'ME', 'も': 'MO',
    'や': 'YA', 'ゆ': 'YU', 'よ': 'YO', 'ら':
    'RA', 'り': 'RI', 'る': 'RU', 'れ': 'RE', 'ろ': 'RO',
    'わ': 'WA', 'を': 'WO', 'ん': 'N'
})
modelJp = keras.models.load_model('./src/jpCharacters.keras')
# -------------------------------------------------------------------------
# 4. Korean mapping/model
# Using the basic Korean consonants and vowels from KOREAN_CHARS
mappingKr = bidict({
    'GA': 0, 'NA': 1, 'DA': 2, 'RA': 3, 'MA': 4,
    'BA': 5, 'SA': 6, 'A': 7, 'JA': 8, 'CHA': 9,
    'KA': 10, 'TA': 11, 'PA': 12, 'HA': 13,
    'AH': 14, 'AE': 15, 'YA': 16, 'YAE': 17, 'EO': 18,
    'E': 19, 'YEO': 20, 'YE': 21, 'O': 22, 'WA': 23,
    'WAE': 24, 'OE': 25, 'YO': 26, 'U': 27, 'WO': 28,
    'WE': 29, 'WI': 30, 'YU': 31, 'EU': 32, 'YI': 33,
    'I': 34
})

# Korean symbol mapping
koreanSymbolMap = bidict({
    'ㄱ': 'GA', 'ㄴ': 'NA', 'ㄷ': 'DA', 'ㄹ': 'RA', 'ㅁ': 'MA',
    'ㅂ': 'BA', 'ㅅ': 'SA', 'ㅇ': 'A', 'ㅈ': 'JA', 'ㅊ': 'CHA',
    'ㅋ': 'KA', 'ㅌ': 'TA', 'ㅍ': 'PA', 'ㅎ': 'HA',
    'ㅏ': 'AH', 'ㅐ': 'AE', 'ㅑ': 'YA', 'ㅒ': 'YAE', 'ㅓ': 'EO',
    'ㅔ': 'E', 'ㅕ': 'YEO', 'ㅖ': 'YE', 'ㅗ': 'O', 'ㅘ': 'WA',
    'ㅙ': 'WAE', 'ㅚ': 'OE', 'ㅛ': 'YO', 'ㅜ': 'U', 'ㅝ': 'WO',
    'ㅞ': 'WE', 'ㅟ': 'WI', 'ㅠ': 'YU', 'ㅡ': 'EU', 'ㅢ': 'YI',
    'ㅣ': 'I'
})

# Try to load the model if it exists, otherwise it will be created when synthetic data is generated
try:
    modelKr = keras.models.load_model('./src/krCharacters.keras')
    print("Korean model loaded successfully")
except Exception as e:
    print(f"Korean model not found or could not be loaded: {e}")
    print("You can generate synthetic data for Korean using the Generate Data page")
    modelKr = None
# -------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = 'language_quiz'

@app.route('/')
def index():
    session.clear()
    return render_template("index.html")

# ==============================================================================

@app.route('/practiceJp', methods=['GET'])
def practiceJp_get():
    character = choice(list(mappingJp.keys()))
    return render_template("practiceJp.html", character=character, correct="")

@app.route('/practiceJp', methods=['POST'])
def practiceJp_post():
    try:
        character = request.form['character']
        print(f"Current character is: {character}")
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')

        # Ensure we have the right number of pixels (50x50=2500)
        if len(pixels) != 2500:
            print(f"Expected 2500 pixels, got {len(pixels)}")
            return render_template("error.html", error=f"Invalid pixel data: expected 2500 pixels, got {len(pixels)}")

        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        # get model prediction
        prediction = np.argmax(modelJp.predict(img), axis=-1)
        prediction = mappingJp.inverse[prediction[0]]
        print(f"prediction is {prediction}")
        
        correct = 'yes' if prediction == character else 'no'
        kana = kanaMap.inverse[character]
        character = choice(list(mappingJp.keys()))

        return render_template("practiceJp.html", character=character, kana=kana, correct=correct)
        
    except ValueError as e:
        print(f"Value Error: {e}")
        return render_template("error.html", error=f"Value Error: {str(e)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return render_template("error.html", error="An unexpected error occurred.")

# ==============================================================================

@app.route('/practiceGr', methods=['GET'])
def practiceGr_get():
    character = choice(list(mappingGr.keys()))
    symbol = greekSymbolMap.inverse.get(character, None)
    return render_template("practiceGr.html", character=character, correct="", symbol=symbol)

@app.route('/practiceGr', methods=['POST'])
def practiceGr_post():
    try:
        character = request.form['character']
        print(f"Current character is: {character}")
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')

        # Ensure we have the right number of pixels (50x50=2500)
        if len(pixels) != 2500:
            print(f"Expected 2500 pixels, got {len(pixels)}")
            return render_template("error.html", error=f"Invalid pixel data: expected 2500 pixels, got {len(pixels)}")

        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        # get model prediction 
        prediction = np.argmax(modelGr.predict(img), axis=-1)
        prediction = mappingGr.inverse[prediction[0]]
        print(f"prediction is {prediction}")

        correct = 'yes' if prediction == character else 'no'
        symbol = greekSymbolMap.inverse[character]
        character = choice(list(mappingGr.keys()))

        return render_template("practiceGr.html", character=character, correct=correct, symbol=symbol)

    except ValueError as e:
        print(f"Value Error: {e}")
        return render_template("error.html", error=f"Value Error: {str(e)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return render_template("error.html", error="An unexpected error occurred.")
 

# ==============================================================================

@app.route('/add-data', methods=['GET'])
def add_data_get():
    message = session.get('message', '')
    selected_language = request.args.get('language', 'jp')  # Default to Japanese if not specified
    
    # Define language configurations
    language_configs = {
        'en': {
            'name': 'English',
            'mapping': mappingEn,
            'labels_file': './data/labels.npy',
            'images_file': './data/images.npy'
        },
        'jp': {
            'name': 'Japanese',
            'mapping': mappingJp,
            'labels_file': './data/labelsJp.npy',
            'images_file': './data/imagesJp.npy'
        },
        'gr': {
            'name': 'Greek',
            'mapping': mappingGr,
            'labels_file': './data/labelsGr.npy',
            'images_file': './data/imagesGr.npy'
        },
        'kr': {
            'name': 'Korean',
            'mapping': mappingKr,
            'labels_file': './data/labelsKr.npy',
            'images_file': './data/imagesKr.npy'
        }
    }
    
    # Validate selected language
    if selected_language not in language_configs:
        selected_language = 'jp'  # Default to Japanese if invalid
    
    config = language_configs[selected_language]
    
    # Load labels and find character with fewest samples
    try:
        labels = np.load(config['labels_file'])
        count = {k: 0 for k in config['mapping'].keys()}
        for label in labels:
            if label in count:  # Ensure the label exists in our mapping
                count[label] += 1
        count = sorted(count.items(), key=lambda x: x[1])
        character = count[0][0]
    except (FileNotFoundError, IndexError, ValueError) as e:
        print(f"Error loading labels or finding character: {e}")
        character = choice(list(config['mapping'].keys()))
    
    # Get symbol for display if applicable
    symbol = None
    if selected_language == 'jp' and character in kanaMap.inverse:
        symbol = kanaMap.inverse[character]
    elif selected_language == 'gr':
        # Use get() with a default value to handle potential missing keys
        symbol = greekSymbolMap.inverse.get(character, None)
        if not symbol and character in greekSymbolMap.values():
            # If the character is a value in greekSymbolMap, find its key
            for key, value in greekSymbolMap.items():
                if value == character:
                    symbol = key
                    break
    
    return render_template(
        "addData.html", 
        character=character, 
        message=message, 
        language=selected_language,
        language_name=config['name'],
        languages=language_configs,
        symbol=symbol
    )

@app.route('/add-data', methods=['POST'])
def add_data_post():
    try:
        label = request.form['character']
        language = request.form.get('language', 'jp')  # Default to Japanese if not specified
        
        # Define language configurations
        language_configs = {
            'en': {
                'labels_file': './data/labels.npy',
                'images_file': './data/images.npy'
            },
            'jp': {
                'labels_file': './data/labelsJp.npy',
                'images_file': './data/imagesJp.npy'
            },
            'gr': {
                'labels_file': './data/labelsGr.npy',
                'images_file': './data/imagesGr.npy'
            },
            'kr': {
                'labels_file': './data/labelsKr.npy',
                'images_file': './data/imagesKr.npy'
            }
        }
        
        # Validate selected language
        if language not in language_configs:
            return render_template("error.html", error=f"Invalid language selection: {language}")
        
        config = language_configs[language]
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')
        
        # Ensure we have the right number of pixels (50x50=2500)
        if len(pixels) != 2500:
            print(f"Expected 2500 pixels, got {len(pixels)}")
            return render_template("error.html", error=f"Invalid pixel data: expected 2500 pixels, got {len(pixels)}")
        
        # Load existing data or create new files if they don't exist
        try:
            labels = np.load(config['labels_file'])
        except FileNotFoundError:
            # Create a new empty labels array if the file doesn't exist
            print(f"Creating new labels file for {language}")
            labels = np.array([], dtype=object)
        
        labels = np.append(labels, label)
        np.save(config['labels_file'], labels)

        img = np.array(pixels).astype(float).reshape(1,50,50)
        
        try:
            imgs = np.load(config['images_file'])
            imgs = np.vstack([imgs, img])
        except FileNotFoundError:
            # Create a new images array if the file doesn't exist
            print(f"Creating new images file for {language}")
            imgs = img
        
        np.save(config['images_file'], imgs)

        session['message'] = f'"{label}" added to {language.upper()} training dataset'
        
        return redirect(url_for('add_data_get', language=language))
        
    except ValueError as e:
        print(f"Value Error: {e}")
        return render_template("error.html", error=f"Value Error: {str(e)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return render_template("error.html", error="An unexpected error occurred.")

# ==============================================================================
# Synthetic Data Generation

@app.route('/generate-data', methods=['GET'])
def generate_data_get():
    languages = {
        'kr': {
            'name': 'Korean',
            'characters': KOREAN_CHARS
        },
        'zh': {
            'name': 'Chinese',
            'characters': CHINESE_CHARS
        },
        'gr_upper': {
            'name': 'Greek (Uppercase)',
            'characters': GREEK_CHARS_UPPER
        },
        'gr_lower': {
            'name': 'Greek (Lowercase)',
            'characters': GREEK_CHARS_LOWER
        }
    }
    
    return render_template(
        "generateData.html", 
        languages=languages,
        selected_language=None,
        message=None
    )

@app.route('/generate-data', methods=['POST'])
def generate_data_post():
    try:
        language = request.form.get('language')
        num_variations = int(request.form.get('variations', '10'))
        verbose = request.form.get('verbose', 'false').lower() == 'true'
        
        # Set verbose environment variable for detailed logging
        if verbose:
            os.environ['verbose'] = 'true'
        
        # Define language configurations
        language_configs = {
            'kr': {
                'name': 'Korean',
                'characters': KOREAN_CHARS,
                'labels_file': './data/labelsKr.npy',
                'images_file': './data/imagesKr.npy',
                'mapping_name': 'mappingKr',
                'font_dir': './fonts/korean/',
                'model_path': './src/krCharacters.keras',
                'training_script': './src/mainKr.py'
            },
            'zh': {
                'name': 'Chinese',
                'characters': CHINESE_CHARS,
                'labels_file': './data/labelsZh.npy',
                'images_file': './data/imagesZh.npy',
                'mapping_name': 'mappingZh',
                'font_dir': './fonts/chinese/',
                'model_path': './src/zhCharacters.keras'
            },
            'gr_upper': {
                'name': 'Greek (Uppercase)',
                'characters': GREEK_CHARS_UPPER,
                'labels_file': './data/labelsGrUpper.npy',
                'images_file': './data/imagesGrUpper.npy',
                'mapping_name': 'mappingGrUpper',
                'font_dir': './fonts/greek/',
                'model_path': './src/grUpperCharacters.keras',
                'training_script': './src/mainGrUpper.py'
            },
            'gr_lower': {
                'name': 'Greek (Lowercase)',
                'characters': GREEK_CHARS_LOWER,
                'labels_file': './data/labelsGrLower.npy',
                'images_file': './data/imagesGrLower.npy',
                'mapping_name': 'mappingGrLower',
                'font_dir': './fonts/greek/',
                'model_path': './src/grLowerCharacters.keras',
                'training_script': './src/mainGrLower.py'
            }
        }
        
        if language not in language_configs:
            return render_template(
                "error.html", 
                error=f"Unsupported language: {language}"
            )
        
        config = language_configs[language]
        
        # Ensure font directory exists
        font_dir = config['font_dir']
        if not os.path.exists(font_dir):
            os.makedirs(font_dir, exist_ok=True)
            print(f"Created font directory: {font_dir}")
        
        # Find fonts that support this language
        print(f"Searching for fonts for {config['name']}...")
        all_fonts = find_system_fonts()
        
        if not all_fonts:
            return render_template(
                "error.html", 
                error=f"No fonts found. Please add .ttf or .otf font files to the '{font_dir}' directory."
            )
        
        # Filter fonts that support the selected language
        supported_fonts = filter_fonts_for_language(all_fonts, language)
        print(f"Found {len(supported_fonts)} fonts supporting {config['name']}")
        
        if not supported_fonts:
            return render_template(
                "error.html", 
                error=f"No fonts found that support {config['name']}. Please add appropriate fonts to the '{font_dir}' directory."
            )
        
        # Generate synthetic data
        print(f"Generating synthetic data for {config['name']} with {num_variations} variations per font...")
        images, labels = generate_character_images(
            config['characters'],
            supported_fonts,
            variations_per_font=num_variations
        )
        
        # Reshape images for saving
        images_reshaped = images.reshape(-1, 50, 50)
        
        # Create data directory if it doesn't exist
        os.makedirs('./data', exist_ok=True)
        
        # Save the data
        np.save(config['labels_file'], labels)
        np.save(config['images_file'], images_reshaped)
        
        # Model information
        model_info = {}
        
        # Determine which model variable to use
        model_var_map = {
            'kr': 'modelKr',
            'gr_upper': 'modelGrUpper',
            'gr_lower': 'modelGrLower'
        }
        
        # Check if we have a training script for this language
        if 'training_script' in config and os.path.exists(config['training_script']):
            print(f"A dedicated training script exists for {config['name']} at {config['training_script']}")
            print(f"To train the model, run: python {config['training_script']}")
            
            # Note: We don't automatically run the training script here as it might be resource-intensive
            # and the user might want to run it separately
            
            model_info = {
                'message': f"Data generated successfully. To train the model, run: python {config['training_script']}",
                'num_samples': len(images)
            }
        else:
            # For languages without dedicated training scripts
            model_var_name = model_var_map.get(language)
            if model_var_name:
                current_model = globals().get(model_var_name)
                
                if current_model is None:
                    print(f"No model found for {config['name']}. Please create one using a dedicated training script.")
                    model_info = {
                        'message': f"No model found for {config['name']}. Please create one using a dedicated training script.",
                        'num_samples': len(images)
                    }
                else:
                    print(f"Using existing model for {config['name']}")
                    model_info = {
                        'message': f"Using existing model for {config['name']}",
                        'num_samples': len(images)
                    }
        
        message = f"Successfully generated {len(images)} synthetic samples for {config['name']}."
        
        # Reset verbose environment variable
        if 'verbose' in os.environ:
            del os.environ['verbose']
        
        # Get font directory info for display
        font_dir_info = {
            'path': font_dir,
            'exists': os.path.exists(font_dir),
            'num_fonts': len([f for f in os.listdir(font_dir) if f.lower().endswith(('.ttf', '.otf'))]) if os.path.exists(font_dir) else 0
        }
        
        # Get font names for display (limit to 5)
        font_names = [os.path.basename(f) for f in supported_fonts[:5]]
        
        # Render the template with the results
        return render_template(
            "generateData.html",
            languages=language_configs,
            selected_language=language,
            message=message,
            num_samples=len(images),
            num_fonts=len(supported_fonts),
            font_names=font_names,
            font_dir_info=font_dir_info,
            model_info=model_info
        )
    
    except Exception as e:
        traceback.print_exc()
        return render_template("error.html", error=str(e))

@app.route('/practiceKr', methods=['GET'])
def practiceKr_get():
    character = choice(list(mappingKr.keys()))
    symbol = koreanSymbolMap.inverse.get(character, None)
    return render_template("practiceKr.html", character=character, correct="", symbol=symbol)

@app.route('/practiceKr', methods=['POST'])
def practiceKr_post():
    try:
        character = request.form['character']
        print(f"Current character is: {character}")
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')
        
        # Ensure we have the right number of pixels (50x50=2500)
        if len(pixels) != 2500:
            print(f"Expected 2500 pixels, got {len(pixels)}")
            return render_template("error.html", error=f"Invalid pixel data: expected 2500 pixels, got {len(pixels)}")
        
        # Check if model exists
        if modelKr is None:
            return render_template("error.html", error="Korean model not found. Please generate synthetic data first.")

        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        # get model prediction 
        prediction = np.argmax(modelKr.predict(img), axis=-1)
        prediction = mappingKr.inverse[prediction[0]]
        print(f"prediction is {prediction}")

        correct = 'yes' if prediction == character else 'no'
        symbol = koreanSymbolMap.inverse[character]
        character = choice(list(mappingKr.keys()))

        return render_template("practiceKr.html", character=character, correct=correct, symbol=symbol)

    except ValueError as e:
        print(f"Value Error: {e}")
        return render_template("error.html", error=f"Value Error: {str(e)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return render_template("error.html", error="An unexpected error occurred.")

# Add new combined practice routes for Greek (uppercase and lowercase)
@app.route('/practiceGreek', methods=['GET'])
def practiceGreek_get():
    # Randomly choose between uppercase and lowercase
    case = choice(['upper', 'lower'])
    
    if case == 'upper':
        # Use uppercase characters
        character = choice(list(GREEK_CHARS_UPPER.keys()))
        symbol = GREEK_CHARS_UPPER[character]
        case_name = "Uppercase"
        model_to_use = "upper"
    else:
        # Use lowercase characters
        character = choice(list(GREEK_CHARS_LOWER.keys()))
        symbol = GREEK_CHARS_LOWER[character]
        case_name = "Lowercase"
        model_to_use = "lower"
    
    return render_template(
        "practiceGreek.html", 
        character=character, 
        symbol=symbol,
        case=case_name,
        model=model_to_use,
        correct=""
    )

@app.route('/practiceGreek', methods=['POST'])
def practiceGreek_post():
    try:
        character = request.form['character']
        model_to_use = request.form['model']
        print(f"Current character is: {character}, using model: {model_to_use}")
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')

        # Ensure we have the right number of pixels (50x50=2500)
        if len(pixels) != 2500:
            print(f"Expected 2500 pixels, got {len(pixels)}")
            return render_template("error.html", error=f"Invalid pixel data: expected 2500 pixels, got {len(pixels)}")

        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        # Choose the appropriate model based on case
        if model_to_use == "upper":
            if modelGrUpper is None:
                return render_template("error.html", error="Greek uppercase model not found. Please generate synthetic data first.")
            prediction = np.argmax(modelGrUpper.predict(img), axis=-1)
            # The model was trained with indices directly corresponding to the mapping
            prediction_char = list(GREEK_CHARS_UPPER.keys())[prediction[0]]
            chars_to_choose_from = list(GREEK_CHARS_UPPER.keys())
            case_name = "Uppercase"
        else:  # lower
            if modelGrLower is None:
                return render_template("error.html", error="Greek lowercase model not found. Please generate synthetic data first.")
            prediction = np.argmax(modelGrLower.predict(img), axis=-1)
            # The model was trained with indices directly corresponding to the mapping
            prediction_char = list(GREEK_CHARS_LOWER.keys())[prediction[0]]
            chars_to_choose_from = list(GREEK_CHARS_LOWER.keys())
            case_name = "Lowercase"
        
        print(f"prediction is {prediction_char}")

        correct = 'yes' if prediction_char == character else 'no'
        
        # Choose a new character, randomly selecting case again
        new_case = choice(['upper', 'lower'])
        if new_case == 'upper':
            new_character = choice(list(GREEK_CHARS_UPPER.keys()))
            new_symbol = GREEK_CHARS_UPPER[new_character]
            new_case_name = "Uppercase"
            new_model = "upper"
        else:
            new_character = choice(list(GREEK_CHARS_LOWER.keys()))
            new_symbol = GREEK_CHARS_LOWER[new_character]
            new_case_name = "Lowercase"
            new_model = "lower"

        return render_template(
            "practiceGreek.html", 
            character=new_character, 
            symbol=new_symbol,
            case=new_case_name,
            model=new_model,
            correct=correct,
            previous_character=character,
            previous_prediction=prediction_char
        )

    except ValueError as e:
        print(f"Value Error: {e}")
        traceback.print_exc()
        return render_template("error.html", error=f"Value Error: {str(e)}")

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        traceback.print_exc()
        return render_template("error.html", error=f"An unexpected error occurred: {str(e)}")

# Add a new route for the alphabets reference page
@app.route('/alphabets', methods=['GET'])
def alphabets():
    # Prepare data for the template
    greek_upper = GREEK_CHARS_UPPER
    greek_lower = GREEK_CHARS_LOWER
    korean = {k: v for k, v in koreanSymbolMap.inverse.items()}
    japanese = kanaMap
    chinese = CHINESE_CHARS
    
    return render_template(
        "alphabets.html",
        greek_upper=greek_upper,
        greek_lower=greek_lower,
        korean=korean,
        japanese=japanese,
        chinese=chinese
    )

if __name__ == '__main__':
    app.run(debug=True)
