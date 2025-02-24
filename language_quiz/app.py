from flask import Flask, render_template, request, redirect, session
from random import choice
from bidict import bidict
import numpy as np
from tensorflow import keras

# -------------------------------------------------------------------------
# 1. English mapping/model
mappingEn = bidict({
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
    'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
    'Y': 25, 'Z': 26
})
model = keras.models.load_model('./src/enCharacters.keras')
# -------------------------------------------------------------------------
# 2. Greek mapping/model 
mappingGr = bidict({
    'α': 0,  'β': 1,  'γ': 2,  'δ': 3,  'ε': 4,
    'ζ': 5,  'η': 6,  'θ': 7,  'ι': 8,  'κ': 9,
    'λ': 10, 'μ': 11, 'ν': 12, 'ξ': 13, 'ο': 14,
    'π': 15, 'ρ': 16, 'σ': 17, 'τ': 18, 'υ': 19,
    'φ': 20, 'χ': 21, 'ψ': 22, 'ω': 23
})
modelGr = keras.models.load_model('./src/grCharacters.keras')
# -------------------------------------------------------------------------
# 3. Hiragana mapping/model 
mappingJp = 1
modelJp = 1
# -------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = 'language_quiz'

@app.route('/')
def index():
    session.clear()
    return render_template("index.html")

# ==============================================================================

@app.route('/practiceEn', methods=['GET'])
def practiceEn_get():
    character = choice(list(mappingEn.keys()))
    return render_template("practiceEn.html", character=character, correct="")

@app.route('/practiceEn', methods=['POST'])
def practiceEn_post():
    try:
        character = request.form['character']
        print(f"Current character is: {character}")
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')

        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        # get model prediction 
        prediction = np.argmax(model.predict(img), axis=-1)
        prediction = mappingEn.inverse[prediction[0]]
        print(f"prediction is {prediction}")

        correct = 'yes' if prediction == character else 'no'
        character = choice(list(mappingEn.keys()))

        return render_template("practiceEn.html", character=character, correct=correct)

    except ValueError as e:
        print(f"Value Error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500

# ==============================================================================

@app.route('/practiceJp', methods=['GET'])
def practiceJp_get():
    return render_template("practiceJp.html")

@app.route('/practiceJp', methods=['POST'])
def practiceJp_post():
    return render_template("practiceJp.html")

# ==============================================================================

@app.route('/practiceGr', methods=['GET'])
def practiceGr_get():
    character = choice(list(mappingGr.keys()))
    return render_template("practiceGr.html", character=character, correct="")

@app.route('/practiceGr', methods=['POST'])
def practiceGr_post():
    try:
        character = request.form['character']
        print(f"Current character is: {character}")
        
        pixels = request.form['pixels']
        pixels = pixels.split(',')

        img = np.array(pixels).astype(float).reshape(1, 50, 50, 1)

        # get model prediction 
        prediction = np.argmax(modelGr.predict(img), axis=-1)
        prediction = mappingGr.inverse[prediction[0]]
        print(f"prediction is {prediction}")

        correct = 'yes' if prediction == character else 'no'
        character = choice(list(mappingGr.keys()))

        return render_template("practiceGr.html", character=character, correct=correct)

    except ValueError as e:
        print(f"Value Error: {e}")
        return jsonify({'error': str(e)}), 400

    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({'error': 'An unexpected error occurred.'}), 500
 

# ==============================================================================


if __name__ == '__main__':
    app.run(debug=True)
