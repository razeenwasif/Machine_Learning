import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from tensorflow.keras.utils import to_categorical
import seaborn as sns
import numpy as np 
import pandas as pd 
from bidict import bidict
import cv2

mapping = bidict({
    'α': 0,  'β': 1,  'γ': 2,  'δ': 3,  'ε': 4,
    'ζ': 5,  'η': 6,  'θ': 7,  'ι': 8,  'κ': 9,
    'λ': 10, 'μ': 11, 'ν': 12, 'ξ': 13, 'ο': 14,
    'π': 15, 'ρ': 16, 'σ': 17, 'τ': 18, 'υ': 19,
    'φ': 20, 'χ': 21, 'ψ': 22, 'ω': 23
})



model.save('./grCharacters.keras')





















