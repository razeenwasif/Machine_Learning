import numpy as np
from bidict import bidict 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns

mapping = bidict({
    'α': 0,  'β': 1,  'γ': 2,  'δ': 3,  'ε': 4,
    'ζ': 5,  'η': 6,  'θ': 7,  'ι': 8,  'κ': 9,
    'λ': 10, 'μ': 11, 'ν': 12, 'ξ': 13, 'ο': 14,
    'π': 15, 'ρ': 16, 'σ': 17, 'τ': 18, 'υ': 19,
    'φ': 20, 'χ': 21, 'ψ': 22, 'ω': 23
})

labels = np.load("../data/labelsGr.npy")
images = np.load("../data/imagesGr.npy")

# convert strings to int
labels = np.array([mapping[x] for x in labels])

# normalize image data 
images = images.astype("float32") / 255

images = np.expand_dims(images, -1)

labels, images = shuffle(labels, images)
split = 0.75

labels_train = labels[ : int(len(labels) * split)]
labels_test = labels[int(len(labels) * split) : ]

images_train = images[ : int(len(images) * split)]
images_test = images[int(len(images) * split) : ]

batch_size = 16
epochs = 20

model = keras.Sequential([
    keras.Input(shape=(50,50,1)),
    layers.Conv2D(32, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.2),
    layers.Conv2D(64, kernel_size=3, activation='relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(len(mapping), activation='softmax')
])

early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5)
optimizer = keras.optimizers.Adam()

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs,
          validation_data=(images_test, labels_test), callbacks=[early_stopping])

labels_pred = np.argmax(model.predict(images_test), axis=-1)

cm = confusion_matrix(labels_test, labels_pred, labels=list(mapping.inverse.keys()))

plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, cbar=False, cmap="Blues", xticklabels=list(mapping.keys()), 
            yticklabels=list(mapping.keys()))
plt.show()



model.save('./grCharacters.keras')





















