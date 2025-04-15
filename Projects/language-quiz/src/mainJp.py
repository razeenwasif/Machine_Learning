import numpy as np 
from bidict import bidict
import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils import shuffle 
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns

mapping = bidict({
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

labels = np.load("../data/labelsJp.npy")
labels = np.array([mapping[x] for x in labels])

images = np.load("../data/imagesJp.npy")
images = images.astype("float32") / 255
images = np.expand_dims(images, -1)

labels, images = shuffle(labels, images)
split = 0.70

labels_train = labels[ : int(len(labels)*split)]
labels_test = labels[int(len(labels)*split) : ]

images_train = images[ : int(len(images)*split)]
images_test = images[int(len(images)*split) : ]

batch_size, epochs = 32, 20 

model = keras.Sequential([
    keras.Input(shape=(50, 50, 1)),
    layers.Conv2D(256, kernel_size=5, activation="relu", padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.3),
    layers.Conv2D(512, kernel_size=5, activation="relu", padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.3),
    layers.Conv2D(1024, kernel_size=5, activation="relu", padding='same'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.3),
    layers.Flatten(),
    layers.Dense(len(mapping)+1, activation="softmax")
])

early_stopping = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=3)
optimizer = keras.optimizers.Adam()

model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

model.fit(images_train, labels_train, batch_size=batch_size, epochs=epochs, 
          validation_data=(images_test, labels_test), callbacks=[early_stopping])

prediction = np.argmax(model.predict(images_test), axis=-1)

cm = confusion_matrix(labels_test, prediction, labels=list(mapping.inverse.keys()))

plt.figure(figsize=(12,12))
sns.heatmap(cm, annot=True, cbar=False, xticklabels=list(mapping.keys()), yticklabels=list(mapping.keys()),)
plt.xticks(rotation=0)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.show()

model.save("jpCharacters.keras")
