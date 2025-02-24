import numpy as np
from bidict import bidict 
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt 
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns

mapping = bidict({
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6,
    'G': 7, 'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12,
    'M': 13, 'N': 14, 'O': 15, 'P': 16, 'Q': 17, 'R': 18,
    'S': 19, 'T': 20, 'U': 21, 'V': 22, 'W': 23, 'X': 24,
    'Y': 25, 'Z': 26
})

labels = np.load("../data/labels.npy")
images = np.load("../data/images.npy")

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
    layers.Dense(len(mapping)+1, activation='softmax')
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

model.save('./enCharacters.keras')
