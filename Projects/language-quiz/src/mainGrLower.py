import numpy as np
from bidict import bidict
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
import seaborn as sns
import math
import tensorflow as tf
import random

# Greek lowercase mapping
mapping = bidict({
    'alpha': 0, 'beta': 1, 'gamma': 2, 'delta': 3, 'epsilon': 4,
    'zeta': 5, 'eta': 6, 'theta': 7, 'iota': 8, 'kappa': 9,
    'lambda': 10, 'mu': 11, 'nu': 12, 'xi': 13, 'omicron': 14,
    'pi': 15, 'rho': 16, 'sigma': 17, 'tau': 18, 'upsilon': 19,
    'phi': 20, 'chi': 21, 'psi': 22, 'omega': 23
})

# Greek symbol mapping for reference
greek_symbol_map = bidict({
    'alpha': 'α', 'beta': 'β', 'gamma': 'γ', 'delta': 'δ', 'epsilon': 'ε',
    'zeta': 'ζ', 'eta': 'η', 'theta': 'θ', 'iota': 'ι', 'kappa': 'κ',
    'lambda': 'λ', 'mu': 'μ', 'nu': 'ν', 'xi': 'ξ', 'omicron': 'ο',
    'pi': 'π', 'rho': 'ρ', 'sigma': 'σ', 'tau': 'τ', 'upsilon': 'υ',
    'phi': 'φ', 'chi': 'χ', 'psi': 'ψ', 'omega': 'ω'
})

def visualize_training_images(images, labels, mapping, symbol_map=None, num_examples=24):
    """
    Visualize training images for each class
    
    Args:
        images: Array of training images
        labels: Array of labels (as integers)
        mapping: Bidirectional dictionary mapping label names to indices
        symbol_map: Optional mapping from label names to symbols
        num_examples: Number of examples to show (default: one per class)
    """
    # Determine grid size based on number of classes
    num_classes = len(mapping)
    grid_size = math.ceil(math.sqrt(num_classes))
    
    plt.figure(figsize=(15, 15))
    plt.subplots_adjust(hspace=0.5, wspace=0.5)
    
    # Create a dictionary to track examples for each class
    class_examples = {}
    
    # First, collect examples for each class
    for i in range(len(images)):
        label_idx = labels[i]
        if label_idx not in class_examples:
            class_examples[label_idx] = i
            
        # If we have found an example for each class, we can stop
        if len(class_examples) == num_classes:
            break
    
    # Check if we found all classes
    if len(class_examples) < num_classes:
        print(f"Warning: Only found examples for {len(class_examples)} out of {num_classes} classes")
        print(f"Missing classes: {set(range(num_classes)) - set(class_examples.keys())}")
    
    # Now plot one example for each class we found
    for idx, (label_idx, img_idx) in enumerate(sorted(class_examples.items())):
        label_name = mapping.inverse[label_idx]
        
        # Get the corresponding symbol if symbol_map is provided
        symbol = None
        if symbol_map is not None:
            symbol = symbol_map.get(label_name, "")
        
        # Create subplot
        plt.subplot(grid_size, grid_size, idx + 1)
        
        # Display the image
        plt.imshow(images[img_idx].reshape(50, 50), cmap='gray')
        
        # Set title with label name and symbol
        title = f"{label_name}"
        if symbol:
            title += f" ({symbol})"
        plt.title(title)
        
        plt.axis('off')
    
    plt.suptitle("Greek Lowercase Character Training Images", fontsize=16)
    plt.tight_layout()
    plt.savefig('../static/gr_lower_training_samples.png')
    plt.show()

# Load the data
labels = np.load("../data/labelsGrLower.npy")
images = np.load("../data/imagesGrLower.npy")

# Print some statistics about the data
print(f"Loaded {len(images)} images and {len(labels)} labels")
print(f"Label distribution:")
unique_labels, counts = np.unique(labels, return_counts=True)
for label, count in zip(unique_labels, counts):
    print(f"  {label}: {count}")

# Convert string labels to integers using the mapping
labels = np.array([mapping[x] for x in labels])

# Check the distribution of integer labels
print("\nInteger label distribution:")
unique_int_labels, counts = np.unique(labels, return_counts=True)
class_counts = {}
for label, count in zip(unique_int_labels, counts):
    class_name = mapping.inverse[label]
    class_counts[label] = count
    print(f"  {label} ({class_name}): {count}")

# Check for class imbalance
min_count = min(counts)
max_count = max(counts)
print(f"\nClass imbalance ratio (max/min): {max_count/min_count:.2f}")

# Normalize image data
images = images.astype("float32") / 255
images = np.expand_dims(images, -1)  # Add channel dimension

# Visualize some training images before shuffling
print("\nVisualizing Greek lowercase training images...")
visualize_training_images(images, labels, mapping, greek_symbol_map)

# Shuffle and split the data
labels, images = shuffle(labels, images)
split = 0.75  # 75% training, 25% testing

labels_train = labels[:int(len(labels) * split)]
labels_test = labels[int(len(labels) * split):]

images_train = images[:int(len(images) * split)]
images_test = images[int(len(images) * split):]

# Calculate class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(labels_train),
    y=labels_train
)
class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
print("\nClass weights to handle imbalance:")
for class_idx, weight in class_weight_dict.items():
    print(f"  {mapping.inverse[class_idx]}: {weight:.4f}")

# Define data augmentation
data_augmentation = keras.Sequential([
    layers.RandomRotation(0.2),  # Increased rotation
    layers.RandomTranslation(height_factor=0.15, width_factor=0.15),  # Increased translation
    layers.RandomZoom(0.15),  # Increased zoom
])

# Define training parameters
batch_size = 16  # Smaller batch size
epochs = 50  # More epochs

# Create a model with stronger regularization
model = keras.Sequential([
    # Data augmentation layers (only applied during training)
    keras.Input(shape=(50, 50, 1)),
    layers.Rescaling(1./1.),  # Identity scaling, just to ensure proper input format
    
    # First convolutional block
    layers.Conv2D(64, kernel_size=3, activation=None, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.3),  # Added dropout
    
    # Second convolutional block
    layers.Conv2D(128, kernel_size=3, activation=None, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.3),  # Added dropout
    
    # Third convolutional block
    layers.Conv2D(256, kernel_size=3, activation=None, padding='same', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.MaxPooling2D(pool_size=2),
    layers.Dropout(0.3),  # Added dropout
    
    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(512, activation=None, kernel_regularizer=keras.regularizers.l2(0.001)),  # Increased size
    layers.BatchNormalization(),
    layers.Activation('relu'),
    layers.Dropout(0.5),
    
    # Output layer
    layers.Dense(len(mapping), activation=None),  # No activation here
    layers.BatchNormalization(),
    layers.Activation('softmax')  # Softmax after batch norm
])

# Print model summary
model.summary()

# Define callbacks and optimizer
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_accuracy", 
    patience=15,  # Increased patience
    restore_best_weights=True
)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.1,  # More aggressive reduction
    patience=8,  # Increased patience
    min_lr=0.000001,  # Lower min learning rate
    verbose=1
)
checkpoint = keras.callbacks.ModelCheckpoint(
    "best_gr_lower_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

# Use a lower learning rate
optimizer = keras.optimizers.Adam(learning_rate=0.0003)  # Lower learning rate

# Compile the model
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=optimizer, 
    metrics=['accuracy']
)

# Check class balance before training
print("\nClass balance in training data:")
unique_train_labels, train_counts = np.unique(labels_train, return_counts=True)
for label, count in zip(unique_train_labels, train_counts):
    print(f"  {mapping.inverse[label]}: {count}")

# Apply data augmentation to balance classes
def augment_underrepresented_classes(images, labels, threshold_percentile=25, augmentation_factor=3):
    """Augment underrepresented classes to improve balance"""
    unique_labels, counts = np.unique(labels, return_counts=True)
    threshold = np.percentile(counts, threshold_percentile)
    
    classes_to_augment = []
    for label, count in zip(unique_labels, counts):
        if count < threshold:
            classes_to_augment.append(label)
    
    if not classes_to_augment:
        print("No classes need augmentation")
        return images, labels
    
    print(f"Augmenting {len(classes_to_augment)} underrepresented classes: {[mapping.inverse[c] for c in classes_to_augment]}")
    
    # Extract samples from underrepresented classes
    aug_images = []
    aug_labels = []
    
    for i, label in enumerate(labels):
        if label in classes_to_augment:
            aug_images.append(images[i])
            aug_labels.append(label)
    
    if not aug_images:
        print("No samples found for underrepresented classes")
        return images, labels
    
    # Convert to arrays
    aug_images = np.array(aug_images)
    aug_labels = np.array(aug_labels)
    
    # Augment these samples
    print(f"Augmenting {len(aug_images)} samples by factor of {augmentation_factor}")
    augmented_images = []
    augmented_labels = []
    
    # Keep original samples
    augmented_images.extend(aug_images)
    augmented_labels.extend(aug_labels)
    
    # Add augmented samples
    for _ in range(augmentation_factor - 1):
        for img, label in zip(aug_images, aug_labels):
            # Apply random augmentations
            img_batch = tf.expand_dims(img, 0)  # Add batch dimension
            augmented_img = data_augmentation(img_batch)[0]  # Remove batch dimension
            
            augmented_images.append(augmented_img)
            augmented_labels.append(label)
    
    # Convert to arrays
    augmented_images = np.array(augmented_images)
    augmented_labels = np.array(augmented_labels)
    
    # Combine with original data
    combined_images = np.concatenate([images, augmented_images])
    combined_labels = np.concatenate([labels, augmented_labels])
    
    # Shuffle
    combined_labels, combined_images = shuffle(combined_labels, combined_images)
    
    print(f"After augmentation: {len(combined_images)} samples")
    
    # Check new class balance
    print("New class balance:")
    unique_labels, counts = np.unique(combined_labels, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"  {mapping.inverse[label]}: {count}")
    
    return combined_images, combined_labels

# Apply augmentation to training data
images_train, labels_train = augment_underrepresented_classes(images_train, labels_train)

# Train the model
print("\nTraining model...")
history = model.fit(
    images_train, labels_train, 
    batch_size=batch_size, 
    epochs=epochs,
    validation_data=(images_test, labels_test), 
    callbacks=[early_stopping, reduce_lr, checkpoint],
    class_weight=class_weight_dict,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.tight_layout()
plt.savefig('../static/gr_lower_training_history.png')
plt.show()

# Load the best model
try:
    best_model = keras.models.load_model("best_gr_lower_model.keras")
    print("Loaded best model from checkpoint")
    model = best_model
except:
    print("Using the last model (best model not found)")

# Evaluate the model
print("\nModel evaluation:")
test_loss, test_acc = model.evaluate(images_test, labels_test)
print(f"Test accuracy: {test_acc:.4f}")

# Generate predictions for confusion matrix
predictions = np.argmax(model.predict(images_test), axis=-1)

# Create confusion matrix
cm = confusion_matrix(labels_test, predictions, labels=list(range(len(mapping))))

# Plot confusion matrix
plt.figure(figsize=(15, 15))
sns.heatmap(
    cm, 
    annot=True, 
    cbar=False, 
    cmap="Blues",
    xticklabels=list(mapping.keys()), 
    yticklabels=list(mapping.keys())
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for Greek Lowercase Characters')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.savefig('../static/gr_lower_confusion_matrix.png')
plt.show()

# Save the model
model.save("grLowerCharacters.keras")
print("Model saved as grLowerCharacters.keras")

# Add a function to visualize examples of each class with their predictions
def visualize_predictions(images, true_labels, predictions, mapping, symbol_map=None, num_examples=10):
    """
    Visualize predictions on test images
    
    Args:
        images: Array of test images
        true_labels: Array of true labels (as integers)
        predictions: Array of predicted labels (as integers)
        mapping: Bidirectional dictionary mapping label names to indices
        symbol_map: Optional mapping from label names to symbols
        num_examples: Number of examples to show
    """
    plt.figure(figsize=(15, 10))
    
    # Count prediction distribution
    pred_counts = {}
    for pred in predictions:
        pred_name = mapping.inverse[pred]
        pred_counts[pred_name] = pred_counts.get(pred_name, 0) + 1
    
    # Print prediction distribution
    print("\nPrediction distribution:")
    for pred_name, count in sorted(pred_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {pred_name}: {count} ({count/len(predictions)*100:.2f}%)")
    
    # Calculate accuracy per class
    class_correct = {}
    class_total = {}
    
    for true, pred in zip(true_labels, predictions):
        true_name = mapping.inverse[true]
        class_total[true_name] = class_total.get(true_name, 0) + 1
        if true == pred:
            class_correct[true_name] = class_correct.get(true_name, 0) + 1
    
    print("\nAccuracy per class:")
    for class_name in sorted(class_total.keys()):
        correct = class_correct.get(class_name, 0)
        total = class_total[class_name]
        accuracy = correct / total if total > 0 else 0
        print(f"  {class_name}: {accuracy:.2f} ({correct}/{total})")
    
    # Show examples of both correct and incorrect predictions
    correct_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true == pred]
    incorrect_indices = [i for i, (true, pred) in enumerate(zip(true_labels, predictions)) if true != pred]
    
    # Ensure we have some of each
    num_correct = min(num_examples // 2, len(correct_indices))
    num_incorrect = min(num_examples - num_correct, len(incorrect_indices))
    
    # If we don't have enough incorrect, show more correct
    if num_incorrect < num_examples - num_correct:
        num_correct = min(num_examples - num_incorrect, len(correct_indices))
    
    # Randomly sample indices
    random.seed(42)  # For reproducibility
    sampled_correct = random.sample(correct_indices, num_correct) if correct_indices else []
    sampled_incorrect = random.sample(incorrect_indices, num_incorrect) if incorrect_indices else []
    
    # Combine and shuffle
    sampled_indices = sampled_correct + sampled_incorrect
    random.shuffle(sampled_indices)
    
    for i, idx in enumerate(sampled_indices[:num_examples]):
        plt.subplot(2, 5, i+1)
        
        # Get the true and predicted labels
        true_idx = true_labels[idx]
        pred_idx = predictions[idx]
        
        true_name = mapping.inverse[true_idx]
        pred_name = mapping.inverse[pred_idx]
        
        # Get the corresponding symbols if symbol_map is provided
        true_symbol = ""
        pred_symbol = ""
        if symbol_map is not None:
            true_symbol = symbol_map.get(true_name, "")
            pred_symbol = symbol_map.get(pred_name, "")
        
        # Display the image
        plt.imshow(images[idx].reshape(50, 50), cmap='gray')
        
        # Set color based on correctness
        color = 'green' if true_idx == pred_idx else 'red'
        
        # Set title with true and predicted labels
        title = f"True: {true_name}"
        if true_symbol:
            title += f" ({true_symbol})"
        title += f"\nPred: {pred_name}"
        if pred_symbol:
            title += f" ({pred_symbol})"
        
        plt.title(title, color=color)
        plt.axis('off')
    
    plt.suptitle("Greek Lowercase Character Predictions", fontsize=16)
    plt.tight_layout()
    plt.savefig('../static/gr_lower_predictions.png')
    plt.show()

# Visualize some predictions
print("\nVisualizing predictions on test data...")
visualize_predictions(
    images_test, 
    labels_test, 
    predictions, 
    mapping,
    greek_symbol_map
) 