# Import required libraries
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import cv2

# Install compatible version of DeepLake
!pip install "deeplake<4"

# Load PlantVillage dataset (with augmentation)
import deeplake
ds = deeplake.load('hub://activeloop/plantvillage-with-augmentation')

# Quick visualization of dataset
ds.visualize()

# Get all available class names
all_classes = ds.labels.info.class_names
print(all_classes)
print("Total classes:", len(all_classes))

# Select a subset of classes (example: corn, grape, cherry)
selected_classes = [
    # Corn
    "Corn_northern_leaf_blight",
    "Corn_gray_leaf_spot",
    "Corn_common_rust",
    "Corn_healthy",
    # Grape
    "Grape_healthy",
    "Grape_black_measles",
    "Grape_black_rot",
    "Grape_leaf_blight",
    # Cherry
    "Cherry_healthy",
    "Cherry_powdery_mildew"
]

# Preprocess dataset: resize images and map labels
x = []
y = []
IMG_SIZE = 128

for i in range(len(ds.labels)):
    label = ds.labels[i].numpy().item()
    class_name = all_classes[label]
    if class_name in selected_classes:
        img = ds.images[i].numpy()
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        x.append(img_resized)
        y.append(selected_classes.index(class_name))

print(np.unique(y))

# Show one example image per selected class
unique_images = []
unique_labels = []

for i in range(len(y)):
    label = y[i]
    if label not in unique_labels:
        unique_images.append(x[i])
        unique_labels.append(label)


    if len(unique_labels) == len(selected_classes):
        break


plt.figure(figsize=(15, 6))
for i in range(len(unique_labels)):
    plt.subplot(2, (len(unique_labels)+1)//2, i+1)
    plt.imshow(unique_images[i])
    plt.title(selected_classes[unique_labels[i]])
    plt.axis('off')
plt.show()

# Normalize images

x = np.array(x, dtype='float32') / 255.0
y = np.array(y)

print("Dataset shape:", x.shape, y.shape)
print("Classes:", selected_classes)

# Split dataset into train / validation / test
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, stratify=y, random_state=42
)

x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

print("Train:", x_train.shape, "Val:", x_val.shape, "Test:", x_test.shape)

# Define CNN architecture

model = Sequential([
    Conv2D(input_shape=(128,128,3), filters=32, kernel_size=(3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(64, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(128, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Conv2D(256, (3,3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D((2,2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(256, activation='relu'),
    Dropout(0.5),

    Dense(10, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Training callbacks
checkpoint = ModelCheckpoint(filepath="PlantVillageFew.h5",
                             monitor="val_accuracy",
                             save_best_only=True,
                             mode="max",
                             verbose=1)

early_stopping = EarlyStopping(monitor="val_accuracy",
                               patience=3,
                               restore_best_weights=True,
                               verbose=1)


history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=64,
                    validation_data=(x_val, y_val),
                    callbacks=[checkpoint, early_stopping])

# Load the best saved model
best_model = load_model("PlantVillageFew.h5")

print(history.history.keys())

# Evaluate performance
print(history.history['accuracy'])
print(history.history['loss'])
print(history.history['val_accuracy'])
print(history.history['val_loss'])

test_loss, test_acc = best_model.evaluate(x_test, y_test, verbose=2)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_acc}")
print(f"Test Accuracy Percent: {test_acc * 100:.2f}%")

# Training curves
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], marker='o', label='Training Accuracy')
plt.plot(history.history['val_accuracy'], marker='o', label='Validation Accuracy')
plt.title('Training vs Validation Accuracy', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], marker='o', label='Training Loss')
plt.plot(history.history['val_loss'], marker='o', label='Validation Loss')
plt.title('Training vs Validation Loss', fontsize=14)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()

# Single image prediction
from tensorflow.keras.preprocessing import image
image_path = "/content/Grape.png" # replace with your own image
img = image.load_img(image_path, target_size=(128, 128))

img_array = image.img_to_array(img)

img_array = img_array / 255.0

img_array = np.expand_dims(img_array, axis=0)
print(img_array.shape)

predictions = best_model.predict(img_array)
predicted_class_index = np.argmax(predictions)
print(f"Predicted class index: {predicted_class_index}")

predicted_class_name = selected_classes[predicted_class_index]
print(f"Predicted class: {predicted_class_name}")