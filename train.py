import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau
)

from sklearn.metrics import confusion_matrix, classification_report

# ==========================================================
# Dataset Paths
# ==========================================================

train_dir = "dataset/train"
test_dir = "dataset/test"

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32
EPOCHS = 30

# ==========================================================
# Data Augmentation
# ==========================================================

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    shear_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

# ==========================================================
# Load Dataset
# ==========================================================

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

print("\nClass Labels")
print(train_generator.class_indices)

# ==========================================================
# Load MobileNetV2
# ==========================================================

base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224,224,3)
)

# Fine Tune Last Layers
base_model.trainable = True

for layer in base_model.layers[:-30]:
    layer.trainable = False

# ==========================================================
# Build Model
# ==========================================================

x = base_model.output

x = GlobalAveragePooling2D()(x)

x = Dense(512, activation="relu")(x)

x = Dropout(0.5)(x)

x = Dense(256, activation="relu")(x)

x = Dropout(0.3)(x)

output = Dense(1, activation="sigmoid")(x)

model = Model(
    inputs=base_model.input,
    outputs=output
)

# ==========================================================
# Compile Model
# ==========================================================

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ==========================================================
# Create Folders
# ==========================================================

os.makedirs("model", exist_ok=True)
os.makedirs("output", exist_ok=True)

# ==========================================================
# Callbacks
# ==========================================================

checkpoint = ModelCheckpoint(
    "model/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

earlystop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-7,
    verbose=1
)

# ==========================================================
# Train Model
# ==========================================================

history = model.fit(
    train_generator,
    validation_data=test_generator,
    epochs=EPOCHS,
    callbacks=[
        checkpoint,
        earlystop,
        reduce_lr
    ]
)

# ==========================================================
# Save Final Model
# ==========================================================

model.save("model/final_model.keras")

print("\nFinal Model Saved Successfully!")

# ==========================================================
# Accuracy Graph
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

plt.title("Training vs Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.savefig("output/accuracy.png")
plt.show()

# ==========================================================
# Loss Graph
# ==========================================================

plt.figure(figsize=(8,6))

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.title("Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig("output/loss.png")
plt.show()

# ==========================================================
# Evaluate Model
# ==========================================================

loss, accuracy = model.evaluate(test_generator)

print("\n====================================")
print("Test Accuracy :", round(accuracy * 100,2), "%")
print("Test Loss     :", round(loss,4))
print("====================================")

# ==========================================================
# Predictions
# ==========================================================

test_generator.reset()

predictions = model.predict(test_generator)

predicted_classes = (predictions > 0.5).astype(int).reshape(-1)

true_classes = test_generator.classes

class_names = list(test_generator.class_indices.keys())

# ==========================================================
# Classification Report
# ==========================================================

print("\nClassification Report\n")

print(
    classification_report(
        true_classes,
        predicted_classes,
        target_names=class_names
    )
)

# ==========================================================
# Confusion Matrix
# ==========================================================

cm = confusion_matrix(
    true_classes,
    predicted_classes
)

plt.figure(figsize=(6,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.savefig("output/confusion_matrix.png")
plt.show()

print("\nConfusion Matrix Saved Successfully!")

print("\n====================================")
print("PROJECT COMPLETED SUCCESSFULLY")
print("====================================")