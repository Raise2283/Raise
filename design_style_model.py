import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Load the CSV file
# The CSV contains image paths and their corresponding styles.
# We'll adjust the image paths to point to the 'design_images' directory on the desktop.
csv_path = 'design_classify1.csv'  # Update this if the CSV is in a different location
df = pd.read_csv(csv_path)

# Assuming images are stored in 'design_images' on the desktop
desktop_path = os.path.expanduser('~/Desktop/design_images')
df['image'] = df['image'].apply(lambda x: os.path.join(desktop_path, x.split('/')[-1]))

# Step 2: Set up data generators
# We'll use ImageDataGenerator for loading and augmenting images.
# 20% of the data will be used for validation.
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Normalize pixel values to [0, 1]
    validation_split=0.2,    # Reserve 20% of data for validation
    rotation_range=20,       # Randomly rotate images for augmentation
    width_shift_range=0.2,   # Randomly shift images horizontally
    height_shift_range=0.2,  # Randomly shift images vertically
    shear_range=0.2,         # Apply shear transformations
    zoom_range=0.2,          # Randomly zoom in on images
    horizontal_flip=True     # Flip images horizontally
)

# Training data generator
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='image',           # Column with image paths
    y_col='style',           # Column with labels (design styles)
    target_size=(224, 224),  # Resize images to match ResNet50 input
    batch_size=32,           # Number of images per batch
    class_mode='categorical',# One-hot encoded labels for multi-class classification
    subset='training'        # Use 80% of data for training
)

# Validation data generator
validation_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='image',
    y_col='style',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'      # Use 20% of data for validation
)

# Step 3: Define the model architecture
# We'll use ResNet50 as the base model with custom classification layers.
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom layers on top of ResNet50
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Reduce spatial dimensions
x = Dense(1024, activation='relu')(x)  # Fully connected layer
predictions = Dense(24, activation='softmax')(x)  # Output layer for 24 classes

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base ResNet50 layers to use pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Step 4: Compile the model
# We'll use Adam optimizer and categorical cross-entropy for multi-class classification.
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Step 5: Train the model
# We'll train for 10 epochs initially; adjust based on performance.
model.fit(
    train_generator,
    epochs=10,  # You can increase this if needed
    validation_data=validation_generator
)

# Step 6: Save the trained model
# The model will be saved as 'design_style_model.h5' in the current directory.
model.save('design_style_model.h5')

print("Model training complete and saved as 'design_style_model.h5'.")