import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import json
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def load_data(json_path, image_dir):
    with open(json_path) as f:
        data = json.load(f)
    
    image_paths = []
    labels = []
    label_to_idx = {}
    
    for key, item in data.items():
        img_path = os.path.join(image_dir, f"{key}.jpg")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            label = item['label']
            if label not in label_to_idx:
                label_to_idx[label] = len(label_to_idx)
            labels.append(label_to_idx[label])
    
    # Convert to numpy arrays
    labels = np.array(labels)
    
    # Split data (8:1:1)
    X_train, X_test, y_train, y_test = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.125, random_state=42)  # 0.125*0.8=0.1
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), label_to_idx

def preprocess_image(image_path, target_size):
    img = load_img(image_path, target_size=target_size)
    img = img_to_array(img)
    img = img / 255.0  # Normalize
    return img

def build_vgg16_model(input_shape=(224, 398, 3), num_classes=22):
    # Load pre-trained VGG16 without top layers
    base_model = VGG16(weights='imagenet', 
                      include_top=False, 
                      input_shape=input_shape)
    
    # Freeze convolutional layers
    for layer in base_model.layers:
        layer.trainable = False
    
    # Add custom fully connected layers
    inputs = Input(shape=input_shape)
    x = base_model(inputs)
    x = Flatten()(x)
    
    # Modified FC layers
    x = Dense(1024, activation='relu', name='FC1_Layer')(x)
    x = Dense(1024, activation='relu', name='FC2_Layer')(x)
    outputs = Dense(num_classes, activation='softmax', name='FC3_Layer')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer=SGD(learning_rate=0.001),
                 loss='categorical_crossentropy',
                 metrics=['accuracy'])
    
    return model

def train_model():
    # Configure GPU
    configure_gpu()
    
    # Load data
    (X_train, y_train), (X_val, y_val), (X_test, y_test), label_to_idx = load_data(
        'data/bones_label (1).json', 'data/image/')
    
    # Create model
    model = build_vgg16_model()
    model.summary()
    
    # Prepare callbacks
    callbacks = [
        ModelCheckpoint(
            filepath='history/vgg16_best.h5',
            save_best_only=True,
            monitor='val_accuracy'
        ),
        TensorBoard(
            log_dir='history/logs',
            histogram_freq=1
        )
    ]
    
    # Create data generator
    def data_generator(image_paths, labels, batch_size):
        num_samples = len(image_paths)
        while True:
            indices = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                batch_indices = indices[i:i+batch_size]
                batch_images = []
                batch_labels = []
                
                for idx in batch_indices:
                    img = preprocess_image(image_paths[idx], (224, 398))
                    batch_images.append(img)
                    batch_labels.append(labels[idx])
                
                yield np.array(batch_images), tf.keras.utils.to_categorical(
                    np.array(batch_labels), num_classes=22)
    
    # Train model
    history = model.fit(
        data_generator(X_train, y_train, batch_size=512),
        steps_per_epoch=len(X_train)//512,
        epochs=132000,
        validation_data=data_generator(X_val, y_val, batch_size=512),
        validation_steps=len(X_val)//512,
        callbacks=callbacks
    )
    
    # Evaluate on test set
    test_generator = data_generator(X_test, y_test, batch_size=512)
    test_steps = len(X_test)//512
    test_loss, test_acc = model.evaluate(test_generator, steps=test_steps)
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    train_model()
