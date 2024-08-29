import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from scipy.spatial.distance import euclidean
import numpy as np
import pickle

# Define image parameters and model paths
img_height, img_width = 224, 224
embedding_model_path = 'space_object_classifier_embeddings.keras'
dataset_images_dir = 'data/train'
embeddings_file = 'embeddings.pkl'
similarity_threshold = 30  # Adjust based on performance

# Define class labels
class_labels = ['Earth', 'Moon', 'Mars']

# Define the model structure
def create_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(len(class_labels), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Function to prepare dataset
def prepare_dataset(dataset_dir):
    images = []
    labels = []
    
    for label in class_labels:
        label_dir = os.path.join(dataset_dir, label)
        for file in os.listdir(label_dir):
            file_path = os.path.join(label_dir, file)
            img = load_img(file_path, target_size=(img_height, img_width))
            img_array = img_to_array(img)
            img_array = preprocess_input(img_array)
            images.append(img_array)
            labels.append(class_labels.index(label))
    
    images = np.array(images)
    labels = np.array(labels)
    labels = to_categorical(labels, num_classes=len(class_labels))
    
    return train_test_split(images, labels, test_size=0.2, random_state=42)

# Train and save the model
def train_and_save_model():
    model = create_model()
    X_train, X_test, y_train, y_test = prepare_dataset(dataset_images_dir)
    
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    
    # Save the model
    model.save(embedding_model_path)
    print(f'Model saved to {embedding_model_path}')

# Load model
def load_model():
    if os.path.exists(embedding_model_path):
        model = tf.keras.models.load_model(embedding_model_path)
        return model
    else:
        raise FileNotFoundError(f"Model file not found at path: {embedding_model_path}")

# Get embeddings for an image
def get_embedding(image_path, model):
    try:
        img = load_img(image_path, target_size=(img_height, img_width))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        embedding = model.predict(img_array)
        return embedding.flatten()
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

# Compute embeddings for all images in a directory
def compute_embeddings(directory, model):
    embeddings = {}
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            embedding = get_embedding(file_path, model)
            if embedding is not None:
                embeddings[file_path] = embedding
    return embeddings

# Save embeddings to a file
def save_embeddings(directory, model):
    embeddings = compute_embeddings(directory, model)
    with open(embeddings_file, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f'Embeddings saved to {embeddings_file}')

# Load embeddings from a file
def load_embeddings():
    if os.path.exists(embeddings_file):
        with open(embeddings_file, 'rb') as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"Embeddings file not found at path: {embeddings_file}")

# Find the most similar image based on embeddings
def find_most_similar_image(image_embedding, stored_embeddings):
    min_dist = float('inf')
    most_similar_image = None
    for img_path, stored_embedding in stored_embeddings.items():
        dist = euclidean(image_embedding, stored_embedding)
        if dist < min_dist:
            min_dist = dist
            most_similar_image = img_path
    return most_similar_image, min_dist

# Classify an image
def classify_image(image_path, model, stored_embeddings):
    img_embedding = get_embedding(image_path, model)
    if img_embedding is not None:
        most_similar_image, distance = find_most_similar_image(img_embedding, stored_embeddings)
        print(f'Most similar image: {most_similar_image}')
        print(f'Distance: {distance}')
        
        # Determine the class based on distance
        if distance < similarity_threshold:
            label = [label for label in class_labels if label.lower() in most_similar_image.lower()]
            return label[0] if label else 'Unknown'
        else:
            return 'Unknown'
    else:
        print("Error: Image embedding could not be computed.")
        return 'Error'

# Main execution
if __name__ == '__main__':
    # Train and save the model (run this only once to train)
    train_and_save_model()
    
    # Load the model
    model = load_model()
    
    # Save embeddings
    save_embeddings(dataset_images_dir, model)
    
    # Load embeddings
    stored_embeddings = load_embeddings()
    
    # Classify a given image
    image_to_classify_path = 'aaa.jpg'
    result = classify_image(image_to_classify_path, model, stored_embeddings)
    print(f'The image {image_to_classify_path} is classified as "{result}".')
