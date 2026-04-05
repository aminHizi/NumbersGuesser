import os
import numpy as np
import struct

def load_images(file_path):
    with open(file_path, 'rb') as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num, rows, cols)
    return images

def load_labels(file_path):
    with open(file_path, 'rb') as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return labels

# Paths
raw_path = "data/raw"
processed_path = "data/processed"
os.makedirs(processed_path, exist_ok=True)

# Load and save training data
train_images = load_images(os.path.join(raw_path, "train-images-idx3-ubyte"))
train_labels = load_labels(os.path.join(raw_path, "train-labels-idx1-ubyte"))
np.save(os.path.join(processed_path, "train_images.npy"), train_images)
np.save(os.path.join(processed_path, "train_labels.npy"), train_labels)

# Load and save test data
test_images = load_images(os.path.join(raw_path, "t10k-images-idx3-ubyte"))
test_labels = load_labels(os.path.join(raw_path, "t10k-labels-idx1-ubyte"))
np.save(os.path.join(processed_path, "test_images.npy"), test_images)
np.save(os.path.join(processed_path, "test_labels.npy"), test_labels)

print("Data processed and saved to", processed_path)