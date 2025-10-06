import numpy as np
import os
from sklearn.model_selection import train_test_split


# List of files
categories = ['1.npy', '2.npy', '3.npy', '4.npy', '5.npy',
              '6.npy', '7.npy', '8.npy', '9.npy', '10.npy', '11.npy']

# Path to Downloads
downloads_path = os.path.expanduser('~/Downloads')

X = []
y = []

for idx, fname in enumerate(categories):
    data = np.load(os.path.join(downloads_path, fname))
    # Binarize
    data_bin = (data > 128).astype(np.uint8)
    X.append(data_bin)
    y.append(np.full(data_bin.shape[0], idx, dtype=np.uint8))

X = np.vstack(X)
y = np.concatenate(y)

print(X.shape, y.shape)  # X is (total_images, 784), y is (total_images,)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)