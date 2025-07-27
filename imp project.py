import os
from PIL import Image
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def load_images(folder, image_size=(48, 48)):
    X, y = [], []
    classes = os.listdir(folder)
    for label in classes:
        class_path = os.path.join(folder, label)
        if not os.path.isdir(class_path):
            continue
        for filename in os.listdir(class_path):
            img_path = os.path.join(class_path, filename)
            try:
                img = Image.open(img_path).convert('L')
                img = img.resize(image_size)
                img_array = np.asarray(img) / 255.0
                X.append(img_array.flatten())
                y.append(label)
            except Exception as e:
                print(f"Error reading {img_path}: {e}")
                continue
    return np.array(X), np.array(y)

# Update these paths to match your folder locations
train_path = r'A:\imp proj\train'
test_path  = r'A:\imp proj\test'

print("Loading training data...")
X_train, y_train = load_images(train_path)
print(f"Loaded {len(X_train)} training images.")

print("Loading testing data...")
X_test, y_test = load_images(test_path)
print(f"Loaded {len(X_test)} testing images.")

# Encode emotion labels
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Reduce dimensionality for speed
print("Applying PCA...")
pca = PCA(n_components=100)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Train Random Forest
print("Training Random Forest...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_pca, y_train_enc)

# Predict
print("Evaluating model...")
y_pred = clf.predict(X_test_pca)

# Report
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))
