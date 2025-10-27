# app.py
import os
import cv2
import dlib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import Model

# ==============================
# STEP 1: Initialize models
# ==============================
detector = dlib.get_frontal_face_detector()
base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg')
print("✅ Dlib & MobileNetV2 Loaded Successfully.\n")

# ==============================
# STEP 2: Extract face using Dlib
# ==============================
def extract_face(image, required_size=(160, 160)):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    faces_array = []
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        face_img = image[y:y+h, x:x+w]
        if face_img.size == 0:
            continue
        face_img = cv2.resize(face_img, required_size)
        faces_array.append((face_img, (x, y, w, h)))
    return faces_array

# ==============================
# STEP 3: Create Embedding with MobileNetV2
# ==============================
def get_embedding(face):
    face = cv2.resize(face, (224, 224))
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    face = preprocess_input(face)
    embedding = base_model.predict(face)
    return embedding[0]

# ==============================
# STEP 4: Load Dataset
# ==============================
def load_dataset(dataset_path):
    X, y = [], []
    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        print(f"📁 Loading faces for: {person_name}")
        for file in os.listdir(person_dir):
            img_path = os.path.join(person_dir, file)
            img = cv2.imread(img_path)
            if img is None:
                continue
            faces = extract_face(img)
            for face_img, _ in faces:
                embedding = get_embedding(face_img)
                X.append(embedding)
                y.append(person_name)
    return np.asarray(X), np.asarray(y)

print("📂 Loading Dataset...\n")
X, y = load_dataset("fixed_train")
print(f"✅ Dataset Loaded: {X.shape}, Labels: {len(np.unique(y))}\n")

# ==============================
# STEP 5: Train KNN Classifier
# ==============================
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X, y)
print("✅ KNN Model Trained Successfully!\n")

# ==============================
# STEP 6: Prediction Function
# ==============================
def recognize_face(image_path):
    img = cv2.imread(image_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = extract_face(img)
    for face_img, (x, y, w, h) in faces:
        embedding = get_embedding(face_img)
        embedding = np.expand_dims(embedding, axis=0)
        prediction = knn.predict(embedding)[0]
        prob = knn.predict_proba(embedding)
        confidence = np.max(prob) * 100
        color = (0, 255, 0) if confidence > 80 else (0, 0, 255)
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img, f'{prediction} ({confidence:.1f}%)', (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.imshow("Result", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# ==============================
# STEP 7: Test Image
# ==============================
test_image = "pat_i.jpg"
recognize_face(test_image)
