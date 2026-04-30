# Plants Classification: Deep Learning Project

Deep learning project untuk klasifikasi jenis tanaman dari gambar menggunakan Convolutional Neural Networks (CNN) dengan TensorFlow/Keras.

## Project Information

- **Course:** Belajar Fundamental Deep Learning (Dicoding)
- **Author:** M. Sobahus Sururin Ni'am
- **Model:** Convolutional Neural Network (CNN)
- **Framework:** TensorFlow 2.x / Keras
- **Dataset:** Plants Classification (Kaggle)
- **Classes:** 5 jenis tanaman
- **Status:** Complete
- **Rating Reviewer:** 4 Bintang rejected: 3x

## Dataset

- **Nama:** Plants Classification
- **Source:** Kaggle (marquis03)
- **Total Gambar:** 22,050 images
- **Jumlah Kelas:** 5 jenis tanaman
- **Ukuran Gambar:** 150x150 pixels
- **Format:** JPEG (RGB)

### Pembagian Data

- Training: 70% (~2,310 gambar)
- Validation: 15% (~495 gambar)
- Test: 15% (~495 gambar)

### Jenis Tanaman

1. Aloevera
2. Banana
3. Bilimbi
4. Cantaloupe
5. Cassava

## Project Structure

```
submission/
├── README.md                    # Dokumentasi proyek
├── requirements.txt             # Python dependencies
├── best_model.keras             # Model terbaik (format Keras)
├── notebook.ipynb               # Jupyter notebook dengan full analysis
├── saved_model/                 # TensorFlow SavedModel format
│   ├── saved_model.pb
│   ├── fingerprint.pb
│   └── variables/
├── tflite/                      # Model untuk mobile deployment
│   ├── model.tflite
│   └── label.txt
└── tfjs_model/                  # Model untuk browser
    └── model.json
```

## Model Architecture

### Arsitektur CNN

Input (150x150x3)
|
Conv2D (32 filters, 3x3) + BatchNorm + MaxPool (2x2)
|
Conv2D (64 filters, 3x3) + BatchNorm + MaxPool (2x2)
|
Conv2D (128 filters, 3x3) + BatchNorm + MaxPool (2x2)
|
Conv2D (256 filters, 3x3) + BatchNorm + MaxPool (2x2)
|
GlobalAveragePooling2D
|
Dense (512 units, ReLU) + Dropout (0.5)
|
Dense (5 units, Softmax) → Output

### Hyperparameter

- Optimizer: Adam (learning_rate=0.001)
- Loss Function: Categorical Crossentropy
- Batch Size: 32
- Epochs: 20 (dengan early stopping)
- Image Augmentation: Rotation, Shift, Zoom, Brightness, Flip

### Regularization

- Batch Normalization di setiap Conv2D layer
- Dropout (0.5) setelah Dense layer
- Early Stopping (patience=15)

- Python 3.8+
- TensorFlow 2.10+
- Lihat requirements.txt untuk dependencies lengkap

### Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Load model yang sudah dilatih:

```python
import tensorflow as tf
model = tf.keras.models.load_model('best_model.keras')
```

## Cara Menggunakan

### 1. Menggunakan Model yang Sudah Dilatih

```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Load model
model = tf.keras.models.load_model('best_model.keras')

# Load dan preprocess gambar
img = load_img('plant_image.jpg', target_size=(150, 150))
img_array = img_to_array(img) / 255.0
img_batch = np.expand_dims(img_array, axis=0)

# Prediksi
predictions = model.predict(img_batch)
predicted_class = np.argmax(predictions[0])
confidence = predictions[0][predicted_class]

# Class mapping
class_names = ['Aloevera', 'Banana', 'Bilimbi', 'Cantaloupe', 'Cassava']
print(f"Prediksi: {class_names[predicted_class]}")
print(f"Confidence: {confidence:.2%}")
```

### 2. Menjalankan Notebook

Buka notebook.ipynb untuk melihat proses training lengkap:

```bash
jupyter notebook notebook.ipynb
```

### 3. Menggunakan SavedModel

```python
import tensorflow as tf

model = tf.keras.models.load_model('saved_model')
predictions = model.predict(input_data)
```

### 4. Mobile Deployment (TFLite)

Model yang sudah dioptimasi tersedia di folder `tflite/model.tflite`

## Performance Metrics

### Training Results

- Training Accuracy: 90-95%
- Validation Accuracy: 85-92%
- Test Accuracy: 85-90%
- Final Loss: < 0.20
- Training Time: 5-10 menit (dengan GPU)

### Model Comparison

Versi sebelumnya menggunakan transfer learning dengan MobileNetV2 untuk klasifikasi 30 kelas. Versi current menggunakan custom CNN architecture yang lebih sederhana dengan 5 kelas untuk hasil yang lebih optimal.

## File Artifacts

- **best_model.keras**: Model terbaik dalam format Keras (50-100 MB)
- **saved_model/**: Model dalam format TensorFlow SavedModel
- **tflite/model.tflite**: Model yang dioptimasi untuk mobile (15-25 MB)
- **tfjs_model/**: Model dalam format JavaScript

## Author

**M. Sobahus Sururin Ni'am**

- Email: sobahusn27@gmail.com
- Dicoding: https://www.dicoding.com/users/sobahusn/
- GitHub: https://github.com/sobahusn

## License

This project is part of the **Dicoding - "Belajar Fundamental Deep Learning"** course submission.

Last Updated: May 2026
