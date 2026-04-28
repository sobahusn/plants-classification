# 🌿 Proyek Klasifikasi Gambar: Plants Classification

> **Image Classification Project** - Klasifikasi otomatis jenis-jenis tanaman menggunakan Deep Learning dengan dataset dari Kaggle

---

## 📋 Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Hasil dan Performa](#hasil-dan-performa)
- [Instalasi dan Setup](#instalasi-dan-setup)
- [Penggunaan](#penggunaan)
- [Output Model](#output-model)
- [Struktur Proyek](#struktur-proyek)
- [Penulis](#penulis)

---

## 🎯 Tentang Proyek

Proyek ini adalah implementasi **Deep Learning** untuk klasifikasi gambar jenis-jenis tanaman secara otomatis. Model dilatih menggunakan dataset publik dari Kaggle yang berisi ribuan gambar tanaman dari berbagai spesies.

### Tujuan Utama:
- Membangun model neural network yang dapat mengklasifikasikan berbagai jenis tanaman dari gambar
- Mencapai akurasi tinggi dalam prediksi klasifikasi
- Mengoptimalkan model untuk deployment di berbagai platform (web, mobile)
- Menerapkan best practices dalam deep learning (data augmentation, callbacks, transfer learning)

---

## 📊 Dataset

### Sumber Data
- **Platform**: Kaggle
- **Dataset**: [Plants Classification - marquis03/plants-classification](https://www.kaggle.com/datasets/marquis03/plants-classification)
- **Jumlah Gambar**: Ribuan gambar tanaman
- **Jumlah Kelas**: Multiple plant species
- **Format File**: JPEG/PNG
- **Resolusi**: Distandarisasi menjadi 224x224 piksel

### Pembagian Data
```
Dataset
├── train/        # Data training (60%)
├── val/          # Data validation (20%)
└── test/         # Data testing (20%)
```

### Preprocessing Data
- **Image Size**: 224 × 224 × 3 (RGB)
- **Batch Size**: 32
- **Normalisasi**: Rescaling 1/255

### Data Augmentation
Model menggunakan augmentasi data untuk meningkatkan generalisasi:
```python
- RandomFlip("horizontal")      # Flip gambar secara horizontal
- RandomRotation(0.2)           # Rotasi hingga 20 derajat
- RandomZoom(0.2)              # Zoom hingga 20%
```

---

## 🏗️ Arsitektur Model

### Model Base: MobileNetV2 (Transfer Learning)

Proyek ini menggunakan **MobileNetV2** sebagai pre-trained model dengan fine-tuning:

```
┌─────────────────────────────────────────┐
│ Input: 224x224x3 RGB Image              │
├─────────────────────────────────────────┤
│ Rescaling Layer (1/255)                 │
├─────────────────────────────────────────┤
│ MobileNetV2 Base Model (ImageNet)       │
│ (Frozen - transfer learning)            │
├─────────────────────────────────────────┤
│ GlobalAveragePooling2D                  │
├─────────────────────────────────────────┤
│ Dense(128, activation='relu')           │
├─────────────────────────────────────────┤
│ Dropout(0.5)                            │
├─────────────────────────────────────────┤
│ Dense(num_classes, activation='softmax')│
├─────────────────────────────────────────┤
│ Output: Probability per class           │
└─────────────────────────────────────────┘
```

### Konfigurasi Model

| Parameter | Nilai |
|-----------|-------|
| **Base Model** | MobileNetV2 (ImageNet weights) |
| **Input Shape** | (224, 224, 3) |
| **Optimizer** | Adam (learning_rate=0.0001) |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Metrics** | Accuracy |
| **Epochs** | 50 |
| **Batch Size** | 32 |

### Training Callbacks

Untuk mengoptimalkan training, digunakan 3 callbacks utama:

1. **EarlyStopping**
   - Monitor: `val_accuracy`
   - Patience: 5 epochs
   - Menghentikan training jika tidak ada peningkatan

2. **ReduceLROnPlateau**
   - Monitor: `val_loss`
   - Factor: 0.5 (kurangi learning rate menjadi 50%)
   - Patience: 2 epochs

3. **ModelCheckpoint**
   - Menyimpan model terbaik ke `best_model.h5`
   - Save best only: True

---

## 📈 Hasil dan Performa

### Metriks Evaluasi

Model dievaluasi menggunakan:
- **Test Accuracy**: Akurasi pada data testing
- **Test Loss**: Loss value pada data testing
- **Confusion Matrix**: Visualisasi prediksi vs actual label
- **Classification Report**: Precision, Recall, F1-Score per class

### Visualisasi Training

Model menghasilkan visualisasi untuk:
- **Model Accuracy**: Menunjukkan performa training vs validation
- **Model Loss**: Menunjukkan penurunan loss selama training
- **Confusion Matrix**: Heatmap prediksi vs actual

### Analisis Per Kelas

Classification report menampilkan:
- **Precision**: Akurasi prediksi positif
- **Recall**: Kemampuan mendeteksi setiap kelas
- **F1-Score**: Rata-rata harmonik precision dan recall
- **Support**: Jumlah sampel per kelas

---

## 🚀 Instalasi dan Setup

### Prerequisites
- Python 3.7+
- pip atau conda
- GPU (optional, untuk training lebih cepat)

### Instalasi Dependencies

1. **Clone atau download repository**
   ```bash
   cd path/to/project
   ```

2. **Install packages yang diperlukan**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download dataset dari Kaggle** (dalam notebook)
   ```bash
   pip install kaggle
   # Upload kaggle.json dan jalankan notebook untuk download dataset
   ```

### Requirements
```
tensorflow      # Deep Learning framework
numpy          # Numerical computations
matplotlib     # Data visualization
tensorflowjs   # Convert model untuk web
```

---

## 💻 Penggunaan

### Cara Menjalankan Notebook

1. **Setup environment**
   ```bash
   pip install -r requirements.txt
   ```

2. **Buka notebook di Jupyter/Colab**
   ```bash
   jupyter notebook notebook.ipynb
   ```

3. **Jalankan cell secara berurutan**:
   - Import libraries
   - Load dataset dari Kaggle
   - Preprocessing dan augmentasi
   - Training model
   - Evaluasi dan visualisasi
   - Konversi model

### Prediksi Gambar Baru

Untuk melakukan prediksi pada gambar baru:

```python
import tensorflow as tf
from PIL import Image

# Load model
model = tf.keras.models.load_model('best_model.h5')

# Load dan preprocessing gambar
img = Image.open('plant_image.jpg').resize((224, 224))
img_array = tf.keras.preprocessing.image.img_to_array(img)
img_array = tf.expand_dims(img_array, 0)

# Prediksi
prediction = model.predict(img_array)
predicted_class = class_names[tf.argmax(prediction[0])]
confidence = tf.reduce_max(prediction[0]).numpy()

print(f"Kelas: {predicted_class}, Confidence: {confidence:.2%}")
```

---

## 📦 Output Model

Proyek ini menghasilkan model dalam berbagai format untuk deployment:

### 1. SavedModel (TensorFlow Format)
```
saved_model/
├── fingerprint.pb           # Fingerprint untuk verifikasi
├── saved_model.pb           # Model definition
└── variables/
    ├── variables.index      # Index variabel
    └── variables.data-*     # Data variabel
```

**Penggunaan**:
```python
model = tf.keras.models.load_model('saved_model/')
```

### 2. TensorFlow Lite (TFLite)
```
tflite/
└── model.tflite             # Model untuk mobile/embedded
```

**Keuntungan**: Ukuran lebih kecil, latency rendah, cocok untuk mobile
**Penggunaan**: Android/iOS apps dengan TFLite runtime

### 3. TensorFlow.js (Web Format)
```
tfjs_model/
└── model.json               # Model definition untuk web
```

**Penggunaan**: Deployment di browser dengan JavaScript

### 4. Best Model (H5 Format)
```
best_model.h5               # Model terbaik dari training
```

**Penggunaan**: Loading cepat, kompatibel dengan berbagai framework

---

## 📁 Struktur Proyek

```
submission/
├── notebook.ipynb              # Main notebook dengan full code
├── README.md                   # Dokumentasi proyek (file ini)
├── requirements.txt            # Dependencies list
├── best_model.h5              # Model terbaik (H5 format)
├── saved_model/               # SavedModel format
│   ├── fingerprint.pb
│   ├── saved_model.pb
│   └── variables/
│       ├── variables.index
│       └── variables.data-00000-of-00001
├── tflite/                    # TensorFlow Lite format
│   └── model.tflite
└── tfjs_model/                # TensorFlow.js format
    └── model.json
```

---

## 🔧 Technologies & Libraries

| Technology | Versi | Fungsi |
|-----------|-------|--------|
| **TensorFlow** | 2.x | Deep Learning framework |
| **Keras** | Built-in | High-level API |
| **NumPy** | Latest | Array operations |
| **Matplotlib** | Latest | Visualization |
| **Scikit-learn** | Latest | Metrics (confusion matrix, etc) |
| **Pillow** | Latest | Image processing |

---

## 📚 Metode & Teknik yang Digunakan

### 1. Transfer Learning
- Menggunakan MobileNetV2 pre-trained dari ImageNet
- Freeze base model layers untuk mempercepat training
- Fine-tune dengan dataset spesifik

### 2. Data Augmentation
- Meningkatkan variasi data training
- Mengurangi overfitting
- Meningkatkan generalisasi model

### 3. Regularization
- Dropout (0.5) untuk mengurangi overfitting
- Early stopping untuk mencegah overtraining
- Learning rate reduction untuk fine-tuning

### 4. Optimization
- Adam optimizer dengan custom learning rate
- Batch processing (batch size 32)
- Model checkpointing untuk menyimpan best weights

---

## 👤 Penulis

**M. Sobahus Sururin Ni'am**
- Email: sobahusn27@gmail.com
- Dicoding: [sobahusn](https://www.dicoding.com/users/sobahusn/)

---

## 📝 Lisensi

Proyek ini dibuat sebagai bagian dari Dicoding Bootcamp - Belajar Fundamental Deep Learning.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Plants Classification by marquis03](https://www.kaggle.com/datasets/marquis03/plants-classification)
- **Framework**: TensorFlow/Keras
- **Pre-trained Model**: MobileNetV2 dari ImageNet

---

**Last Updated**: April 2026