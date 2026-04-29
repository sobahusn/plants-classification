# 🌿 Proyek Klasifikasi Gambar: Plants Classification

> **Image Classification Project** - Klasifikasi otomatis jenis-jenis tanaman menggunakan Deep Learning dengan dataset dari Kaggle

---

## Daftar Isi

- [Tentang Proyek](#tentang-proyek)
- [Dataset](#dataset)
- [Arsitektur Model](#arsitektur-model)
- [Hasil dan Performa](#hasil-dan-performa)
- [Instalasi dan Setup](#instalasi-dan-setup)
- [Penggunaan](#penggunaan)
- [Output Model](#output-model)
- [Struktur Proyek](#struktur-proyek)
- [Penulis](#penulis)

## Tentang Proyek

Proyek ini adalah implementasi **Deep Learning** untuk klasifikasi gambar jenis-jenis tanaman secara otomatis. Model dilatih menggunakan dataset publik dari Kaggle yang berisi ribuan gambar tanaman dari berbagai spesies.

### Tujuan Utama:

- Membangun model neural network yang dapat mengklasifikasikan berbagai jenis tanaman dari gambar
- Mencapai akurasi tinggi dalam prediksi klasifikasi
- Mengoptimalkan model untuk deployment di berbagai platform (web, mobile)
- Menerapkan best practices dalam deep learning (data augmentation, callbacks, transfer learning)

## Dataset

### Sumber Data

- **Platform**: Kaggle
- **Dataset**: [Plants Classification - marquis03/plants-classification](https://www.kaggle.com/datasets/marquis03/plants-classification)
- **Jumlah Gambar**: Ribuan gambar tanaman
- **Jumlah Kelas**: 30 jenis tanaman
- **Format File**: JPEG/PNG
- **Resolusi**: Distandarisasi menjadi 224x224 piksel

### Daftar Kelas Tanaman (30 Classes)

```
Aloevera        Banana          Bilimbi         Cantaloupe      Cassava
Coconut         Corn            Cucumber        Curcuma         Eggplant
Galangal        Ginger          Guava           Kale            Longbeans
Mango           Melon           Orange          Paddy           Papaya
Peperchili      Pineapple       Pomelo          Shallot         Soybeans
Spinach         Sweetpotatoes   Tobacco         Waterapple      Watermelon
```

### Pembagian Data

Proses splitting data dilakukan secara otomatis menggunakan `tf.keras.utils.image_dataset_from_directory`:

1. **Penggabungan Dataset**: Folder `val` dan `test` digabungkan ke dalam folder `combined_val_test_ds`
2. **Automatic Split**:
   - **Training**: 80% dari total data
   - **Validation**: 10% dari total data
   - **Test**: 10% dari total data

```
Dataset
├── train/                    # Data training + val + test (digabung)
│   └── [30 plant classes]/
└── combined_val_test_ds/     # Backup data asli val+test
    └── [30 plant classes]/
```

```python
# 80% untuk Training
train_ds = image_dataset_from_directory(
    ..., validation_split=0.2, subset="training", ...
)

# 20% sisanya, kemudian dibagi menjadi 50% val dan 50% test
temp_ds = image_dataset_from_directory(
    ..., validation_split=0.2, subset="validation", ...
)
val_ds = temp_ds.take(total_batches // 2)     # 10%
test_ds = temp_ds.skip(total_batches // 2)    # 10%
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

## Arsitektur Model

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

| Parameter         | Nilai                           |
| ----------------- | ------------------------------- |
| **Base Model**    | MobileNetV2 (ImageNet weights)  |
| **Input Shape**   | (224, 224, 3)                   |
| **Optimizer**     | Adam (learning_rate=0.0001)     |
| **Loss Function** | Sparse Categorical Crossentropy |
| **Metrics**       | Accuracy                        |
| **Epochs**        | 50                              |
| **Batch Size**    | 32                              |

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
import numpy as np

# Daftar kelas tanaman
class_names = [
    'aloevera', 'banana', 'bilimbi', 'cantaloupe', 'cassava', 'coconut',
    'corn', 'cucumber', 'curcuma', 'eggplant', 'galangal', 'ginger',
    'guava', 'kale', 'longbeans', 'mango', 'melon', 'orange', 'paddy',
    'papaya', 'peperchili', 'pineapple', 'pomelo', 'shallot', 'soybeans',
    'spinach', 'sweetpotatoes', 'tobacco', 'waterapple', 'watermelon'
]

# Load model (pilih salah satu)
# Option 1: Load dari H5
model = tf.keras.models.load_model('best_model.h5')

# Option 2: Load dari SavedModel
# model = tf.keras.models.load_model('saved_model/')

# Load dan preprocessing gambar
img = Image.open('plant_image.jpg').resize((224, 224))
img_array = np.array(img) / 255.0  # Normalisasi
img_array = np.expand_dims(img_array, 0)  # Add batch dimension

# Prediksi
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])
predicted_class = class_names[predicted_class_idx]
confidence = predictions[0][predicted_class_idx]

print(f"Kelas: {predicted_class}")
print(f"Confidence: {confidence:.2%}")

# Tampilkan top-3 predictions
top_3_idx = np.argsort(predictions[0])[-3:][::-1]
print("\nTop 3 Predictions:")
for idx in top_3_idx:
    print(f"  {class_names[idx]}: {predictions[0][idx]:.2%}")
```

### Deployment di Mobile (TFLite)

Untuk aplikasi Android/iOS, gunakan model TFLite:

**Android (Kotlin)**:

```kotlin
val interpreter = Interpreter(loadModelFile())
val input = Array(1) { FloatArray(224 * 224 * 3) }
val output = Array(1) { FloatArray(30) }
interpreter.run(input, output)
```

**iOS (Swift)**:

```swift
let interpreter = try Interpreter(modelPath: modelPath)
try interpreter.invoke()
let output = try interpreter.output(at: 0).data
```

### Deployment di Web (TFJS)

Untuk web application dengan JavaScript:

```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
<script>
  async function predictPlant(imagePath) {
    // Load model
    const model = await tf.loadGraphModel("file://tfjs_model/model.json");

    // Prepare image
    const img = new Image();
    img.src = imagePath;
    await new Promise((resolve) => (img.onload = resolve));

    // Preprocess
    let tensor = tf.browser
      .fromPixels(img)
      .resizeNearestNeighbor([224, 224])
      .toFloat()
      .div(255);
    tensor = tensor.expandDims(0);

    // Predict
    const predictions = model.predict(tensor);
    const classIdx = predictions.argMax(-1).dataSync()[0];
    const confidence = predictions.dataSync()[classIdx];

    console.log(`Plant: ${classNames[classIdx]}`);
    console.log(`Confidence: ${(confidence * 100).toFixed(2)}%`);

    tensor.dispose();
  }
</script>
```

## 📦 Output Model

Proyek ini menghasilkan model dalam berbagai format untuk deployment di berbagai platform:

### 1. SavedModel (TensorFlow Format)

Format standar TensorFlow yang universal dan production-ready.

```
saved_model/
├── fingerprint.pb           # Hash untuk verifikasi integritas model
├── saved_model.pb           # Model graph definition
└── variables/
    ├── variables.index      # Index untuk mengakses variabel
    └── variables.data-*     # Data weight dan bias (terkompresi)
```

**Karakteristik**:

- Format standar untuk production
- Kompatibel dengan TensorFlow Serving
- Ukuran: Besar (full precision)

**Penggunaan**:

```python
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('saved_model/')

# Inference
predictions = model.predict(image_array)
```

### 2. TensorFlow Lite (TFLite)

Dioptimalkan untuk perangkat mobile dan embedded dengan latency rendah.

```
tflite/
└── model.tflite             # Single file model (quantized)
```

**Karakteristik**:

- Ukuran sangat kecil (~2-5MB)
- Latency rendah
- Memory footprint minimal
- Cocok untuk: Android, iOS, Raspberry Pi, Edge devices

**Keuntungan**:

- On-device inference (tidak perlu internet)
- Privacy terjaga (data tidak dikirim ke server)
- Performa real-time

**Penggunaan** (Android):

```java
// TensorFlow Lite Interpreter
Interpreter tflite = new Interpreter(modelBuffer);
float[][] output = new float[1][num_classes];
tflite.run(inputArray, output);
```

### 3. TensorFlow.js (Web Format)

Untuk deployment di browser menggunakan JavaScript.

```
tfjs_model/
├── model.json               # Model definition dan metadata
└── weights*.bin             # Model weights (dapat multiple parts)
```

**Karakteristik**:

- Berjalan di browser (client-side)
- Tidak perlu backend
- Ukuran medium (~5-10MB)

**Penggunaan** (JavaScript):

```javascript
// Load model
const model = await tf.loadGraphModel("file://tfjs_model/model.json");

// Inference
const predictions = model.predict(inputTensor);
```

### 4. Best Model (H5 Format)

Model dalam format Keras H5 - model terbaik dari hasil training.

```
best_model.h5               # Single file model
```

**Karakteristik**:

- Kompatibilitas tinggi
- Loading cepat
- Format legacy namun masih widely used

**Penggunaan**:

```python
model = tf.keras.models.load_model('best_model.h5')
```

---

### Ringkasan Format Model

| Format         | Ukuran  | Latency | Platform       | Use Case               |
| -------------- | ------- | ------- | -------------- | ---------------------- |
| **SavedModel** | ~50MB   | Sedang  | Server/Backend | Production deployment  |
| **TFLite**     | ~2-5MB  | Rendah  | Mobile/IoT     | On-device inference    |
| **TFJS**       | ~5-10MB | Sedang  | Web Browser    | Real-time web app      |
| **H5**         | ~50MB   | Sedang  | Python/Keras   | Development & research |

## Struktur Proyek

```
submission/
├── notebook.ipynb                          # Main notebook dengan full code
├── README.md                               # Dokumentasi proyek (file ini)
├── requirements.txt                        # Dependencies list
├── best_model.h5                           # Model terbaik (H5 format)
│
├── saved_model/                            # SavedModel format (TensorFlow)
│   ├── fingerprint.pb                      # Fingerprint untuk verifikasi
│   ├── saved_model.pb                      # Model definition
│   └── variables/
│       ├── variables.index                 # Index variabel model
│       └── variables.data-00000-of-00001   # Data variabel terkompresi
│
├── tflite/                                 # TensorFlow Lite (Mobile/Edge)
│   └── model.tflite                        # Model untuk Android/iOS/Embedded
│
└── tfjs_model/                             # TensorFlow.js (Web/Browser)
    ├── model.json                          # Model definition untuk web
    └── weights*.bin                        # Bobot model (dapat multi-part)
```

### Penjelasan Output Model

| Format         | Lokasi          | Ukuran | Kegunaan                             |
| -------------- | --------------- | ------ | ------------------------------------ |
| **SavedModel** | `saved_model/`  | Besar  | Production, inference di backend     |
| **TFLite**     | `tflite/`       | Kecil  | Mobile apps, embedded devices        |
| **TFLite**     | `tflite/`       | Kecil  | Mobile apps, embedded devices        |
| **TFJS**       | `tfjs_model/`   | Medium | Web browsers, client-side prediction |
| **H5**         | `best_model.h5` | Besar  | Quick loading, model checkpoints     |

## Technologies & Libraries

| Technology       | Versi    | Fungsi                          |
| ---------------- | -------- | ------------------------------- |
| **TensorFlow**   | 2.x      | Deep Learning framework         |
| **Keras**        | Built-in | High-level API                  |
| **NumPy**        | Latest   | Array operations                |
| **Matplotlib**   | Latest   | Visualization                   |
| **Scikit-learn** | Latest   | Metrics (confusion matrix, etc) |
| **Pillow**       | Latest   | Image processing                |

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

## 👤 Penulis

**M. Sobahus Sururin Ni'am**

- Email: sobahusn27@gmail.com
- Dicoding: [sobahusn](https://www.dicoding.com/users/sobahusn/)

---

## 📝 Lisensi

Proyek ini dibuat sebagai bagian dari Dicoding Bootcamp - Belajar Fundamental Deep Learning.

- **Dataset**: [Kaggle - Plants Classification by marquis03](https://www.kaggle.com/datasets/marquis03/plants-classification)
- **Framework**: TensorFlow/Keras
- **Pre-trained Model**: MobileNetV2 dari ImageNet
