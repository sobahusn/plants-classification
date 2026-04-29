# ============================================================
# Import Library
# - streamlit : framework untuk membuat web app interaktif
# - numpy     : operasi array/matrix untuk data gambar
# - PIL       : membuka dan memanipulasi file gambar
# - tensorflow: memuat dan menjalankan model deep learning
# - pandas    : membuat tabel/dataframe untuk visualisasi
# ============================================================
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import os

# Path model relatif terhadap lokasi file main.py ini
MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "model", "saved_model"
)


# ============================================================
# Daftar nama kelas sesuai urutan saat training
# Urutan ini HARUS sama persis dengan urutan folder dataset
# saat model dilatih (alphabetical order)
# ============================================================
class_names = [
    "aloevera",
    "banana",
    "bilimbi",
    "cantaloupe",
    "cassava",
    "coconut",
    "corn",
    "cucumber",
    "curcuma",
    "eggplant",
    "galangal",
    "ginger",
    "guava",
    "kale",
    "longbeans",
    "mango",
    "melon",
    "orange",
    "paddy",
    "papaya",
    "peperchili",
    "pineapple",
    "pomelo",
    "shallot",
    "soybeans",
    "spinach",
    "sweetpotatoes",
    "tobacco",
    "waterapple",
    "watermelon",
]


# ============================================================
# @st.cache_resource memastikan model hanya dimuat SEKALI
# saat aplikasi pertama kali dijalankan, bukan setiap kali
# user upload gambar. Ini menghemat waktu dan memori.
# ============================================================
@st.cache_resource
def load_model():
    # tf.saved_model.load memuat model dari folder saved_model
    return tf.saved_model.load(MODEL_PATH)


def main():
    # --------------------------------------------------------
    # Judul dan deskripsi aplikasi
    # st.title  : teks besar di bagian atas halaman
    # st.write  : teks biasa / markdown
    # --------------------------------------------------------
    st.title("Plants Classification")
    st.write("Upload an image of a plant to classify it.")

    # Muat model (hanya sekali karena ada cache_resource)
    model = load_model()

    # --------------------------------------------------------
    # Widget upload gambar
    # type=["jpg","jpeg","png"] membatasi format yang diterima
    # Mengembalikan None jika belum ada file yang diupload
    # --------------------------------------------------------
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        # Buka gambar dan konversi ke RGB
        # .convert("RGB") penting agar gambar PNG dengan
        # transparansi (RGBA) tidak menyebabkan error
        image = Image.open(uploaded_file).convert("RGB")

        # --------------------------------------------------------
        # st.columns(2) membagi halaman menjadi 2 kolom sejajar
        # col1 = kolom kiri, col2 = kolom kanan
        # --------------------------------------------------------
        col1, col2 = st.columns(2)
        with col1:
            # Tampilkan gambar yang diupload di kolom kiri
            st.image(image, caption="Uploaded Image", use_container_width=True)

        # --------------------------------------------------------
        # Preprocessing gambar sebelum dimasukkan ke model:
        # 1. resize ke (224, 224) sesuai input size model
        # 2. ubah ke numpy array
        # 3. expand_dims menambah dimensi batch: (224,224,3) -> (1,224,224,3)
        # 4. cast ke float32
        # CATATAN: TIDAK perlu dibagi 255 karena model sudah
        # punya layer Rescaling(1/255) di dalamnya
        # --------------------------------------------------------
        img = np.expand_dims(np.array(image.resize((224, 224))), axis=0).astype(
            "float32"
        )

        # --------------------------------------------------------
        # Jalankan inferensi menggunakan signature "serving_default"
        # Output berupa dict, kita ambil key "output_0"
        # .flatten() mengubah array (1, 30) menjadi (30,)
        # sehingga mudah diakses per index
        # --------------------------------------------------------
        output = model.signatures["serving_default"](tf.constant(img, dtype=tf.float32))
        prediction = output["output_0"].numpy().flatten()

        # np.argmax mengambil index dengan nilai probabilitas tertinggi
        pred_class = int(np.argmax(prediction))
        confidence = float(prediction[pred_class])

        with col2:
            # st.success menampilkan kotak hijau dengan teks hasil prediksi
            st.success(f"Hasil Klasifikasi: **{class_names[pred_class].title()}**")
            # st.metric menampilkan angka besar dengan label di atasnya
            st.metric("Confidence", f"{confidence * 100:.2f}%")

        # --------------------------------------------------------
        # Visualisasi Top 5 prediksi sebagai bar chart
        # np.argsort mengurutkan index dari kecil ke besar,
        # [::-1] membalik urutannya (besar ke kecil),
        # [:5] mengambil 5 teratas
        # --------------------------------------------------------
        st.write("### Top 5 Predictions")
        top5_idx = np.argsort(prediction)[::-1][:5]
        df = pd.DataFrame(
            {
                "Class": [class_names[i].title() for i in top5_idx],
                "Confidence (%)": [
                    round(float(prediction[i]) * 100, 2) for i in top5_idx
                ],
            }
        ).set_index("Class")

        # st.bar_chart menampilkan bar chart dari DataFrame
        st.bar_chart(df)

    else:
        # st.warning menampilkan kotak kuning sebagai peringatan
        st.warning("Please upload an image to classify.")


# Titik masuk aplikasi — hanya dijalankan jika file ini
# dieksekusi langsung (bukan di-import sebagai modul)
if __name__ == "__main__":
    main()
