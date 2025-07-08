# 🛣️ Road Damage Detection with YOLOv8

Proyek ini merupakan implementasi **Object Detection** menggunakan **YOLOv8** untuk mendeteksi berbagai jenis kerusakan jalan (retak, lubang, dan lainnya) dari gambar. Proyek ini dikembangkan menggunakan **Google Colab** dan bertujuan untuk mendukung pemantauan kondisi jalan secara otomatis berbasis AI.

---

## 📌 Fitur Utama

* Deteksi kerusakan jalan dengan **YOLOv8**
* Visualisasi distribusi data dan ukuran gambar
* Train-test split otomatis
* Hyperparameter tuning melalui file YAML
* Simpan dan jalankan model untuk prediksi

---

## 📁 Struktur Proyek

```
road-damage-detection/
├── dataset/
│   ├── images/
│   └── labels/
├── runs/
├── data.yaml
├── bhypa.yaml
├── Road_Damage_Detection_Yolov8.ipynb
└── README.md
```

---

## 🔗 Dataset

Dataset dapat diunduh dari link berikut:

[Road Damage Dataset](https://drive.google.com/file/d/191mgRxzIvNGlEwk6fM-WRRfsYzg3yfZP/view?usp=sharing)

Letakkan file ZIP ke dalam Google Drive Anda dan ekstrak di lokasi:

```
/content/drive/MyDrive/Dataset/Project_Data_Science/new_data/road_damage_dataset.zip
```

---

## ⚙️ Instalasi dan Dependensi

### 1. Clone Repo (opsional)

```bash
https://github.com/Rizkyard17/Road-damage-detection.git
cd road-damage-detection
```

### 2. Instalasi Library

Jalankan di Google Colab:

```python
!pip install ultralytics
```

Atau jika lokal (opsional):

```bash
pip install ultralytics pillow matplotlib seaborn pandas
```

📄 **requirements.txt** (jika diperlukan untuk local setup):

```
ultralytics
pillow
matplotlib
seaborn
pandas
ipython
```

---

## 🚀 Cara Kerja Program

### 1. Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive/')
```

### 2. Ekstrak Dataset

```python
!unzip -q /content/drive/MyDrive/Dataset/Project_Data_Science/new_data/road_damage_dataset.zip -d /content/dataset/
```

### 3. Exploratory Data Analysis (EDA)

* Visualisasi distribusi kelas bounding box
* Analisis ukuran gambar dan outlier

### 4. Preprocessing:

* Generate file `data.yaml` otomatis
* Buat hyperparameter file `bhypa.yaml`
* Train-test split gambar dan label ke folder `train/` dan `val/`

### 5. Training Model

```python
from ultralytics import YOLO
model = YOLO("yolov8s.pt")
model.train(
    data="/content/dataset/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    cfg="bhypa.yaml",
    name="gridSearch"
)
```

### 6. Save Model ke Google Drive

```python
shutil.copytree(source, destination)
```

### 7. Test Model dengan Gambar Baru

* Upload gambar menggunakan:

```python
from google.colab import files
uploaded = files.upload()
```

* Jalankan prediksi:

```python
model = YOLO('/content/runs/detect/gridSearch/weights/best.pt')
results = model.predict(source=filename, save=True, imgsz=640, conf=0.3)
```

---

## 📊 Visualisasi yang Dihasilkan

✅ Distribusi Kelas Label
✅ Distribusi Ukuran Lebar dan Tinggi Gambar
✅ Distribusi Ukuran Resolusi
✅ Visualisasi Bounding Box Hasil Prediksi

---

## ⚙️ Hyperparameter

File `bhypa.yaml` digunakan untuk mengatur hyperparameter training YOLOv8.

Contoh:

```yaml
lr0: 0.001
lrf: 0.001
momentum: 0.8
weight_decay: 0.0001
box: 0.05
cls: 0.7
```

---

## 💡 Catatan Penting

* Pastikan folder `images` dan `labels` sudah dalam format YOLO (`.txt`).
* Semua file dan folder berada di path yang sesuai di Google Drive atau local.
* Proses training cukup berat, disarankan menggunakan GPU runtime di Google Colab.

---

## 📌 To Do (Pengembangan Selanjutnya)

* Deploy ke web menggunakan **Streamlit** atau **Flask**
* Implementasi pipeline prediksi video real-time

---
