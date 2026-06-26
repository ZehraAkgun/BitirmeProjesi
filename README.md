# Görüntü İşleme Tabanlı Gerçek Zamanlı Kültürel Jestlerin Analizi ve Tercümesi

Bu proje, **Yıldız Teknik Üniversitesi Kimya-Metalurji Fakültesi Matematik Mühendisliği Bölümü** bünyesinde gerçekleştirilen Lisans Bitirme Tezi kapsamında geliştirilmiştir. 

Projenin temel amacı; Türk kültürüne özgü dinamik jest ve mimiklerin, MediaPipe Holistic altyapısı ve hibrit Derin Öğrenme (LSTM) mimarileri kullanılarak gerçek zamanlı olarak sınıflandırılması ve anlamlandırılmasıdır.

# Proje Ekibi
* **Yazar:** Zehra AKGÜN (22052612)
* **Danışman:** Prof. Dr. Arzu TURAN DİNCEL
* **Dönem:** Haziran 2026

# Teknolojiler ve Kütüphaneler
* **Programlama Dili:** Python 3.10+
* **Bilgisayarlı Görü (Computer Vision):** OpenCV, MediaPipe Holistic
* **Derin Öğrenme & Optimizasyon:** TensorFlow, Keras, NumPy
* **Geliştirme Ortamı:** VS Code

# Proje Yapısı (Pipeline)
Proje, modüler bir yazılım mimarisi üzerine inşa edilmiştir:

* `gesture_data_collector.py`: Kamera akışından MediaPipe ile anlık landmark (iskelet noktası) verilerinin toplanması.
* `verify_dataset/`: Toplanan ham koordinat verilerinin kontrolü ve optimizasyonu.
* `train_model/`: LSTM yapay sinir ağı mimarisinin eğitilmesi ve ağırlık matrislerinin kaydedilmesi.
* `realtime_inference/`: Eğitilen modelin canlı kamera görüntüsü üzerinde gerçek zamanlı olarak test edilmesi.

# Kurulum ve Çalıştırma
Projenin çalıştırılabilmesi için gerekli kütüphaneleri yükleyin!:
```bash
pip install opencv-python numpy mediapipe tensorflow