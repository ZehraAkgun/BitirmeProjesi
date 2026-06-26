"""
# ==============================================================================
# MODEL EĞİTİM DOSYASI (VERİ ÖN İŞLEME + LSTM MİMARİSİ VE EĞİTİM PROTOKOLÜ)
# ==============================================================================
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import random

np.random.seed(42)
tf.random.set_seed(42)
random.seed(42) 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (confusion_matrix,
                                     classification_report,
                                     ConfusionMatrixDisplay)

from tensorflow.keras.models      import Sequential
from tensorflow.keras.layers      import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers  import Adam
from tensorflow.keras.utils       import to_categorical
from tensorflow.keras.callbacks   import (EarlyStopping,
                                          ModelCheckpoint,
                                          ReduceLROnPlateau,
                                          TensorBoard)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 1 — YAPILANDIRMA SABİTLERİ
# ══════════════════════════════════════════════════════════════════════════════

# ─── VERİ YOLU ────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join("MP_Data")

# Topladığım ve eğiteceğim kültürel jest sınıflarının tam listesi
actions = np.array([
    "Allah_korusun",
    "bilmem",
    "culsuzum",
    "hayir",
    "iki_cay",
    "kapak",
    "kusmek",
    "odum_koptu",
    "ohoo",
    "para"
])

# ─── VERİ PARAMETRELERI ────────────────────────────────────────────────────────
NO_SEQUENCES    = 30    # Her jest için toplam video tekrar sayısı
SEQUENCE_LENGTH = 30    # Her video için kare sayısı
FEATURE_DIM     = 1662  # Tek kare landmark vektörü girdi boyutu

# ─── EĞİTİM PARAMETRELERİ ─────────────────────────────────────────────────────
# hiperparametreler
EPOCHS          = 200
BATCH_SIZE      = 16
LEARNING_RATE   = 1e-3
VALIDATION_SPLIT= 0.20   # %80 eğitim, %20 doğrulama

# ─── ÇIKTI DOSYALARI ──────────────────────────────────────────────────────────
MODEL_SAVE_PATH = "gesture_lstm_model.keras"   
BEST_MODEL_PATH = "gesture_lstm_best.keras"    # EarlyStopping için en iyi yakaladığıa ğırlıklar
PLOT_SAVE_DIR   = "training_plots"
os.makedirs(PLOT_SAVE_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 2 — VERİ YÜKLEYİCİ (load_dataset)
# ══════════════════════════════════════════════════════════════════════════════

def load_dataset(data_path: str, actions: np.ndarray,
                 no_sequences: int, sequence_length: int) -> tuple:
    """
    MP_Data dizin yapısından tüm sekans örneklerini okur ve birleştirir.

    ── Mevcut Dizin Yapısına Göre Yükleme ───────────────────────────────────
    Veri toplama aşamasında her kare ayrı bir .npy dosyası olarak aşağıdaki gibi kaydedildi:

        MP_Data/
        └── ALLAH korusun/
            ├── 0/
            │   ├── 0.npy   → (1662,) vektör
            │   ├── 1.npy
            │   └── 29.npy
            ├── 1/
            └── 29/

    Bu fonksiyon, her sekans klasöründeki 30 kareyi sıralı olarak okuyarak
    (SEQUENCE_LENGTH, FEATURE_DIM) = (30, 1662) boyutunda bir tensöre
    dönüştürür. Bu tensör, LSTM modelinin beklediği zaman serisi formatıdır.

    ── Vektör Boyutu Karşılaştırması ────────────────────────────────────────
    Farklı çalışmalarla karşılaştırmak için özellik boyutu notu:
        • Bu çalışma: pose(132)+face(1404)+el(126) = 1662 boyut (tam vücut)
        • Yalnızca el tabanlı sistemler: 21×3×2 = 126 boyut (Molchanov, 2016)
        • Tam vücut temsili, mimik+jest kombinasyonu gerektiren
          Türk kültürel hareketleri için seçilmiştir (Escalera vd., 2013).

    ── Hata Yönetimi ─────────────────────────────────────────────────────────
    Eksik kare dosyası (bozuk kayıt vb.) tespit edildiğinde, ilgili sekans
    veri setine dahil edilmez ve konsola uyarı mesajı verilir. Bu yaklaşım,
    kısmi kayıpların model eğitimini bozmasını önler.

    Args:
        data_path:       MP_Data klasörünün yolu.
        actions:         Jest isimlerini içeren NumPy dizisi.
        no_sequences:    Her jest için tekrar sayısı.
        sequence_length: Her tekrardaki kare sayısı.

    Returns:
        X (np.ndarray): Şekil (N, 30, 1662) — giriş tensörü.
        y (np.ndarray): Şekil (N,)           — ham tamsayı etiketler.
        label_map (dict): Sınıf adı → tamsayı indeks eşlemesi.
    """
    sequences = []  # Her biri (30, 1662) olan listeler
    labels    = []  # Karşılık gelen tamsayı sınıf etiketleri

    skipped = 0

    for label_idx, action in enumerate(actions):
        for seq_num in range(no_sequences):
            window = []  # Tek bir sekansın kareleri

            seq_ok = True
            for frame_num in range(sequence_length):
                frame_path = os.path.join(
                    data_path, action, str(seq_num), f"{frame_num}.npy"
                )

                if not os.path.exists(frame_path):
                    print(f"  [!] Eksik kare: {frame_path} — sekans atlandı.")
                    seq_ok = False
                    skipped += 1
                    break

                frame_vector = np.load(frame_path)   # → (1662,)

                # Boyut doğrulama: beklenen 1662, başka bir şeyse uyar
                if frame_vector.shape[0] != FEATURE_DIM:
                    print(f"  [!] Boyut uyuşmazlığı: {frame_path} → "
                          f"beklenen {FEATURE_DIM}, bulunan {frame_vector.shape[0]}")
                    seq_ok = False
                    break

                window.append(frame_vector)

            if seq_ok and len(window) == sequence_length:
                sequences.append(window)
                labels.append(label_idx)

    if skipped > 0:
        print(f"\n  [⚠] Toplam {skipped} kare eksikti; ilgili sekanslar atlandı.")

    X = np.array(sequences, dtype=np.float32)  # → (N, 30, 1662)
    y = np.array(labels,    dtype=np.int32)     # → (N,)

    label_map = {action: idx for idx, action in enumerate(actions)}

    print(f"\n[✓] Veri yükleme tamamlandı:")
    print(f"    X boyutu : {X.shape}   (N_örnek, zaman, özellik)")
    print(f"    y boyutu : {y.shape}")
    print(f"    Sınıflar : {label_map}\n")

    return X, y, label_map


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 3 — VERİ ÖN İŞLEME
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(X: np.ndarray, y: np.ndarray,
               num_classes: int, val_split: float = 0.20) -> tuple:
    """
    Topladığım ham veri tensörlerini LSTM modelimin işleyebileceği standarda getiriyorum.

    ── One-Hot Encoding Yaklaşımım ─────────────────────────────────────────────
    Tamsayı sınıf etiketlerini (0, 1, 2, ...) One-Hot vektörlerine dönüştürüyorum.
    Örnek (3 kültürel sınıf için):
        'eyvallah' → [1, 0, 0]
        'nah'      → [0, 1, 0]
        'cik'      → [0, 0, 1]

    Bu dönüşümü tercih etmemin iki temel akademik/teknik nedeni var:
    (1) Kültürel jestlerimiz arasında herhangi bir hiyerarşi veya sıralama yoktur. 
        Tamsayı temsili bıraksaydım model sınıflar arasında (2 > 1 > 0) gibi yapay 
        bir büyüklük ilişkisi kurup yanlılığa (bias) sürüklenebilirdi.
    (2) Modelimin çıkış katmanında kullandığım Softmax aktivasyonu, her düğüm için 
        bağımsız bir olasılık üretir; bu yapı One-Hot formatıyla doğrudan örtüşmektedir.
        (Goodfellow vd., 2016 — Deep Learning, MIT Press, s. 179)

    ── Train/Test Ayrımı ─────────────────────────────────────────────────────
    Stratified (tabakalı) rastgele bölme kullanılır: her sınıfın eğitim ve
    test kümelerindeki oranı korunur. Bu, sınıf dağılımı dengesizliğinin
    test performansını yanıltmasını önler (Pedregosa vd., 2011).

    Args:
        X:           (N, 30, 1662) ham veri tensörü.
        y:           (N,) tamsayı etiket dizisi.
        num_classes: Toplam jest sınıfı sayısı.
        val_split:   Doğrulama kümesi oranı (varsayılan 0.20).

    Returns:
        X_train, X_val, y_train, y_val: Bölünmüş ve kodlanmış veri kümeleri.
    """
    # One-Hot Encoding
    y_cat = to_categorical(y, num_classes=num_classes).astype(np.float32)

    # Stratified train/val bölmesi
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat,
        test_size=val_split,
        random_state=42,       # Tekrarlanabilirlik için sabit tohum
        stratify=y             # Tabakalı bölme — sınıf dengesini koruyan parametre
    )

    print(f"[✓] Ön işleme tamamlandı:")
    print(f"    Eğitim seti  : X={X_train.shape}  y={y_train.shape}")
    print(f"    Doğrulama    : X={X_val.shape}    y={y_val.shape}\n")

    return X_train, X_val, y_train, y_val


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 4 — LSTM MODELİ
# ══════════════════════════════════════════════════════════════════════════════

def build_lstm_model(input_shape: tuple, num_classes: int) -> Sequential:
    """
    Türk kültürel jestlerini gerçek zamanlı tanımak amacıyla tasarlanılan
    LSTM tabanlı derin öğrenme mimarisi

    ── Mimari Gerekçesi (Tez Metodoloji Bölümü İçin) ────────────────────────
    Dinamik jestler, zaman boyutunda bağımlılık içerir:
    'el öpüp başa koyma' jestinin üç alt aşaması (eğilme → öpme → koyma)
    sıralı ve nedenseldir; bu nedenle Uzun Kısa Süreli Bellek (Long Short-Term
    Memory — LSTM) ağları tercih edilmiştir. LSTM'nin unutma (forget), giriş
    (input) ve çıkış (output) kapı mekanizmaları, standart tekrarlayan sinir
    ağlarının (RNN) "kaybolan gradient" sorununu çözerek uzun vadeli
    bağımlılıkların öğrenilmesini sağlar (Hochreiter & Schmidhuber, 1997).

    ── Katman Yapısı ─────────────────────────────────────────────────────────

    [Girdi]  →  (30, 1662)   zaman serisi

    [LSTM-1] →  128 birim, return_sequences=True
        • return_sequences=True: Her zaman adımının çıktısını bir sonraki
          LSTM katmanına aktarır. Bu, çok katmanlı LSTM için zorunludur.
        • 128 birim: Sıralar arası karmaşık bağımlılıkları yakalamak için
          yeterli kapasiteyi sağlar; 256'ya çıkarılabilir (ablasyon).

    [Dropout-1] → oran=0.3
        • Eğitim sırasında nöronların %30'unu rastgele devre dışı bırakır.
        • Overfitting'i azaltır: model belirli nöronlara bağımlı olamaz.
        • Referans: Srivastava vd. (2014).

    [LSTM-2] →  64 birim, return_sequences=False
        • return_sequences=False: Yalnızca son zaman adımının çıktısını
          döndürür; bu, sınıflandırma (classification) görevi için standart
          yapıdır. Modelin tüm sekansı özetlemesini zorunlu kılar.

    [BatchNorm] → eğitim kararlılığı ve hız için
        • Her mini-batch'in ortalamasını ve varyansını normalleştirir.
        • Learning rate'e duyarlılığı azaltır (Ioffe & Szegedy, 2015).

    [Dropout-2] → oran=0.2

    [Dense-1]  →  64 nöron, ReLU aktivasyonu
        • Sınıflar arası doğrusal olmayan karar sınırları öğrenir.

    [Dense-2]  →  num_classes nöron, Softmax aktivasyonu
        • Her jest sınıfı için [0,1] aralığında olasılık üretir.
        • Tüm olasılıkların toplamı = 1 (Goodfellow vd., 2016).

    Args:
        input_shape: LSTM girdi boyutu → (SEQUENCE_LENGTH, FEATURE_DIM)
                     Bu çalışmada (30, 1662).
        num_classes: Tanınacak jest sınıfı sayısı.

    Returns:
        model (Sequential): Derlenmemiş Keras modeli.
    """
    model = Sequential(name="TurkishGesture_LSTM_v1")

    # ── LSTM Katman 1 ──────────────────────────────────────────────────────────
    model.add(LSTM(
        units=128,
        return_sequences=True,   # Bir sonraki LSTM katmanına tüm zaman adımları
        activation="tanh",       # LSTM'nin varsayılan ve literatürde standart kabul edilen ve önerilen aktivasyonu
        input_shape=input_shape,
        name="lstm_1"
    ))
    model.add(Dropout(0.3, name="dropout_1"))

    # ── LSTM Katman 2 ──────────────────────────────────────────────────────────
    model.add(LSTM(
        units=64,
        return_sequences=False,  # Yalnızca son zaman adımı → sınıflandırma için
        activation="tanh",
        name="lstm_2"
    ))

    # ── Batch Normalization ────────────────────────────────────────────────────
    model.add(BatchNormalization(name="batch_norm"))
    model.add(Dropout(0.2, name="dropout_2"))

    # ── Tam Bağlantılı (Dense) Katmanlar ──────────────────────────────────────
    model.add(Dense(64, activation="relu", name="dense_1"))
    model.add(Dense(num_classes, activation="softmax", name="output"))

    return model


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 5 — EĞİTİM CALLBACK'LERİ
# ══════════════════════════════════════════════════════════════════════════════

def build_callbacks(best_model_path: str) -> list:
    """
    Model eğitimini dinamik olarak izlemek ve kontrol etmek için callback nesneleri döndürür.

    ── EarlyStopping (Erken Durdurma) Stratejisi ─────────────────────────────────────────────────────────
    Doğrulama kaybı (val_loss) art arda 'patience' epoch boyunca iyileşmezse
    eğitimi durdurur. Bu mekanizmanın iki işlevi vardır:
    (1) Overfitting'i önler: model eğitim verilerini ezberlemeden durur.
    (2) Hesaplama kaynağı tasarrufu sağlar.
    restore_best_weights=True: Eğitim sonunda en iyi doğrulama skoruna
    karşılık gelen ağırlıklar geri yüklenir.
    (Yao vd., 2007 — erken durdurma tekniğinin teorik temeli)

    ── ModelCheckpoint (En İyiyi Yakalama)───────────────────────────────────────────────────────
    Her epoch sonunda val_loss iyileşmişse modeli diske kaydeder.
    Bu, uzun eğitimlerde çökme gibi beklenmedik durumlarda veri kaybını önler.

    ── ReduceLROnPlateau (Öğrenme Hızı Optimizasyonu) ─────────────────────────────────────────────────────
    Doğrulama kaybı belirli bir süre iyileşmediğinde öğrenme hızını
    'factor' oranında azaltır. Bu uyarlamalı öğrenme hızı stratejisi,
    sabit öğrenme hızına kıyasla genellikle daha iyi yakınsama sağlar.
    (Schaul vd., 2013 — hiperparametre optimizasyonu)

    Args:
        best_model_path: En iyi modelin kaydedileceği dosya yolu.

    Returns:
        list: Keras callback nesnelerinin listesi.
    """
    early_stop = EarlyStopping(
        monitor="val_loss",
        patience=25,
        restore_best_weights=True,
        verbose=1
    )

    checkpoint = ModelCheckpoint(
        filepath=best_model_path,
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,       # Öğrenme hızını yarıya indir
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    tensorboard = TensorBoard(
        log_dir="logs/",
        histogram_freq=1    # Her epoch sonunda ağırlık histogramları
    )

    return [early_stop, checkpoint, reduce_lr, tensorboard]


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 6 — AKADEMİK GÖRSEL ÇIKTILAR
# ══════════════════════════════════════════════════════════════════════════════

def plot_training_history(history, save_dir: str):
    """
    Eğitim geçmişini (accuracy & loss) tez 'Bulgular' bölümü için çizeriz.

    ── Grafik Yorumlama Rehberi ─────────────────────────────────
    Eğitim/doğrulama eğrileri arasındaki boşluk, modelin genelleşme
    kapasitesi hakkında bilgi verir

    • Küçük boşluk      → iyi genelleşme (underfitting/overfitting yok)
    • Eğitim >> Doğrulama → overfitting (Dropout artır veya veri artır)
    • Her ikisi düşük   → underfitting (model kapasitesini artır)


    Args:
        history: model.fit() dönüşünden elde edilen History nesnesi.
        save_dir: Grafiklerin kaydedileceği dizin.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "LSTM Model Eğitim Performansı — Türk Kültürel Jest Tanıma",
        fontsize=13, fontweight="bold", y=1.02
    )

    epochs_range = range(1, len(history.history["loss"]) + 1)

    # ── Kayıp (Loss) Grafiği ──────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(epochs_range, history.history["loss"],
             label="Eğitim Kaybı",     color="#E63946", linewidth=2)
    ax1.plot(epochs_range, history.history["val_loss"],
             label="Doğrulama Kaybı",  color="#457B9D",
             linewidth=2, linestyle="--")
    ax1.set_title("Categorical Cross-Entropy Kaybı", fontsize=11)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Kayıp (Loss)")
    ax1.legend(loc="upper right")
    ax1.set_xlim(1, len(epochs_range))

    # ── Doğruluk (Accuracy) Grafiği ───────────────────────────────────────────
    ax2 = axes[1]
    acc_key     = "categorical_accuracy"
    val_acc_key = "val_categorical_accuracy"

    ax2.plot(epochs_range, history.history[acc_key],
             label="Eğitim Doğruluğu",    color="#E63946", linewidth=2)
    ax2.plot(epochs_range, history.history[val_acc_key],
             label="Doğrulama Doğruluğu", color="#457B9D",
             linewidth=2, linestyle="--")
    ax2.set_title("Sınıflandırma Doğruluğu", fontsize=11)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Doğruluk (Accuracy)")
    ax2.legend(loc="lower right")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(1, len(epochs_range))

    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_history.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Eğitim grafiği kaydedildi → {save_path}")
    plt.show()


def plot_confusion_matrix(model, X_val: np.ndarray, y_val: np.ndarray,
                          actions: np.ndarray, save_dir: str):
    """
    Doğrulama kümesi üzerinde karmaşıklık matrisi (confusion matrix) oluşturur.

    ── Proje İçin Önemi ────────────────────────────────────────────────────────
    Karmaşıklık matrisi, hangi jest çiftlerinin birbiriyle karıştırıldığını
    görselleştirir. Bu, sistemin hata profilini ortaya koyar. Örneğin 'cik' ve
    'omuz_silkme' hareketlerinin karıştırılması, kaş ve baş hareketlerinin
    benzer landmark değişimlerine yol açmasından kaynaklanıyor olabilir.

    Metrikler:
    • Precision: TP / (TP + FP) — tahmin doğruluğu
    • Recall:    TP / (TP + FN) — saptama kapsamı
    • F1-Score:  2·P·R / (P+R)  — dengeli harmonic ortalama
    Bkz. Powers (2011) — Evaluation: From Precision, Recall and F-Measure...

    Args:
        model:    Eğitilmiş Keras modeli.
        X_val:    Doğrulama giriş tensörü.
        y_val:    One-Hot kodlu doğrulama etiketleri.
        actions:  Sınıf isim dizisi.
        save_dir: Kayıt dizini.
    """
    # One-Hot → tamsayı etiket
    y_true = np.argmax(y_val,            axis=1)
    y_pred = np.argmax(model.predict(X_val), axis=1)

    # Sınıflandırma raporu (precision, recall, F1)
    print("\n" + "═" * 60)
    print("  SINIFLANDIRMA RAPORU (Doğrulama Kümesi)")
    print("═" * 60)
    report = classification_report(y_true, y_pred, target_names=actions)
    print(report)

    # Raporu teze eklemek için dosyaya kaydet
    report_path = os.path.join(save_dir, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Sınıflandırma Raporu — Türk Kültürel Jest Tanıma\n")
        f.write("=" * 60 + "\n")
        f.write(report)
    print(f"[✓] Sınıflandırma raporu kaydedildi → {report_path}")

    # Karmaşıklık Matrisi Görseli
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(max(6, len(actions)), max(5, len(actions))))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[a.replace("_", "\n") for a in actions]
    )
    disp.plot(
        ax=ax, cmap="Blues",
        colorbar=True,
        xticks_rotation=45
    )
    ax.set_title(
        "Karmaşıklık Matrisi — Doğrulama Kümesi\n"
        "Türk Kültürel Jest Tanıma (LSTM)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    cm_path = os.path.join(save_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    print(f"[✓] Karmaşıklık matrisi kaydedildi → {cm_path}")
    plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# BÖLÜM 7 — ANA FONKSİYON
# ══════════════════════════════════════════════════════════════════════════════

def main():
    """
    Veri yükleme → Ön işleme → Model oluşturma → Eğitim → Değerlendirme
    adımlarını sırasıyla çalıştırır.

    ── Tekrarlanabilirlik (Reproducibility) İlkesi-─────────────────────────────────
    Akademik çalışmalarda sonuçların bağımsız araştırmacılar tarafından
    tekrarlanabilmesi zorunludur. Bunun için:
    - random_state=42 (scikit-learn bölmesi)
    - numpy / tensorflow rastgele tohum (seed) sabitlenmesi önerilir.
    """
    # Tekrarlanabilirlik tohumları
    np.random.seed(42)
    import tensorflow as tf
    tf.random.set_seed(42)

    num_classes = len(actions)
    print(f"\n{'═'*60}")
    print(f"  Türk Kültürel Jest Tanıma — LSTM Eğitim Pipeline")
    print(f"  Sınıf sayısı: {num_classes} | Epoch: {EPOCHS} | Batch: {BATCH_SIZE}")
    print(f"{'═'*60}\n")

    # ── Aşama 3a: Veri Yükleme ────────────────────────────────────────────────
    print("[1/5] Veri yükleniyor...")
    X, y, label_map = load_dataset(
        DATA_PATH, actions, NO_SEQUENCES, SEQUENCE_LENGTH
    )

    # ── Aşama 3b: Ön İşleme ───────────────────────────────────────────────────
    print("[2/5] Ön işleme uygulanıyor...")
    X_train, X_val, y_train, y_val = preprocess(X, y, num_classes, VALIDATION_SPLIT)

    # ── Aşama 4a: Model Oluşturma ─────────────────────────────────────────────
    print("[3/5] LSTM modeli oluşturuluyor...")
    model = build_lstm_model(
        input_shape=(SEQUENCE_LENGTH, FEATURE_DIM),
        num_classes=num_classes
    )

    # Adam optimizer: Uyarlamalı momentum tabanlı gradient iniş algoritması.
    # Varsayılan lr=1e-3 jest tanıma literatüründe yaygın başlangıç noktasıdır.
    # Referans: Kingma & Ba (2015).
    optimizer = Adam(learning_rate=LEARNING_RATE)

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        # categorical_crossentropy: çok sınıflı One-Hot kodlama için standart
        # kayıp fonksiyonu. Modelin her sınıf için ürettiği log-olasılığı
        # cezalandırır. (Goodfellow vd., 2016, s.178)
        metrics=["categorical_accuracy"]
    )

    model.summary()

    # ── Aşama 4b: Eğitim ──────────────────────────────────────────────────────
    print(f"\n[4/5] Model eğitimi başlıyor ({EPOCHS} epoch, batch={BATCH_SIZE})...")
    callbacks = build_callbacks(BEST_MODEL_PATH)

    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    # ── Model Kaydetme ────────────────────────────────────────────────────────
    # .keras formatı: TF 2.12+ için önerilen standart format.
    # Hem model mimarisini hem ağırlıkları hem de optimizer durumunu korur.
    # Bu format, gerçek zamanlı tanıma scriptinde tf.keras.models.load_model()
    # ile doğrudan yüklenebilir.
    model.save(MODEL_SAVE_PATH)
    print(f"\n[✓] Son model kaydedildi  → {MODEL_SAVE_PATH}")
    print(f"[✓] En iyi model kaydedildi → {BEST_MODEL_PATH}")

    # Etiket haritasını da kaydet (gerçek zamanlı tanıma için gerekli)
    import json
    with open("label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)
    print("[✓] Etiket haritası kaydedildi → label_map.json\n")

    # ── Aşama 5: Akademik Görsel Çıktılar ─────────────────────────────────────
    print("[5/5] Akademik görseller üretiliyor...")
    plot_training_history(history, PLOT_SAVE_DIR)
    plot_confusion_matrix(model, X_val, y_val, actions, PLOT_SAVE_DIR)

    # ── Özet Metrikleri ────────────────────────────────────────────────────────
    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"\n{'═'*60}")
    print(f"  SONUÇ ÖZETİ")
    print(f"  Doğrulama Kaybı    : {val_loss:.4f}")
    print(f"  Doğrulama Doğruluğu: {val_acc:.4f}  ({val_acc*100:.1f}%)")
    print(f"  Eğitim süresi      : {len(history.history['loss'])} epoch")
    print(f"{'═'*60}\n")

    return model, history


# ══════════════════════════════════════════════════════════════════════════════
# GİRİŞ NOKTASI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    """
    Çalıştırmak için terminalde:
        python train_model.py

    Beklenen çıktılar:
        gesture_lstm_model.keras    → son epoch modeli
        gesture_lstm_best.keras     → en iyi val_loss modeli
        label_map.json              → sınıf isim ↔ indeks eşlemesi
        training_plots/
            training_history.png    → acc & loss grafikleri 
            confusion_matrix.png    → karmaşıklık matrisi
            classification_report.txt

    """
    model, history = main()