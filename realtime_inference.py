"""
Yıldız Teknik Üniversitesi - Matematik Mühendisliği Bölümü
Bitirme Çalışması: Görüntü İşleme Tabanlı Gerçek Zamanlı Kültürel Jest Analizi
Modül: Gerçek Zamanlı Çıkarım ve Çift Dilli HUD Gösterimi (realtime_inference.py)

Geliştirici: Zehra Akgün (22052612)
Danışman: Prof. Dr. Arzu TURAN DİNCEL
"""

import cv2
import numpy as np
import json
import collections
import mediapipe as mp
from tensorflow.keras.models import load_model

# ==============================================================================
# 1. YAPILANDIRMA VE SÖZLÜK AYARLARI
# ==============================================================================

MODEL_PATH = "gesture_lstm_best.keras"
LABEL_MAP_PATH = "label_map.json"

# Zaman serisi ardışıklığı için modelin eğitime girdi olarak aldığı pencere boyutu (T=30)
SEQUENCE_LENGTH = 30  
# Girdi uzayının toplam boyutu (Pose:132 + Face:1404 + LH:63 + RH:63 = 1662)
FEATURE_DIM = 1662  

# Sınıflandırma güvenilirlikk ve pürüzlendirme eşikleri
CONFIDENCE_THRESHOLD = 0.90
SMOOTHING_WINDOW = 10
WINDOW_NAME = "Turk Kulturel Jest Tanima Sistemi / Bilingual Inference HUD"

# ── ÇİFT DİLLİ JEST SÖZLÜĞÜ ──────────────────────────────────────────────────
# train_model.py içindeki ham klasör isimlerini eksiksiz buraya işliyorum.
# Kültürel jestlerin arayüzde hem yerel hem de uluslararası literatüre uygun 
# Erasmus öğrencilerine çevirmek için Türkçe açıklamaları ve İngilizce deyimsel tercümelerini ekledim
JEST_SOZLUGU = {
    "iki_cay": ("Iki Cay", "Two Teas"),
    "bilmem": ("Bilmem ", "I Don't Know"),
    "hayir": ("Hayir (Cik)", "No / Disapproval"),
    "para": ("Para", "Money"),
    "kusmek": ("Kusmek", "Sulking"),
    "odum_koptu": ("Odum Koptu", "I Got Scared!"),
    "ohoo": ("Ohoo", "Yeah, right!")
}

# MediaPipe grafik tabanlı iskelet çıkarım bileşenleri
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles


# ==============================================================================
# 2. VERİ YÜKLEME VE MATRİS İŞLEMLERİ
# ==============================================================================

def load_resources():
    """ Diskten eğitilmiş LSTM model mimarisini ve sınıf indeks haritasını yükler. """
    model = load_model(MODEL_PATH)
    
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    # Model çıktı indekslerini (0, 1, 2...) metinsel sınıflara geri eşliyoruz
    idx2label = {v: k for k, v in label_map.items()}
    print(f"[OK] Model başarıyla yüklendi: {MODEL_PATH}")
    print(f"[OK] Aktif Sınıf Listesi: {list(idx2label.values())}\n")
    return model, idx2label


def mediapipe_detection(frame, model):
    """ OpenCV BGR matrisini MediaPipe için RGB uzayına taşır ve çıkarım yapar. """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False  # Bellek optimizasyonu için yazma iznini geçici olarak kapatıyoruz
    results = model.process(rgb)
    rgb.flags.writeable = True
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), results


def extract_keypoints(results) -> np.ndarray:
    """ 
    İskelet noktalarını çıkarıp doğrusal bir öznitelik vektörüne (1662,) dönüştürür.
    Eğitim verisindeki tensör dizilimi korunmuştur: [Pose | Face | Left Hand | Right Hand]
    """
    # Kamera kadrajında uzuvların kaybolması durumunda sıfır matrisi (zero-padding) uyguluyoruz
    pose = (np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark], np.float32).flatten()
            if results.pose_landmarks else np.zeros(33 * 4, np.float32))

    face = (np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark], np.float32).flatten()
            if results.face_landmarks else np.zeros(468 * 3, np.float32))

    lh = (np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark], np.float32).flatten()
          if results.left_hand_landmarks else np.zeros(21 * 3, np.float32))

    rh = (np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark], np.float32).flatten()
          if results.right_hand_landmarks else np.zeros(21 * 3, np.float32))

    return np.concatenate([pose, face, lh, rh])


def draw_landmarks(image, results):
    """ Matematiksel modelin takip ettiği 2D/3D projeksiyon çizgilerini ekrana çizer. """
    if results.face_landmarks:
        mp_drawing.draw_landmarks(
            image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style()
        )
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_styles.get_default_pose_landmarks_style()
        )
    for hand_lm in [results.left_hand_landmarks, results.right_hand_landmarks]:
        if hand_lm:
            mp_drawing.draw_landmarks(
                image, hand_lm, mp_holistic.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
    return image


# ==============================================================================
# 3. ÇİFT DİLLİ TAHMİN HUD ARAYÜZÜ (BILINGUAL INTERFACE)
# ==============================================================================

def draw_prediction_hud(image: np.ndarray, prediction: str, confidence: float, 
                        probs: np.ndarray, idx2label: dict, buffer_fill: int, 
                        history: list, filter_active: bool = False) -> np.ndarray:
    """ Tahmin sonuçlarını ve istatistiksel olasılıkları eşzamanlı olarak ekrana basar. """
    h, w = image.shape[:2]
    overlay = image.copy()

    # ── Üst HUD Bilgi Paneli (Saydam Karartma Arka Planı) ──────────────────────
    cv2.rectangle(overlay, (0, 0), (w, 105), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, image, 0.30, 0, image)

    # ── Sınıf Karşılıklarını Sözlükten Çekme ve Fallback Mantığı ────────────────
    if prediction == "...":
        tr_text, en_text = "Bekleniyor...", "Waiting for gesture..."
        text_color = (140, 140, 140)
    else:
        # Eğer sözlükte varsa oradan alır, yoksa sistem ham ismi temizleyip iki dile de basar
        default_formatted = prediction.replace("_", " ").title()
        tr_text, en_text = JEST_SOZLUGU.get(prediction, (default_formatted, default_formatted))
        text_color = (0, 220, 90) if confidence >= CONFIDENCE_THRESHOLD else (0, 165, 255)

    # ── Ekran Kartı Üzerine Metinlerin Basılması ──────────────────────────────
    cv2.putText(image, f"TR: {tr_text}", (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(image, f"EN: {en_text}", (20, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
    
    if prediction != "...":
        cv2.putText(image, f"%{confidence * 100:.1f}", (w - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, text_color, 3)

    # ── SİSTEM DURUMU / DİNAMİK FİLTRE GÖSTERGESİ YAZIMI ──────────────────────
    if filter_active:  
        cv2.putText(image, "[DYNAMIC FILTER: ACTIVE]", (w - 340, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(image, "SYSTEM STATUS: STABLE", (w - 240, 35), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
   
    # ── Zaman Serisi Tampon Gösterge Çubuğu ──────────────────────────────────
    bar_x1, bar_y1, bar_x2, bar_y2 = 20, 92, w - 20, 98
    fill_w = int((buffer_fill / SEQUENCE_LENGTH) * (bar_x2 - bar_x1))
    cv2.rectangle(image, (bar_x1, bar_y1), (bar_x2, bar_y2), (50, 50, 50), -1)
    cv2.rectangle(image, (bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2), (0, 195, 255), -1)

    # ── Sağ Panel: İstatistiki Olasılık Dağılımı Yazae (Top-5 Probabilities) ─────────
    panel_x = w - 280
    cv2.rectangle(image, (panel_x - 10, 115), (w - 10, 115 + 5 * 38 + 15), (20, 20, 20), -1)

    top5_idx = np.argsort(probs)[::-1][:5]
    for i, idx in enumerate(top5_idx):
        p = float(probs[idx])
        lbl = idx2label.get(idx, str(idx))
        p_tr, _ = JEST_SOZLUGU.get(lbl, (lbl.replace("_", " ").title(), ""))
        y_pos = 145 + i * 38

        cv2.putText(image, p_tr[:14], (panel_x, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
        cv2.putText(image, f"{p*100:.1f}%", (w - 65, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (170, 170, 170), 1)

        # Olasılık çubuğu çizimi
        cv2.rectangle(image, (panel_x + 100, y_pos - 8), (w - 75, y_pos + 2), (50, 50, 50), -1)
        bar_len = int(p * (w - 75 - (panel_x + 100)))
        bar_col = (0, 200, 80) if p >= 0.85 else (0, 165, 255) if p >= 0.60 else (80, 80, 200)
        cv2.rectangle(image, (panel_x + 100, y_pos - 8), (panel_x + 100 + bar_len, y_pos + 2), bar_col, -1)

    # ── Sol Alt: Kararlı Tahmin Geçmişi Yazar────────────────────────────────────
    if history:
        for j, (past_lbl, past_conf) in enumerate(history):
            alpha = 0.3 + 0.7 * ((j + 1) / len(history))
            color = tuple(int(c * alpha) for c in (0, 210, 90))
            h_tr, _ = JEST_SOZLUGU.get(past_lbl, (past_lbl.replace("_", " ").title(), ""))
            cv2.putText(image, f"-> {h_tr}  %{past_conf*100:.0f}", (15, h - 15 - (len(history) - 1 - j) * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1)

    cv2.putText(image, "Q: Cikis / Exit", (w - 110, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (90, 90, 90), 1)
    return image


# ==============================================================================
# 4. ANA DÖNGÜ VE ÇIKARIM PIPELINE'I
# ==============================================================================

def run_inference():
    """ Kamera akışını eşzamanlı okuyarak kayan pencere (sliding window) çıkarımı yapar. """
    model, idx2label = load_resources()

    # Bellek verimliliği ve O(1) eleman değişimi için deque yapısı kullandk.
    sequence_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    pred_history_raw = collections.deque(maxlen=SMOOTHING_WINDOW)
    display_history = []

    current_pred = "..."
    current_conf = 0.0
    current_probs = np.zeros(len(idx2label))

    # Filtre takibi için gereken durum değişkenleri tanımlandı
    filter_active_flag = False  
    filter_counter = 0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("[HATA] Entegre veya harici kamera akışı başlatılamadı.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)

    print("[INFO] Gerçek zamanlı analiz başlatıldı. Kapatmak için videoda [Q] tuşuna basın.\n")

    with mp_holistic.Holistic(
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=1
    ) as holistic:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Ayna etkisi düzeltmesi için
            frame = cv2.flip(frame, 1)

            # Algılama ve Geometrik Çizimler
            image, results = mediapipe_detection(frame, holistic)
            image = draw_landmarks(image, results)

            # Öznitelik uzayını çıkartıp (1662,) zaman serisi tamponuna ekliyoruz
            keypoints = extract_keypoints(results)
            sequence_buffer.append(keypoints)

            # Matris penceremiz dolduğu an (T=30 kareye ulaşıldığında) çıkarım başlar
            if len(sequence_buffer) == SEQUENCE_LENGTH:
                # Girdi tensörünü ağ yapısına uygun hale getirme: (30, 1662) -> (1, 30, 1662)
                input_tensor = np.expand_dims(np.array(sequence_buffer, dtype=np.float32), axis=0)

                # Softmax olasılık dağılım vektörünü alma
                probs = model.predict(input_tensor, verbose=0)[0]
                pred_idx = int(np.argmax(probs))
                pred_label = idx2label[pred_idx]

                pred_history_raw.append(pred_label)

                # Çoğunluk Oylaması (Majority Voting) ile gürültü bastırma adımı
                smoothed_label = max(set(pred_history_raw), key=list(pred_history_raw).count)
                smoothed_conf = float(probs[list(idx2label.values()).index(smoothed_label)])
                current_probs = probs

                # Eşik Değer Kontrolü ve Dinamik Durum Filtresi
                if smoothed_conf >= CONFIDENCE_THRESHOLD:
                    if smoothed_label != current_pred:
                        display_history.append((smoothed_label, smoothed_conf))
                        if len(display_history) > 5:
                            display_history.pop(0)
                    
                    current_pred = smoothed_label
                    current_conf = smoothed_conf
                else:
                    # Modelin kararsız olduğu anlarda tamponlar sıfırlanır
                    current_pred = "..."
                    current_conf = 0.0
                    pred_history_raw.clear()
                    sequence_buffer.clear()

                    # Kararsızlık yakalandığı an bayrak kaldırılır ve 20 karelik sayaç başlar
                    filter_active_flag = True  
                    filter_counter = 20

                    # Filtre aktifken sayacı her karede bir azalt, süre dolunca durumu stable yap
                if filter_active_flag:
                    filter_counter -= 1
                if filter_counter <= 0:
                    filter_active_flag = False

            # Ekrana HUD Bilgilerini Yazdırma
            image = draw_prediction_hud(
                image, current_pred, current_conf, current_probs,
                idx2label, len(sequence_buffer), display_history, filter_active_flag
            )

            cv2.imshow(WINDOW_NAME, image)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_inference()