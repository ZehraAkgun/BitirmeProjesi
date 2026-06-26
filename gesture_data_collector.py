import cv2
import numpy as np
import os
import mediapipe as mp

# ---------------------------------------------------------
# 1. MEDIAPIPE KURULUMU VE YARDIMCI FONKSİYONLAR
# ---------------------------------------------------------
# Yüz, el ve vücut noktalarını tek seferde yakalamak için MediaPipe Holistic modelini seçtim.
# MediaPipe'ın Holistic modelini ve çizim araçlarını ekrandan anlık takip yapabilmek için ekliyorum.
# Holistic: Yüz, el ve vücut iskeletini aynı anda tespit eden genel modeldir.
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """
    Kameradan (OpenCV) gelen BGR görüntüyü MediaPipe'ın doğru işleyebilmesi için RGB'ye çeviriyorum.
    Performansı artırmak ve belleği yormamak için geçici olarak yazmayi kapatıp tahmini alıyorum,
    ardından ekranda düzgün göstrmek için tekrar BGR formatına dönüp görüntüyü geri veriyorum
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # Renk uzayını dönüştür
    image.flags.writeable = False                  # Bellek optimizasyonu için görüntüyü kilitlyorum
    results = model.process(image)                 # Model tahmini-Landmark çıkarımı yaptığımız adım
    image.flags.writeable = True                   # Görüntüyü tekrar yazılabilir - yap kilidi kaldırmak
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # OpenCV'de Görüntüleme için BGR'a geri dön
    return image, results

def draw_landmarks(image, results):
    """
    Çıkarılan iskelet noktalarını ekranda canlı görmek için bu fonksiyonu yazdım.
    Böylece veri toplarken veya test ederken kameranın beni tam görüp görmediğini anlıyorum.
    """
    # Yüz, Vücut, Sol El ve Sağ El için noktaları (landmarks) ve bağları ayrı ayrıçizdiriyoruz
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION)
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def extract_keypoints(results):
    """
    KRİTİK FONKSİYON;
    MediaPipe'tan dönen karmaşık sonuç objesini, Makine Öğrenmesi modellerinin anlayabileceği 
    tek boyutlu (1D) flatten bir numpy dizisine çevirir.
    Eğer o karede bir organ (örn. sol el) kamerada görünmüyorsa, NaN hatası almamak için 
    o bölgeyi sıfırlardan oluşan bir dizi ile doldurur.
    """
    # Vücut: 33 nokta * 4 değer (x, y, z, görünürlük) = 132 boyutlu vektör
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    # Yüz: 468 nokta * 3 değer (x, y, z) = 1404 boyutlu vektör
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # Sol El: 21 nokta * 3 değer = 63 boyutlu vektör
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    # Sağ El: 21 nokta * 3 değer = 63 boyutlu vektör
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    
    # Tüm vektörleri ardışık olarak birleştirir. Toplam: Bir kare için 1662 boyutlu tek bir vektör elde edilir.
    return np.concatenate([pose, face, lh, rh])

# ---------------------------------------------------------
# 2. VERİ TOPLAMA PARAMETRELERİ
# ---------------------------------------------------------
DATA_PATH = os.path.join('MP_Data') # Verileri Toplayacağım ana klasör
gesture_name = input("Kaydedilecek jestin adını girin (örn: 'küsmek', 'para'): ")
gesture_type = input("Bu jest dinamik mi (zaman serisi) yoksa statik mi? (d/s): ").lower()

no_sequences = 30 # Her jestten kaç farklı video/örnek toplanacağı (
sequence_length = 30 # SADECE DİNAMİK İÇİN: Her bir videonun kaç kare (frame) olacağı (örn 30 fps = 1 saniye)

# Klasör yapısını oluşturuyouz:
if gesture_type == 'd':
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, gesture_name, str(sequence)))
        except: pass
elif gesture_type == 's':
    try:
        os.makedirs(os.path.join(DATA_PATH, gesture_name))
    except: pass

# ---------------------------------------------------------
# 3. KAMERA DÖNGÜSÜ VE VERİ KAYDETME 
# ---------------------------------------------------------
cap = cv2.VideoCapture(0) # 0 numaralı bilgisayarın varsayılan web kamerasını başlat
# MediaPipe Holistic modelini başlat (Güvenilirlik eşikleri 0.5 olarak ayarlandı gereksiz titremeler önlemek için)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    print(f"\n--- {gesture_name.upper()} İÇİN VERİ TOPLAMA BAŞLIYOR ---")
    
    if gesture_type == 'd':
        # LSTM modelini eğitmek için zaman serisi (video) mantığıyla veri topluyorum
        for sequence in range(no_sequences):
            for frame_num in range(sequence_length):
                ret, frame = cap.read()
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                
                # Ekrana kullanıcı için yönlendirme yazıları ekliyoruz
                if frame_num == 0: 
                    cv2.putText(image, 'BASLANGIC ICIN BEKLENIYOR...', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, f'Sira: {sequence} | Video Toplaniyor', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000) # Her yeni harekete başlamadan önce kullanıcıya 2 saniye hazırlanma payı veriyoeruz
                else: 
                    cv2.putText(image, f'Sira: {sequence} | Video Toplaniyor', (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    cv2.imshow('OpenCV Feed', image)
                
                # Koordinatları temizleyip ve klasöre .npy olarak kaydettik
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, gesture_name, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    elif gesture_type == 's':
        # Geleneksel modeller için (RF/SVM)STATİK Jestlerde  's' tuşuna basarak tek kare fotograf kaydediyoruz. 
        print("Statik veri toplanıyor. Her 's' tuşuna bastığınızda bir örnek kaydedilir. Çıkmak için 'q'ya basın.")
        sample_count = 0
        while sample_count < no_sequences:
            ret, frame = cap.read()
            image, results = mediapipe_detection(frame, holistic)
            draw_landmarks(image, results)
            
            cv2.putText(image, f'Kaydedilen Ornek: {sample_count}/{no_sequences}', (15,30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('OpenCV Feed', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('s'): # 's' tuşuna basıldığında o anki kareyi kaydet
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, gesture_name, f"sample_{sample_count}")
                np.save(npy_path, keypoints)
                print(f"Örnek {sample_count} kaydedildi.")
                sample_count += 1
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Veri toplama tamamlandı!")