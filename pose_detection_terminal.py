import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import argparse
import os
import numpy as np 
import math 
import sys 

# Mediapıpe çözümlerini yüklüyoruz (Sadece Pose)
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Sabit Değer
Y_THRESHOLD_RATIO = 0.15 
elbow_threshold   = 0.05

def main(image_path):
    """Görüntüyü yükler, açıyı hesaplar, kol pozisyonunu analiz eder ve sonucu Matplotlib penceresinde gösterir."""
    
    # 1. Resmi Yükleme ve Hazırlık
    image = cv2.imread(image_path)
    if image is None:
        print(f"HATA: Resim bulunamadı: {image_path}")
        return
    
    # OpenCV'den Mediapıpe'a: BGR'den RGB'ye çeviriyoruz
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    annotated_image = image_rgb.copy()
    
    detected_arm_status = "None" 
    
    # 2. Mediapıpe Pose Modelini Başlatma
    with mp_pose.Pose(static_image_mode=True, model_complexity=1) as pose:
        
        print(f"Analiz ediliyor: {os.path.basename(image_path)}")
        
        results = pose.process(image_rgb)
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks
            
            # --- 3. ANALİZ: 4 KİLİT NOKTA X VE Y KOORDİNATLARI ÇEKİLİYOR ---
            
            # Y koordinatları
            left_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
            right_wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
            left_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
            right_shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
            left_elbow_y  = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
            right_elbow_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y

            
            # X koordinatları
            left_wrist_x = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
            right_wrist_x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
            left_shoulder_x = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x
            right_shoulder_x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x
            left_elbow_x  = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
            right_elbow_x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x

            # --- POZİSYON KONTROLÜ ---
            
            # Delta Y (Pozitif fark = Yukarıda)
            left_diff_y = left_shoulder_y - left_wrist_y
            right_diff_y = right_shoulder_y - right_wrist_y
            
            # Delta X (Vektör Yönü)
            left_diff_x = left_wrist_x - left_shoulder_x
            right_diff_x = right_wrist_x - right_shoulder_x
            
            # Eşik Kontrolü (Boolean Değerler)
            is_left_raised = left_diff_y > Y_THRESHOLD_RATIO
            is_right_raised = right_diff_y > Y_THRESHOLD_RATIO

            # ELBOW BEARING
            left_diff_elbow_y  = left_elbow_y  - left_wrist_y
            right_diff_elbow_y = right_elbow_y - right_wrist_y
            
            # --- DURUM TESPİTİ ---
            KOLLAR_DURUMU_BOOL = [is_left_raised, is_right_raised]
                        
            if is_left_raised==False:
                if left_diff_elbow_y >= elbow_threshold:
                    KOLLAR_DURUMU_BOOL[0]= True

            if is_right_raised==False:
                if right_diff_elbow_y >= elbow_threshold:
                    KOLLAR_DURUMU_BOOL[1]= True

            if is_right_raised==False & is_left_raised==False:
                if left_diff_elbow_y >= elbow_threshold:
                    KOLLAR_DURUMU_BOOL[0]= True
                elif right_diff_elbow_y >= elbow_threshold:
                    KOLLAR_DURUMU_BOOL[1]= True

            if KOLLAR_DURUMU_BOOL == [True, True]:
                detected_arm_status = "Both"
            elif KOLLAR_DURUMU_BOOL == [True, False]:
                detected_arm_status = "Left"
            elif KOLLAR_DURUMU_BOOL == [False, True]:
                detected_arm_status = "Right"
            elif KOLLAR_DURUMU_BOOL == [False, False]:
                detected_arm_status = "None"
            
            # 4. Görselleştirme
            mp_drawing.draw_landmarks(
                annotated_image, landmarks, mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=3),
                connection_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )
            
            print(f"\nİkili Boolean Durum: {KOLLAR_DURUMU_BOOL}")
            print(f"🎉 SONUÇ: {detected_arm_status}")
            
        else:
            print("İnsan figürü algılanamadı.")

    # 5. Sonucu Matplotlib Penceresinde Gösterme (Açık Kalma Çözümü)
    plt.figure(figsize=(10, 10))
    plt.imshow(annotated_image)
    plt.title(f"Pose Analizi | Sonuç: {detected_arm_status}")
    plt.axis('off')
    
    plt.show() 

    # Terminalde Görüntü Açık Kalana Kadar Bekletme
    try:
        input("\n[BİLGİ] Görüntü açıldı. Kapatmak için terminalde Enter tuşuna basın...")
    except Exception:
        pass


if __name__ == '__main__':
    # Komut satırı argümanlarını tanımlama
    parser = argparse.ArgumentParser(description='Mediapıpe Pose ile Kol Pozisyonu Analizi.')
    
    parser.add_argument('path_to_input_image', type=str, 
                        help='Analiz edilecek resmin tam yolu (PATH-TO-INPUT-IMAGE).')
    
    args = parser.parse_args()
    
    main(args.path_to_input_image)