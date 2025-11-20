import cv2
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from face_utils import get_face_keypoints
import argparse, sys, cv2
import numpy as np
from sklearn.neighbors import KernelDensity
from scipy.stats import entropy
import mediapipe as mp
import matplotlib.cm as cm
#############################################################################################
##Settings
yaw_trust_ratio          = 0.3
Gx_trust_ratio           = 0.2
new_metric_trust_ratio   = 0.5
right_threshold          = -0.15
left_threshold           = 0.15
total_variance_threshold = 1
##############################################################################################
# ---- Çekirdek noktalar (MediaPipe FaceMesh indexleri) ----
CORE_IDX = {
    "nose":        [1, 6],                   # burun ucu, köprü
    "eyes":        [33,133, 263,362, 159],  # sol dış/iç, sağ iç/dış, üstler
    "mouth":       [61,291, 13,14],          # sol/sağ köşe, üst/alt orta
    #"brows":       [46,55, 276,285],         # sol dış/iç, sağ dış/iç
    #"jaw_ears":    [152, 234,454, 205,425],  # çene ucu, sol/sağ yan yüz (kulak hizası), yanaklar
    #"ears":        [119,118,100,149, 348,349,330,379, 234,127,132, 454,356,361, 205,50,209, 425,280,429],
    #"forehead":    [10, 338, 297],
    "nose_sides":  [98, 327, 49, 279],
    #"cheeks":      [205, 425],
    #"mouth_inner": [78, 308],
    "eyes_lower":  [145, 374],
    #"brow_center": [66, 296],
}
# renkler (BGR)
COLORS = {
    "forehead":  (255, 200, 0),
    "nose":     (0, 255, 255),   # sarı
    "eyes":     (0, 200, 0),     # yeşil
    "mouth":    (60, 60, 255),   # kırmızı
    "brows":    (255, 120, 0),   # mavi ton
    "jaw_ears": (200, 0, 200),   # mor
    "ears":     (255, 0, 150),
    "nose_sides": (10, 10, 100),      # Kahverengi/Koyu (Burun Kenarları)
    "cheeks":    (150, 150, 150),     # Gri (Yanaklar)
    "mouth_inner": (255, 120, 0),     # Mavi ton (İç Dudaklar)
    "eyes_lower": (0, 120, 120),      # Teal (Göz Altları)
    "brow_center": (150, 50, 20),     # Kahverengi (Kaş Merkezi)
}
######################################################################
def estimate_kde_pdf(lms, core_indices, grid=25, bandwidth=0.02):
    
    # 1. Sabitler ve Filtreleme
    Z_SCALE_FACTOR = 20.0 # Z eksenini genişleten çarpan
    
    # CORE_IDX'teki tüm indeksleri topla
    all_indices = [idx for sublist in core_indices.values() for idx in sublist]
    unique_indices = sorted(list(set(all_indices)))
    
    # Hata Kontrolü
    if len(unique_indices) < 2:
        return None, None, None, None, None

    # 2. SEÇİLEN KİLİT NOKTALARIN KOORDİNATLARINI ALMA (X ve ÖLÇEKLENMİŞ Z)
    # Bu, KDE modelinin eğitileceği veridir.
    pts = np.array([[lms[idx].x, lms[idx].z * Z_SCALE_FACTOR] for idx in unique_indices])
    pts = np.clip(pts, -10.0, 10.0) # Sadece aşırı uçuk değerleri kesmek için geniş tutuldu

    # 3. KDE modeli oluştur ve eğit
    try:
        kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
        kde.fit(pts) 
    except np.linalg.LinAlgError:
        print("UYARI: KDE hesaplanamadı (Veri Tekillik hatası).")
        return None, None, None, None, None
    
    
    # --- 4. PDF GÖRSELLEŞTİRMESİ İÇİN IZGARA DEĞERLERİNİ HESAPLAMA ---
    
    # X EKSENİ: Normal (0'dan 1'e)
    xv = np.linspace(0,1,grid)
    
    # Y EKSENİ (Z'yi Temsil Eder): Ölçeklenmiş aralık (-1.0'dan 1.0'a)
    yv = np.linspace(-1, 1, grid) # Z, 10 ile çarpıldığı için bu aralık mantıklıdır.
    
    xv, yv = np.meshgrid(xv, yv) # 2D Izgara oluşturuldu
    grid_points = np.vstack([xv.ravel(), yv.ravel()]).T 

    # 5. Entropi ve PDF Haritası Hesaplama
    
    # Entropi için (Eğitilmiş noktalardaki yoğunluk)
    log_p = kde.score_samples(pts)
    probabilities = np.exp(log_p) / np.sum(np.exp(log_p))
    H_entropy = entropy(probabilities, base=2)
    
    # PDF Haritası için (Izgaradaki yoğunluk)
    log_p_grid = kde.score_samples(grid_points)
    pdf_map = np.exp(log_p_grid).reshape(grid, grid)
    pdf_map /= pdf_map.sum()

    E_X = np.var(pts[:, 0]) 
    
    # Z Koordinatlarının ortalaması (E[Z_scaled])
    E_Z = np.var(pts[:, 1]) 
    
    # E_XZ = [E_X, E_Z] listesi olarak döndürelim.
    total_variance = E_X + E_Z
    
    # PDF haritasının toplam olasılığını al (Yaklaşık 1.0 olmalı)
    P_total = pdf_map.sum() 

    # 1. Beklenen Değer E[X] (Yatay Merkez)
    # Formül: Σ [Koordinat * Olasılık Yoğunluğu] / Σ Olasılık
    E_X = np.sum(xv * pdf_map) / P_total
    
    # 2. Beklenen Değer E[Z] (Derinlik Merkezi)
    E_Z = np.sum(yv * pdf_map) / P_total

    # kde_model (kde) döndürülür
    return H_entropy, pdf_map, xv, yv, total_variance, kde, E_X, E_Z
# ----------------------------------------------------------------------
#################################################################################
def draw_points(img, landmarks):
    h, w = img.shape[:2]
    r = max(2, int(0.004 * max(w, h)))          # nokta yarıçapı, ölçekli
    fs = max(0.3, 0.0009 * max(w, h))           # yazı font ölçeği
    thick = max(1, int(0.0018 * max(w, h)))     # kalem kalınlığı

    overlay = img.copy()

    # tüm kategorileri çiz
    for cat, idxs in CORE_IDX.items():
        color = COLORS[cat]
        for idx in idxs:
            p = landmarks[idx]
            x, y = int(p.x * w), int(p.y * h)
            cv2.circle(overlay, (x, y), r, color, -1, lineType=cv2.LINE_AA)
            # index etiketini hafif kaydırarak yaz
            cv2.putText(overlay, str(idx), (x + r+2, y - r-2),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, color, max(1, thick-1), cv2.LINE_AA)

    # hafif saydamlıkla bindir
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    return img
#####################################################################################
def metric_G_front_face(kp):
    # Max Dikey Eğim Metriği (Roll)

    """
    Yüzün ön kisminin/simetri merkezini (C, N, M) hesaplar.
    Sonuç [G'x, G'y] listesi olarak Normalized (0.0 - 1.0) döner.
    """
    if kp is None:
        return None
        
    # 3 nokta var: C (Merkez), N (Burun), M (Ağız)
    x_coords = [kp[k][0] for k in ['C', 'N', 'M']]
    #y_coords = [kp[k][1] for k in ['C', 'N', 'M']]
    
    # NumPy'ın ortalama (mean) fonksiyonu ile ağırlık merkezini hesaplama
    Gx_prime = np.mean(x_coords)
    #Gy_prime = np.mean(y_coords)
    Sg = (Gx_prime - 0.5) / 0.25     # merkezleme + ölçekleme
    Zg = max(-1.0, min(1.0, Sg / 2.0))
    
    return Zg
#############################################################################################
def metric_yaw(kp):
    S_MAX = 0.5  # Max Yatay Dönüş Metriği (Yaw)
    if kp is None or kp['D'] <= 1e-6:
        # Hata kontrolü: Eğer kilit nokta yoksa veya gözler arası mesafe sıfırsa None döner.
        return None
        
    # Burun (N) ve Göz Merkezi (C) arasındaki X koordinat farkı
    xn, xc = kp['N'][0], kp['C'][0] 
    temp = (xn - xc) / kp['D']

    # Gözler arası mesafe (D) ile normalize etme
    return np.clip(temp / S_MAX, -1.0, 1.0)

##############################################################################################
# --- Görselleştirme ---
def draw_points_and_label(image, kp): # Artık label, S ve draw_arrow parametrelerini almıyoruz.
    """
    Sadece 5 kilit noktayı (L, R, C, N, M) daire olarak çizer.
    Tüm etiketler, oklar ve sonuç kutusu tamamen kaldırılmıştır.
    """
    h_orig, w_orig = image.shape[:2]

    def to_px(pt):
        x, y = float(pt[0]), float(pt[1])
        if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
            return int(x * w_orig), int(y * h_orig) # normalize -> piksel
        else:
            return int(x), int(y) # zaten piksel

    # Renkler
    col = {
        'L': (255,140,  0),  # turuncu
        'R': (180,  0,255),  # mor
        'C': (255,255,  0),  # sarı
        'N': (  0,  0,255),  # kırmızı
        'M': (  0,255,  0),  # yeşil
    }

    # Güvenli anahtar kontrolü
    for k in ['L','R','C','N','M']:
        if k not in kp:
            return image # Eksik nokta varsa çizim yapmadan orijinal resmi döndür

    # Noktaları piksel koordinatlarına çevir
    pts_px = {k: to_px(kp[k]) for k in ['L','R','C','N','M']}
    
    # Boyutlara göre dinamik yarıçap hesaplama
    base_size = min(w_orig, h_orig)
    radius = max(3, base_size // 150)   # Nokta yarıçapı
    
    # ******************************************************
    # SADECE 5 DAİRE ÇİZİLİYOR
    # ******************************************************
    
    cv2.circle(image, pts_px['L'], radius, col['L'], -1) 
    cv2.circle(image, pts_px['R'], radius, col['R'], -1)
    cv2.circle(image, pts_px['C'], radius+1, col['C'], -1) # Merkez biraz daha büyük olabilir
    cv2.circle(image, pts_px['N'], radius, col['N'], -1)
    cv2.circle(image, pts_px['M'], radius, col['M'], -1)

    return image
####################################################################################3
# --- Ana akış (main fonksiyonu) ---
# --- ANA AKIŞ FONKSİYONU (Düzeltildi) ---
# ⚠️ Bu fonksiyon artık image_path argümanını almalıdır.

def main(image_path): 
    print("Face Direction Estimator (Sadece Ham Noktalar)")
    
    path = image_path # Yolu doğrudan komut satırından alıyoruz

    if not os.path.exists(path):
        print(f"[Hata] Dosya bulunamadı: {path}")
        return

    img = cv2.imread(path)
    if img is None:
         print(f"[Hata] Görüntü okunamadı: {path}")
         return
         
    kp = get_face_keypoints(img) # kilit noktalar alınır
    
    if kp is None:
        print("[Sonuç] Yüz tespit edilemedi.")
        return

# --- MEDIAPIPE İŞLEME BLOKU ---
    mp_face = mp.solutions.face_mesh
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True) as fm:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = fm.process(rgb)

    lms = res.multi_face_landmarks[0].landmark

    # --- METRİKLERİ HESAPLAMA (İsim Hatası Giderildi) ---
    G_prime  = metric_G_front_face(kp) # Yüz Ön Simetrik Merkezi
    yaw_metric = metric_yaw(kp) # Yaw Metriği (S)
    H_entropy, pdf_map, xv, yv, total_variance, kde, E_X, E_Z = estimate_kde_pdf(lms, CORE_IDX)
    #new_metric = 2*math.tanh((total_variance) + 2.5 * (E_X - 0.5) -0.5)-1 # IN ORDER TO MAKE SYMMETRIC AROUND ZERO
    new_metric = -(3*math.tanh((total_variance) - 5 * (E_X - 0.5))-2) # IN ORDER TO MAKE SYMMETRIC AROUND ZERO

    final_score = math.tanh(new_metric_trust_ratio * new_metric + yaw_trust_ratio * yaw_metric + Gx_trust_ratio * G_prime)
  
    print(total_variance)
    print(final_score)

    OUTPUT = "FRONT"
    if final_score < right_threshold:
        if (total_variance > total_variance_threshold) & (yaw_trust_ratio * yaw_metric + Gx_trust_ratio * G_prime < right_threshold):
            OUTPUT = "RIGHT"
    elif final_score > left_threshold:
        if (total_variance > total_variance_threshold) & (yaw_trust_ratio * yaw_metric + Gx_trust_ratio * G_prime > left_threshold):
            OUTPUT = "LEFT"

    # Metrikleri hesaplamaya devam ediyoruz (yoruma alınmış kısımlar atlandı)
    
    # --- KONSOL RAPORU (İsim Hatası Giderildi) ---
    print("\n--- ANALİZ SONUÇLARI ---")
    # Değişkenler: G_prime ve yaw_metric olmalıydı.
    print(f"Gmetric: {G_prime}")
    print(f"yawmetric: {yaw_metric}") 
    
    # Sadece noktaları çizdiren fonksiyonu çağır
    vis = draw_points_and_label(img.copy(), kp) 

    # Görseli BGR → RGB çevir (matplotlib RGB ister)
    vis_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(8, 6))
    plt.imshow(vis_rgb)
    # Başlıkta sadece analiz sonucunu göster
    plt.title(f"Special Points (OUTPUT: {OUTPUT})") 
    plt.axis("off")
    plt.show()

# --------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Mediapipe yüz yönü analizi')
    parser.add_argument(
        'path_to_input_image',
        type=str,
        help='Analiz edilecek resmin tam yolu (örnek: "C:\\path\\to\\image.png")'
    )
    # Argümanları ayrıştır
    args = parser.parse_args()

    # Programı başlat: main fonksiyonunu komut satırından gelen yolla çağır
    main(args.path_to_input_image)