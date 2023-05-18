import numpy as np

# Veriler
veri = np.array([[80, 85],
                 [90, 82],
                 [75, 80],
                 [60, 55],
                 [95, 90],
                 [85, 92]])

k = 2  # Küme sayısı
epsilon = 0.01  # Durma kriteri

# Başlangıç küme merkezlerini rastgele seç
np.random.seed(0)
baslangic_merkezler = veri[np.random.choice(range(len(veri)), k, replace=False)]

while True:
    # Aidiyet dereceleri
    aidiyet = np.zeros((len(veri), k))
    for i, veri_noktasi in enumerate(veri):
        for j, merkez in enumerate(baslangic_merkezler):
            uzaklik = np.linalg.norm(veri_noktasi - merkez)
            if uzaklik == 0:
                aidiyet[i, j] = 1
            else:
                aidiyet[i, j] = 1 / np.sum([(uzaklik / np.linalg.norm(veri_noktasi - m)) ** 2 if np.linalg.norm(veri_noktasi - m) != 0 else 0 for m in baslangic_merkezler])

    # küme merkezlerini güncelle
    yeni_merkezler = np.zeros((k, veri.shape[1]))
    for j in range(k):
        aidiyet_kare = aidiyet[:, j] ** 2
        toplam_aidiyet = np.sum(aidiyet_kare)
        if toplam_aidiyet != 0:
            yeni_merkezler[j] = np.sum((aidiyet_kare.reshape(-1, 1) * veri), axis=0) / toplam_aidiyet

    # Küme merkezlerinin değişimini kontrol
    if np.linalg.norm(yeni_merkezler - baslangic_merkezler) < epsilon:
        break

    baslangic_merkezler = yeni_merkezler

# Sonuçlar
for i, veri_noktasi in enumerate(veri):
    en_yakin_kume = np.argmax(aidiyet[i])
    print(f"Veri Noktası {veri_noktasi} -> Küme {en_yakin_kume+1}")

print("Küme Merkezleri:")
for j, merkez in enumerate(baslangic_merkezler):
    print(f"Küme {j+1}: {merkez}")
