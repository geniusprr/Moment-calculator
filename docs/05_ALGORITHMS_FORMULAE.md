# 05_ALGORITHMS_FORMULAE.md - Hesap Mantığı (MVP)

## 1. Temel Denge Denklemleri
- Kuvvet dengesi: ΣF_y = 0
- Moment dengesi (referans A noktası): ΣM_A = 0
- Uygulama: R_A + R_B = ΣP + Σ(w_i * L_i) ve R_B * L - Σ(P_j * (x_j - x_A)) - Σ(w_i * L_i * (x_i^c - x_A)) = 0

## 2. Reaksiyon Hesabı Adımları
1. Tüm yayılı yükleri eşdeğer noktasal yüke çevir: P_eq = w * L, etki noktası x_c = start + L/2.
2. Noktasal yüklerin yönünü aşağı pozitif kabul ederek toplamı al.
3. Moment denkleminden R_B değerini çöz, ardından kuvvet denkleminden R_A.
4. Sonuçları üç ondalık basamakta yuvarla; işaret pozitif ise yukarı yön kabul edilir.

## 3. Kesme Kuvveti Diyagramı V(x)
- Başlangıçta V(0) = R_A.
- Her noktasal yük konumunda V(x) değeri yük kadar sıçrar.
- UDL segmenti üzerinde V(x) = V(x_start) - w * (x - x_start) lineer azalır.
- NumPy ile uygulama: kiriş boyunca eşit aralıklı x vektörü oluştur; her yük için parça parça katkı ekle.

## 4. Eğilme Momenti Diyagramı M(x)
- dM/dx = V(x) olduğundan, moment için kümülatif integral hesapla.
- Parça parça lineer bölgeler için trapzodiyal kural yeterli: M[i] = M[i-1] + 0.5 * (V[i] + V[i-1]) * Δx.
- Başlangıç koşulu M(0) = 0; uçta M(L) ≈ 0 kontrolü doğrulama amaçlı yapılır.

## 5. LaTeX Türetim Adımları
- Denge denklemlerini sembolik olarak sırala ve KaTeX uyumlu string üret.
- UDL dönüşümü için ayrı satır: w_{eq} = w * L, x_c = start + L/2.
- V(x) ve M(x) fonksiyonları Heaviside gösterimi ile ifade edilir (öğretici mod için):
  - V(x) = R_A - Σ P_j H(x - x_j) - Σ w_i (x - start_i) H(x - start_i) + Σ w_i (x - end_i) H(x - end_i)
  - M(x) = ∫_0^x V(s) ds

## 6. Sayısal Stabilite ve Kontroller
- Δx = L / (sampling_points - 1); varsayılan 0.025 m adım (L=5 m için 201 nokta).
- Son noktada |M(L)| < 1e-6 şartı; değilse yuvarlama hatası kullanıcıya uyarı olarak döner.
- Girişler metre ve kilonewton cinsinden yorumlanır; dönüşüm yapılmıyorsa kullanıcı uyarılır.

## 7. Gelecek Geliştirmeler (Not)
- Trapez yükler için parçalı lineer yoğunluk fonksiyonu eklenebilir.
- Noktasal momentler için ΣM_A denklemine ek terim, V(x) türevinde delta fonksiyonu etkisi.
