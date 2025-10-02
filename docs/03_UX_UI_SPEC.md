# 03_UX_UI_SPEC.md - Arayüz (MVP)

## Tasarım İlkeleri
- Sade, yüksek kontrastlı, mühendislik odaklı bir görünüm.
- Tüm kritik bilgiler tek ekranda; scroll minimum.
- Tutarlı ikonografi: Pin ve roller için standart semboller, yükler için oklar.

## Sayfa Düzeni
1. **Sol Panel (40%)** – Kiriş çizimi: Plotly veya Canvas üzerinde kiriş, destek ikonları, yük işaretleri. Kullanıcı yükleri sürükleyerek konum ayarlayabilir.
2. **Sağ Panel (30%)** – Sonuç kartları: Reaksiyonlar, denge denklemleri, LaTeX formatlı çözüm adımları. Kartlar akordeon şeklinde açılır.
3. **Alt Panel (30%)** – Grafik alanı: Sekmeli yapı ile Kesme Kuvveti V(x) ve Eğilme Momenti M(x) grafikleri arasındaki geçiş. Plotly animasyonları ile güncellenir.
4. Üstte ince bir araç çubuğu: Logo, tema modu (açık/koyu), “Temizle” ve “Hesapla” butonları.

## Bileşen Detayları
- **Girdi Formu:** Kiriş uzunluğu, yük ekleme listesi (tip, şiddet, konum, aralık). Sezgisel varsayılan değerler ve ipuçları.
- **Reaksiyon Kartları:** RA ve RB değerleri, yön okları, büyüklük bar grafiği.
- **Denklem Alanı:** KaTeX ile render edilen denge denklemleri; adımlar numaralandırılmış.
- **Grafik Sekmeleri:** Plotly gradient teması, referans çizgileri, maksimum/minimum etiketi; kullanıcı üzerine geldiğinde değer tooltip ile gösterilir.

## Mikro Etkileşimler
- Yük kartı üzerine gelindiğinde kiriş üzerinde ilgili yük highlight olur.
- “Hesapla” butonu aktif olduğunda hafif glow animasyonu; işlem sırasında spinner.
- Grafik güncellemeleri easing ile 200–300 ms arasında tamamlanır; gereksiz titreşim engellenir.

## Durumlar
- **İlk Durum:** Demo verileri önceden yüklü (L=5m, tek noktasal yük) ve sonuçlar hazır.
- **Hata Durumu:** Form alanının altında kırmızı açıklama, sonuç paneli gri-out; kullanıcıya nasıl düzelteceği anlatılır.
- **Yükleme Durumu:** Sonuç kartları eski değeri grileştirir, üstte ince progress bar.

## Responsive / Demo Gereksinimleri
- Masaüstü öncelikli; 1280x720 sunum ekranında piksel tasarrufu.
- Tablet kırılımında paneller dikey stack olur; grafikler tam genişlik.
- Animasyonlar düşük performanslı cihazlarda devre dışı bırakılabilir (ayar switchi).

## Erişilebilirlik
- Klavye ile tüm form alanlarına erişim.
- Renk kontrastı WCAG AA seviyesinde; kritik değerler metinle de gösterilir.
- Tooltip içeriği kısa ve net; ekran okuyucu için ARIA label’ları eklenir.
