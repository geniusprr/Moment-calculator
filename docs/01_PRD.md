# 01_PRD.md - Ürün Gereksinim Dokümanı (MVP)

## 1. Vizyon ve Amaç
- İnşaat mühendisliği öğrencilerinin basit kiriş problemlerini saniyeler içinde çözüp mantığını görmesini sağlayan öğretici bir web uygulaması.
- Demo ortamında güvenilir hesap sonuçlarını şık ve anlaşılır görsellerle sunarak jüriyi etkilemek.

## 2. Hedef Kullanıcılar ve İhtiyaçlar
### Öğrenci
- Farklı yük kombinasyonlarını kolayca tanımlamak ve reaksiyon/V-M diyagramlarını hızlıca görmek.
- Hesap adımlarını ders notlarına uygun formatta (LaTeX) takip etmek.
- Basit ve dikkat dağıtmayan bir arayüz ile değerleri değiştirdikçe sonucu anında görmek.
### Jüri / Değerlendirici
- Uygulamanın gerçekten hesap yaptığını doğrulamak için şeffaf sonuç özetleri görmek.
- Sunum esnasında gecikme yaşamadan çıkış almak ve animasyonlu grafiklerle etkileşimi göstermek.

## 3. Ürün Hedefleri
- Doğru sonuç: İzostatik kirişlerde ±0.5% toleransla reaksiyon ve diyagram değerleri.
- Hızlı deneyim: Hesaplama süresi < 1 sn, toplam yanıt < 2 sn.
- Öğretici çıktı: En az üç ana denklem adımının LaTeX formatında gösterimi.
- Sunum kalitesi: Plotly grafiklerinin 60 fps animasyonla güncellenmesi ve sade tasarım.

## 4. MVP Kapsamı
- **Girdi Modelleme:** Kiriş uzunluğu (0.5–10 m), iki destek (pin + roller sabit konumda), noktasal yük(ler) ve uniform yayılı yük aralıkları.
- **Hesaplama Çıktıları:** RA, RB reaksiyonları; 0–L aralığında örneklenmiş V(x) ve M(x); hesap adımları.
- **Arayüz Yetkinlikleri:** Sol panelde kiriş şeması ve yük ikonları, sağ panelde reaksiyonlar ve LaTeX denklemleri, alt panelde V/M grafikleri sekmeli.
- **Backend Uç Noktası:** Tek POST /api/solve uç noktası, JSON giriş/çıkış, hata mesajlarında girdi doğrulama detayı.
- **Analitik:** Sadece istemci tarafında olay kaydı (kaç problem çözüldü, yük tipi dağılımı).

## 5. Başarı Ölçütleri
- Akademik doğrulama: Test senaryolarının %100’ünde denge denklemleri sağlanır.
- Kullanıcı memnuniyeti: Demo geri bildirimi—öğrenci anketinde ≥4/5 kullanım kolaylığı.
- Görsel kalite: Plotly teması ve Tailwind tasarımı jüri sunumunda olumlu geri bildirim.
- Teknik güvenilirlik: Demo sırasında hata mesajı oranı < %5.

## 6. Kullanıcı Akışı (MVP)
1. Kullanıcı ana sayfada kiriş uzunluğu ve yükleri tanımlar.
2. Frontend girişleri yerinde doğrular; hatalar inline gösterilir.
3. Geçerli form backend’e gönderilir; backend hesaplamayı yapar.
4. Frontend reaksiyon kartlarını ve grafik sekmelerini animasyonla günceller.
5. Kullanıcı parametreleri değiştirerek yeni sonuçları canlı izler.

## 7. Teknoloji ve Entegrasyon
- **Backend:** Python 3.11, FastAPI, NumPy; bağımsız statik çözüm modülü; pytest ile doğrulama.
- **Frontend:** Next.js 14 (App Router), React 18, Tailwind CSS, Plotly.js, KaTeX.
- **Barındırma / DevOps:** Yerel demo için Docker Compose; gelecekte bulut dağıtımı (opsiyonel).
- **Analitik:** İstemci tarafı metric toplama (LocalStorage tamponu, opsiyonel export).

## 8. Varsayımlar ve Sınırlar
- Yalnızca iki mesnetli izostatik kirişler desteklenir.
- Yük kombinasyonu: Maksimum iki noktasal yük ve bir UDL (MVP basitliği için).
- Birimler SI; kullanıcı girişleri farklı birimde ise manuel dönüşüm gerekebilir.
- Normal kuvvet ve sehim hesapları MVP kapsamı dışındadır.

## 9. Riskler ve Azaltma
- **Hesaplama hatası:** Formüller için birim testleri + sembolik kontrol (SymPy prototipi).
- **Performans:** NumPy vektörleştirmesi, önceden ayrılmış x-ekseni örnekleme aralığı.
- **Sunum sırasında hata:** Mock verili fallback; arayüzde hata toast ve önerilen düzeltme.
- **Animasyon gecikmesi:** Plotly animasyonlarını minimal veri noktası ile sınırlamak.

## 10. Gelecek Sürümler İçin Notlar
- v1.1: Trapez yük, noktasal moment, normal kuvvet diyagramı, PDF dışa aktarma.
- v2.0: İç mafsal ve hiperstatik sistemler, sehim hesapları, çerçeve analizi.

