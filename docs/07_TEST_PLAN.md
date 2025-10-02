# 07_TEST_PLAN.md - Test Planı (MVP)

## Amaç
- Statik hesap motorunun doğruluğunu, API sözleşmesini ve arayüz davranışını doğrulamak.
- Demo sırasında kritik hataların görünmesini engellemek.

## Test Seviyeleri
- **Birim Testleri:** Hesap fonksiyonları için senaryo bazlı doğrulamalar (pytest).
- **API Testleri:** /api/solve uç noktasına istek/yanıt doğrulaması (FastAPI TestClient, schemathesis).
- **UI Testleri:** Girdi doğrulama, grafik güncelleme, hata mesajları (Playwright bileşen testleri).

## Kritik Senaryolar
1. **Basit Mesnetli Kiriş + UDL**
   - Girdi: L=6 m, w=5 kN/m.
   - Beklenen: RA = RB = 15 kN, Mmax = 45 kNm, V(x) lineer, M(x) parabolik.
2. **Tek Noktasal Yük**
   - Girdi: L=4 m, P=12 kN @ x=1.5 m.
   - Beklenen: RA = 7.5 kN, RB = 4.5 kN, moment diyagramı parça parça lineer.
3. **Karma (UDL + Noktasal)**
   - Girdi: L=5 m, w=3 kN/m (0-5), P=10 kN @ x=2 m.
   - Beklenen: Reaksiyonlar denge denklemlerine uygun; grafiklerde süreksizlik ve eğim değişimi doğru.
4. **Geçersiz Girdi**
   - Girdi: Yük pozisyonu L dışında.
   - Beklenen: 400 hata, kullanıcı dostu mesaj.

## Performans Testi
- Tek istekte çözüm süresi < 200 ms (lokal). 20 ardışık istek için ortalama ve 95. yüzdelik kaydedilir.

## Geriye Dönük Kontrol
- Reaksiyon toplamı eşittir toplam yük; ssert abs((RA+RB) - ΣYük) < 1e-6.
- M(L) sıfıra yakın olmalı; aksi halde test hatası.

## Test Otomasyonu
- Git hook (pre-push) birim ve API testlerini çalıştırır.
- UI smoke testi demo öncesi manuel: form doldur, sonuç gözlemi, hataya sebep olacak vaka.

## Kabul Kriterleri
- Tüm kritik senaryolar otomatik testlerde geçer.
- UI smoke test raporu kontrol listesi ile belgelenir.
- Açık hata bulunmaması ve demo ortamında en az iki kez başarıyla çalıştırılmış olması.
