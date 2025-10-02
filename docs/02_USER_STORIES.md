# 02_USER_STORIES.md - Kullanıcı Hikâyeleri (MVP)

## Persona: Öğrenci
- US-01 Öğrenci olarak kiriş uzunluğu ve yükleri girip reaksiyonların doğru hesaplandığını görmek istiyorum.
  - Kabul Kriterleri: Form gönderildiğinde RA ve RB değerleri üç ondalık hassasiyetle görüntülenir; denge denklemleri sağlanmazsa hata verilir.
- US-02 Öğrenci olarak V(x) ve M(x) diyagramlarının nasıl değiştiğini anlık görmek istiyorum.
  - Kabul Kriterleri: Yük veya konum güncellendiğinde grafikler 300 ms içinde yeni şekle sorunsuz animasyonla geçer.
- US-03 Öğrenci olarak çözüme nasıl ulaşıldığını LaTeX formatında takip etmek istiyorum.
  - Kabul Kriterleri: Reaksiyon denklemi, kesme ve moment integralleri en az üç adımda gösterilir.

## Persona: Jüri / Eğitmen
- US-04 Jüri üyesi olarak sistemin gerçekten statik hesap yaptığını kanıtlayan kontrol adımlarını görmek istiyorum.
  - Kabul Kriterleri: Çıktıda denge denklemleri ve sonuç değerleri özetlenir; test senaryoları listelenir.
- US-05 Jüri üyesi olarak demo sırasında hızlıca farklı kombinasyonlar denemek istiyorum.
  - Kabul Kriterleri: Arayüz, son kullanılan yükleri saklar ve yeni hesaplama 2 saniye içinde tamamlanır.

## Persona: Proje Ekibi
- US-06 Geliştirici olarak hatalı girişlerin sistemde hata oluşturmamasını istiyorum.
  - Kabul Kriterleri: Girdi doğrulama kuralları istemci ve sunucu tarafında eşleşir; hata mesajı kullanıcıyı yönlendirir.
