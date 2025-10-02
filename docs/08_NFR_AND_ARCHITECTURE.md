# 08_NFR_AND_ARCHITECTURE.md - NFR ve Mimari (MVP)

## Performans
- Sunucu yanıtı: Ortalama < 200 ms, en kötü < 500 ms (lokal demo ortamı).
- Frontend ilk boyama: 1.5 sn hedefi (Next.js statik pre-render + hydration).
- Grafik güncellemesi: 300 ms içinde tamamlanmalı.

## Güvenilirlik
- Hesap motoru beklenmeyen girdi aldığında anlamlı hata mesajı döndürür, servis çökmez.
- API hata kayıtları logging modülüyle JSON formatında tutulur.

## Kullanılabilirlik
- Arayüz İngilizce/Türkçe dil desteğine hazır (metinler config dosyasında).
- Demo sırasında offline çalışabilmesi için tüm bağımlılıklar bundle edilir.

## Bakım Kolaylığı
- Katmanlar ayrıştırılmış: solver modülü + pi katmanı + ui.
- Statik tip kontrolü (mypy) ve format (ruff/black) pipeline’da zorunlu.

## Güvenlik
- Girdi doğrulamaları enjeksiyon riskini azaltır; sadece sayısal alanlar kabul edilir.
- CORS sadece frontend domainine izin verir; demo aşamasında localhost.

## Mimari Akış
1. Kullanıcı Next.js arayüzünde formu doldurur.
2. Frontend istemci doğrulaması yapar, JSON isteği hazırlar.
3. POST isteği FastAPI katmanına ulaşır.
4. FastAPI, solver modülünü çağırır, sonuçları formatlar.
5. Yanıt frontend’e döner; Plotly grafikleri ve KaTeX denklemleri güncellenir.
6. Telemetri olayı (isteğe bağlı) LocalStorage’a yazılır.

## Bileşen Şeması (Sözel)
- **UI Layer:** Next.js sayfaları, durum yönetimi (Zustand veya Context) ve Plotly/KaTeX bileşenleri.
- **API Layer:** FastAPI router, Pydantic şemaları, hata yöneticisi.
- **Compute Layer:** NumPy tabanlı çözüm fonksiyonları, LaTeX üretici yardımcıları.
- **Test / Tooling:** Pytest, Playwright, Prettier + ESLint.

## Yayın Stratejisi
- MVP: Docker Compose ile tek komut (docker-compose up) demo ortamı.
- CI: GitHub Actions ile test ve kalite kontrolleri.
- Gelecek: Sunucuya konteyner push (opsiyonel).
