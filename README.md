# Beam Solver - Kiriş Analiz Uygulaması Kurulum Kılavuzu

## Genel Bakış

Bu belge, Beam Solver kiriş analiz uygulamasının kurulum ve çalıştırma prosedürlerini detaylandırmaktadır. Uygulama, Next.js tabanlı frontend ve FastAPI tabanlı backend olmak üzere iki ana bileşenden oluşmaktadır.

---

## Sistem Gereksinimleri

### Gerekli Yazılımlar

#### 1. Node.js (v18.0 veya üzeri)
- **İndirme:** https://nodejs.org/
- **Önerilen Versiyon:** LTS (Long Term Support) sürümü
- **Kurulum Notu:** Sistem PATH değişkenine otomatik olarak eklenir

#### 2. Python 3.11
- **İndirme:** https://www.python.org/downloads/
- **Kritik Gereksinim:** Kurulum sırasında "Add Python to PATH" seçeneği mutlaka işaretlenmelidir
- **Versiyon Kontrolü:** Terminal/komut satırında `python --version` komutu ile doğrulanabilir

---

## Kurulum Prosedürü

### 1. Proje Dosyalarının Temini

Projeye ait tüm dosyaları bilgisayarınıza indirin. Proje yapısı aşağıdaki gibi olmalıdır:

```
moment-calculator/
├── (Next.js frontend dosyaları - ana dizinde)
├── backend/
│   ├── beam_solver_backend/
│   ├── tests/
│   └── pyproject.toml
├── package.json
├── next.config.js
└── README.md
```

### 2. Frontend Kurulumu

Frontend, proje ana dizininde bulunmaktadır. Kurulum için aşağıdaki adımları takip ediniz.

#### Adım 2.1: Proje Ana Dizinine Erişim
Terminal veya komut satırı arayüzünü açarak `moment-calculator` klasörüne geçiş yapınız:

```bash
cd moment-calculator
```

**Not:** `moment-calculator` klasörü içinde olduğunuzdan emin olunuz. Klasör içeriğini kontrol etmek için:
- Windows: `dir` komutu
- macOS/Linux: `ls` komutu

#### Adım 2.2: Frontend Bağımlılıklarının Yüklenmesi

Node.js paket yöneticisi npm kullanılarak gerekli tüm bağımlılıklar yüklenecektir:

```bash
npm install
```

**Bu komut ne yapar?**
- `package.json` dosyasında tanımlı tüm Next.js kütüphanelerini indirir
- React ve ilgili bağımlılıkları yükler
- Proje için gerekli diğer frontend paketlerini konfigüre eder
- `node_modules/` klasörü oluşturulur ve tüm paketler bu klasöre yerleştirilir

**Beklenen süre:** İnternet hızınıza bağlı olarak 1-3 dakika

**Başarılı kurulum göstergeleri:**
- Terminal'de hata mesajı görünmemelidir
- `node_modules/` klasörü oluşturulmuş olmalıdır
- `package-lock.json` dosyası oluşturulmuş olmalıdır

### 3. Backend Kurulumu

#### Adım 3.1: Backend Dizinine Erişim
`moment-calculator` klasörü içindeyken backend klasörüne geçiş yapınız:

```bash
cd backend
```

#### Adım 3.2: Python Sanal Ortamının Oluşturulması

Bağımlılık izolasyonu için Python sanal ortamı oluşturunuz:

**Windows:**
```bash
python -m venv .venv
```

**macOS/Linux:**
```bash
python3 -m venv .venv
```

#### Adım 3.3: Sanal Ortamın Aktivasyonu

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

**Not:** Başarılı aktivasyon sonrası komut satırının başında `(.venv)` etiketi görünecektir.

#### Adım 3.4: Python Bağımlılıklarının Yüklenmesi

```bash
python -m pip install --upgrade pip
pip install -e .[dev]
```

**Bu komutlar ne yapar?**
- İlk komut: pip paket yöneticisini en güncel versiyona yükseltir
- İkinci komut: `pyproject.toml` dosyasında tanımlı tüm backend bağımlılıklarını (FastAPI, NumPy, Uvicorn vb.) ve development araçlarını (pytest vb.) yükler

**Beklenen süre:** 2-5 dakika

---

## Uygulamanın Çalıştırılması

Uygulama iki ayrı servisten oluştuğu için iki ayrı terminal oturumu gerektirir.

### Terminal Oturumu 1: Backend Servisi

#### Adım 1: Backend dizinine geçiş ve sanal ortam aktivasyonu
Terminal açınız ve `moment-calculator/backend` klasörüne geçiş yapınız:

```bash
cd moment-calculator/backend
```

**Windows:**
```bash
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
source .venv/bin/activate
```

#### Adım 2: FastAPI sunucusunun başlatılması
```bash
uvicorn beam_solver_backend.main:app --reload
```

**Servis Bilgileri:**
- **Erişim Adresi:** http://127.0.0.1:8000
- **API Dokümantasyonu:** http://127.0.0.1:8000/docs
- **Reload Modu:** Kod değişikliklerinde otomatik yeniden başlatma aktif

**Başarılı başlatma çıktısı:**
```
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
INFO:     Started reloader process
INFO:     Application startup complete.
```

### Terminal Oturumu 2: Frontend Servisi

#### Adım 1: Ana dizine geçiş
Yeni bir terminal penceresi açınız ve `moment-calculator` klasörüne geçiş yapınız:

```bash
cd moment-calculator
```

**Not:** Ana dizinde (`moment-calculator` klasöründe) olduğunuzdan emin olunuz, `backend` klasörünün içinde değil.

#### Adım 2: Next.js development sunucusunun başlatılması
```bash
npm run dev
```

**Bu komut ne yapar?**
- Next.js development sunucusunu başlatır
- Hot reload özelliğini aktif eder (kod değişikliklerinde otomatik yenileme)
- Frontend uygulamasını varsayılan olarak 3000 portunda yayına alır

**Servis Bilgileri:**
- **Erişim Adresi:** http://localhost:3000 (varsayılan)
- **Hot Reload:** Kod değişikliklerinde otomatik güncelleme

**Başarılı başlatma çıktısı:**
```
▲ Next.js 14.x.x
- Local:        http://localhost:3000
- Ready in X.Xs
```

### Uygulamaya Erişim

Web tarayıcınızda aşağıdaki adrese gidiniz:
```
http://localhost:3000
```

---

## API Endpoint Referansı

### Backend API Endpointleri

#### 1. Kiriş Çözümleme
- **Endpoint:** `POST /api/solve`
- **Açıklama:** İzostatik ve konsol kiriş hesaplamaları
- **Çıktı:** Mesnet reaksiyonları, kesme/moment/normal kuvvet diyagramları, metadata

#### 2. Baca Periyot Hesabı
- **Endpoint:** `POST /api/chimney/period`
- **Açıklama:** Baca yapılarının birinci mod periyodu hesaplaması
- **Girdi:** Yükseklik, EI, kütle parametreleri
- **Çıktı:** Periyot, frekans, hesaplama notları

#### 3. Sağlık Kontrolü
- **Endpoint:** `GET /health`
- **Açıklama:** Servis durumu kontrolü

**API Dokümantasyonu:** Backend servisi çalışırken http://127.0.0.1:8000/docs adresinden interaktif Swagger UI'ya erişilebilir.

---

## Test Prosedürü

Backend için otomatik testlerin çalıştırılması:

```bash
cd moment-calculator/backend
.venv\Scripts\activate  # veya source .venv/bin/activate
pytest
```

---

## Sorun Giderme

### Yaygın Hatalar ve Çözümleri

#### Hata: "command not found: python" veya "python is not recognized"
**Neden:** Python sistem PATH'ine eklenmemiş  
**Çözüm:** Python'u "Add to PATH" seçeneği işaretli olarak yeniden kurunuz

#### Hata: "command not found: npm"
**Neden:** Node.js sistem PATH'ine eklenmemiş  
**Çözüm:** Node.js'i yeniden kurunuz ve sistemi yeniden başlatınız

#### Hata: "Address already in use" veya "Port already in use"
**Neden:** İlgili port başka bir süreç tarafından kullanılıyor  
**Çözüm:**  
- Çalışan servisleri durdurunuz (CTRL+C)
- Port kullanımını kontrol ediniz: 
  - Windows: `netstat -ano | findstr :8000` veya `netstat -ano | findstr :3000`
  - macOS/Linux: `lsof -i :8000` veya `lsof -i :3000`

#### Hata: "ModuleNotFoundError" veya paket import hataları
**Neden:** Python bağımlılıkları eksik veya sanal ortam aktif değil  
**Çözüm:**  
1. Sanal ortamın aktif olduğunu doğrulayınız `(.venv)` etiketi
2. Bağımlılıkları yeniden yükleyiniz: `pip install -e .[dev]`

#### Hata: Frontend bağlantı hataları (CORS, API çağrıları)
**Neden:** Backend servisi çalışmıyor  
**Çözüm:**  
1. Backend servisinin http://127.0.0.1:8000 adresinde aktif olduğunu doğrulayınız
2. http://127.0.0.1:8000/health endpoint'ini kontrol ediniz

#### Hata: npm install sırasında paket hataları
**Neden:** Node.js veya npm versiyonu eski, ya da cache sorunu  
**Çözüm:**  
1. Node.js versiyonunu kontrol ediniz: `node --version`
2. npm versiyonunu kontrol ediniz: `npm --version`
3. Gerekirse Node.js'i güncelleyiniz
4. npm cache'i temizleyiniz: `npm cache clean --force`
5. Mevcut `node_modules/` klasörünü siliniz
6. `package-lock.json` dosyasını siliniz
7. Tekrar `npm install` çalıştırınız

#### Hata: "Cannot find module" veya Next.js başlatma hataları
**Neden:** npm bağımlılıkları düzgün yüklenmemiş  
**Çözüm:**  
1. `node_modules/` klasörünün var olduğunu kontrol ediniz
2. Ana dizinde (`moment-calculator`) olduğunuzdan emin olunuz
3. `npm install` komutunu tekrar çalıştırınız
4. Hala sorun devam ederse: `rm -rf node_modules package-lock.json` (veya Windows'ta bu klasör/dosyaları manuel siliniz) ve ardından `npm install`

#### Hata: "No such file or directory" veya klasör bulunamıyor
**Neden:** Yanlış dizinde bulunuyorsunuz  
**Çözüm:**  
1. Mevcut konumunuzu kontrol ediniz: `pwd` (macOS/Linux) veya `cd` (Windows)
2. `moment-calculator` klasörüne geçiş yaptığınızdan emin olunuz
3. Klasör içeriğini kontrol ediniz: `ls` (macOS/Linux) veya `dir` (Windows)

---

## Servislerin Durdurulması

Her iki terminal oturumunda:
```
CTRL + C
```

Sanal ortamdan çıkış (opsiyonel):
```bash
deactivate
```

---

## Proje Mimarisi

### Frontend (Next.js)
- **Konum:** `moment-calculator` ana dizini
- **Framework:** Next.js (React tabanlı)
- **Paket Yöneticisi:** npm
- **Development Server:** Next.js built-in server
- **Port:** 3000 (varsayılan)
- **Konfigürasyon:** `package.json`, `next.config.js`

### Backend (FastAPI)
- **Konum:** `moment-calculator/backend/` dizini
- **Framework:** FastAPI
- **ASGI Server:** Uvicorn
- **Hesaplama Kütüphanesi:** NumPy
- **Port:** 8000 (varsayılan)
- **CORS:** Frontend erişimi için yapılandırılmış
- **Konfigürasyon:** `pyproject.toml`

---

## Geliştirici Notları

- Backend kodu `backend/beam_solver_backend/` dizininde bulunmaktadır
- Test dosyaları `backend/tests/` dizinindedir
- API şemaları `schemas.py` dosyasında tanımlanmıştır
- Hesaplama motorları `solvers.py` dosyasında yer almaktadır
- Frontend Next.js uygulaması ana dizinde konumlanmıştır
- Frontend yapılandırması ana dizindeki `package.json` dosyasındadır

---

## Hızlı Başlangıç Özeti

```bash
# 1. Frontend bağımlılıklarını yükleyin
cd moment-calculator
npm install

# 2. Backend kurulumu
cd backend
python -m venv .venv

# Windows için:
.venv\Scripts\activate

# macOS/Linux için:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -e .[dev]

# 3. Backend'i başlatın (Terminal 1)
cd moment-calculator/backend
.venv\Scripts\activate  # veya source .venv/bin/activate
uvicorn beam_solver_backend.main:app --reload

# 4. Frontend'i başlatın (Terminal 2)
cd moment-calculator
npm run dev

# 5. Tarayıcıda açın: http://localhost:3000
```

---

## Versiyon Bilgileri

- **Node.js:** ≥18.0
- **Python:** 3.11
- **Next.js:** (package.json'da belirtilen versiyon)
- **FastAPI:** (pyproject.toml'da belirtilen versiyon)

---

**Son Güncelleme:** 2025  
**Doküman Versiyonu:** 1.3