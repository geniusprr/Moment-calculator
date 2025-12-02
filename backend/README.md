# Beam Solver Backend

FastAPI tabanlı izostatik ve konsol kiriş hesaplama servisi. NumPy ile reaksiyonları, kesme/moment/normal diyagramlarını ve özet metaverisini üretir; detaylı türetim adımları tutulmaz.

## Kurulum
1. Python 3.11 sürümünü kullanın.
2. Sanal ortam oluşturun ve etkinleştirin:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```
3. Bağımlılıkları yükleyin:
   ```bash
   python -m pip install --upgrade pip
   pip install -e .[dev]
   ```

## Geliştirme
- Uygulamayı başlatmak için:
  ```bash
  uvicorn beam_solver_backend.main:app --reload
  ```
- Testleri çalıştırmak için:
  ```bash
  pytest
  ```

## Yapı
- `beam_solver_backend/main.py` FastAPI uygulamasını kurar ve CORS + health endpointlerini yönetir.
- `beam_solver_backend/api.py` `/api/solve` ve `/api/chimney/period` uç noktalarını tanımlar.
- `beam_solver_backend/schemas.py` tüm Pydantic istek/yanıt modellerini tek dosyada toplar.
- `beam_solver_backend/solvers.py` basit kiriş, konsol kiriş ve baca periyodu hesaplarını içerir.
- `tests/` otomatik FastAPI ve solver testlerini barındırır.

## API
- `POST /api/solve` — Basit veya konsol kiriş parametrelerini alır, mesnet reaksiyonları ile diyagram verilerini (`x`, `shear`, `moment`, `normal`) ve metaveriyi (`recommendation`, moment ekstramaları, çözüm süresi) döner.
- `POST /api/chimney/period` — Baca yüksekliği, EI ve kütle bilgileriyle ilk mod periyodunu, frekansını ve hesap notlarını raporlar.

Sağlık kontrolü için `GET /health` uç noktasını kullanabilirsiniz.
