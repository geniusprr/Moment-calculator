# Beam Solver Backend

FastAPI tabanlı izostatik kiriş hesaplama servisi. NumPy ile reaksiyon, kesme ve moment diyagramlarını üretir; LaTeX formatlı türetim adımları döner.

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
- `beam_solver_backend/solver` mevzu statik hesap mantığını içerir.
- `beam_solver_backend/api` FastAPI uç noktalarını barındırır.
- `tests/` otomatik test senaryoları.

## API
- `POST /api/solve` — kiriş parametreleri gönderildiğinde reaksiyonlar, diyagram verisi ve LaTeX adımlarını döner.

Sağlık kontrolü için `GET /health` uç noktasını kullanabilirsiniz.
