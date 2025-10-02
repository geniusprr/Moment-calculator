# 04_BACKEND_API_SPEC.md - Backend API (MVP)

## Genel
- REST bazlı, JSON giriş/çıkış, base path /api.
- Tek servis: statik kiriş çözücü; asenkron FastAPI uç noktası.
- Tüm yanıtlar pplication/json; charset=utf-8.

## POST /api/solve
- Açıklama: Kullanıcının tanımladığı kiriş için reaksiyon, kesme ve moment sonuçlarını hesaplar.
- Yetkilendirme: Yok (demo).

### İstek Gövdesi
`json
{
  "length": 5.0,
  "supports": {
    "left": { "type": "pin", "position": 0.0 },
    "right": { "type": "roller", "position": 5.0 }
  },
  "point_loads": [
    { "magnitude": 10.0, "position": 2.5, "direction": "down" }
  ],
  "udls": [
    { "magnitude": 3.0, "start": 0.0, "end": 5.0 }
  ],
  "sampling": { "points": 201 }
}
`
- length zorunlu (metre cinsinden > 0.5).
- supports sabit: sol pin, sağ oller.
- point_loads ve udls dizileri boş olabilir; MVP’de toplam yük sayısı yönetilebilir seviyede (≤3).
- sampling.points opsiyonel; verilmezse varsayılan 201.

### Yanıt Gövdesi (200)
`json
{
  "reactions": {
    "RA": 8.5,
    "RB": 9.5
  },
  "diagram": {
    "x": [0.0, 0.025, 0.05, "..."],
    "shear": [8.5, 8.5, -1.5, "..."],
    "moment": [0.0, 0.2125, 0.425, "..."]
  },
  "derivations": [
    "\\sum F_y = 0: R_A + R_B - 10 - 3*5 = 0",
    "\\sum M_A = 0: R_B*5 - 10*2.5 - 3*5*2.5 = 0",
    "V(x) = R_A - 10 H(x-2.5) - 3 x",
    "M(x) = \int_0^x V(s) ds"
  ],
  "meta": {
    "solve_time_ms": 120,
    "validation_warnings": []
  }
}
`
- diagram.x, shear, moment aynı uzunlukta yüzen sayı listeleri.
- derivations LaTeX formatlı string dizisi.
- meta.solve_time_ms performans ölçümü için.

### Hata Yanıtları
- 400 Bad Request: Girdi doğrulama hatası; detail alanında kullanıcıya dönük mesaj.
- 422 Unprocessable Entity: FastAPI şema validasyonu.
- 500 Internal Server Error: Beklenmeyen hata; log’da ayrıntı, yanıt üzerinde genel mesaj.

### Örnek Hata Yanıtı
`json
{
  "detail": [
    {
      "loc": ["point_loads", 0, "position"],
      "msg": "Yük kiriş uzunluğu içinde olmalıdır",
      "type": "value_error"
    }
  ]
}
`

## İç Modül Gereksinimleri
- Statik hesap fonksiyonu saf Python/NumPy; I/O katmanından ayrıştırılır.
- Hesaplama adımları listesi, sembolik LaTeX oluşturucu yardımcı fonksiyonla üretilir.
- V(x) ve M(x) eğrileri için vektörize çözüm; parça parça fonksiyon desteği.
- Birim testleri: Üç senaryoda reaksiyon ve diyagram karşılaştırması.
