# 09_ANALYTICS_TELEMETRY.md - Analitik (MVP)

## Amaç
- Demo sırasında hangi senaryoların denendiğini, çözüm sürelerini ve hata oranlarını izlemek.
- Gelecek iterasyon için kullanım alışkanlıklarını çıkarmak.

## Temel Metrikler
- problems_solved: Başarıyla tamamlanan çözüm sayısı.
- load_mix: Noktasal yük vs UDL kullanım oranı.
- vg_solution_time_ms: İstekten yanıtın alınmasına kadar geçen süre ortalaması.
- error_count: 4xx/5xx hata sayısı.

## Olay Şeması
`json
{
  "event": "solve_completed",
  "timestamp": "2025-02-03T12:30:45.123Z",
  "payload": {
    "duration_ms": 140,
    "point_loads": 1,
    "udls": 1,
    "warnings": []
  }
}
`
- Olaylar istemci tarafında toplanır, LocalStorage’da saklanır.
- Demo sonunda .json olarak indirilebilir veya konsola yazdırılabilir.

## Uygulama Notları
- Ağ bağlantısına ihtiyaç yok; veri sadece istemci tarafında.
- İleri sürümde backend’e gönderilecek telemetri için POST /analytics uç noktası eklenebilir.

## Dashboard (İleri Bakış)
- Basit React bileşeni ile toplam çözüm ve ortalama süre gösterimi.
- Gelecekte Grafana/Metabase entegrasyonu değerlendirilebilir.
