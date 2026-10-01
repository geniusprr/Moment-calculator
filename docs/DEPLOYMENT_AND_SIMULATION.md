# Vercel ve sehim simülasyonu

## Dağıtım

Depo kökünden Vercel projesi oluşturulur. `vercel.json` iki Services tanımlar:
Next.js arayüz (`root: .`) ve FastAPI sunucu (`root: backend`, `main:app`).
`/api/*` ve `/health` istekleri Python servisine, diğer yollar arayüze gider.
Vercel Services beta özelliğidir; canlı dağıtım Vercel hesabı erişimi ve proje
oluşturulması tamamlandıktan sonra doğrulanmalıdır. Yerel CLI 62.1.0 iki servisi
doğru tanımıştır; bu çalışma ortamının ağ arayüzü kısıtı CLI'nin yerel frontend
başlatıcısını engellemiştir. Bu, bulutta yayınlandığı anlamına gelmez.

`NEXT_PUBLIC_API_BASE_URL` yayımlanan projede boş bırakılmalı veya `/api`
olmalıdır. Eski `http://localhost:8000/api` ayarı üretimde kullanılmamalıdır.
Python hesapları tarayıcıda veya JavaScript'e çevrilerek çalıştırılmaz;
`backend/main.py` mevcut `beam_solver_backend.main:app` uygulamasını yükler.
Çalışma bağımlılıkları doğrulanan sürümlere sabitlenmiştir.

Yerel geliştirme:

```sh
npm ci
python -m pip install -e './backend[dev]'
python -m uvicorn beam_solver_backend.main:app --port 8000
# Ayrı terminalde:
npm run dev
```

Next.js geliştirme proxy'si `/api` isteklerini Python'a iletir; arayüz ve
hesaplama aynı tarayıcı origin'i üzerinden kullanılır. GitHub'da `.gitignore`
içindeki genel `lib/` kuralı yüzünden kayıp olan `src/lib/api.ts` geri
eklenmiştir; Python'un kök `/lib/` ignore kuralı korunmuştur.

## Fizik modeli

Statik sehim: mevcut Python Macaulay çözümü, `EI w'' = -M`, aşağı yön pozitif.
Veri metre ve kN ile çözülür; sehim mm, dönme rad olarak döner.
`1 cm⁴ = 10⁻⁸ m⁴`, `E(GPa) × 10⁶ × I(m⁴) = EI(kN·m²)`.

Dinamik sehim: `/api/simulate`, iki düğümlü kübik Hermite Euler–Bernoulli
sonlu elemanları, tutarlı kütle matrisi ve mesnet sınır koşulları. Destek ve
yük süreksizlikleri düğüm yapılır; yaklaşık 40 veya daha fazla eleman vardır.
Yayılı yük vektörleri dört noktalı Gauss integrasyonuyla elde edilir.
`K φ = ω² M φ` problemi kütle Cholesky dönüşümüyle simetrik özdeğer problemi
olarak çözülür. Bütün serbest modlar tutulur; mod kesimi uygulanmaz.

Sıfır başlangıç yer değiştirmesi/hızıyla t=0 anında uygulanan ve sabit kalan
yükün sönümlü modal çözümü:

```text
w(x,t) = Σ w_static_mode(x) ×
 [1 − exp(−ζωt) × (cos(ω√(1−ζ²)t) + ζ/√(1−ζ²) sin(ω√(1−ζ²)t))]
```

Tarayıcı fiziksel zamanı bu kapalı formülde değerlendirir; örnek frame dizisi
üzerinden enterpolasyon veya keyfî salınım animasyonu kullanmaz. Çizim kübik
Hermite enterpolasyonla değerlendirilmiş mod şekillerini gösterir. Varsayılan
0.1× hız yalnızca oynatmayı yavaşlatır; frekans ve mm değerlerini değiştirmez.

Kütle kg/m ve sönüm kritik sönüm yüzdesi olarak girilir. Kütle öz ağırlık
kuvvetine otomatik çevrilmez. Çizimi görünür kılan büyütme gerçek sehim
değerlerinden ayrıdır; 1× gerçek geometrik ölçek seçilebilir. Statik yük
katsayısı −100%…+100% bütün kuvvet ve momentlerin yönünü/ölçeğini birlikte
değiştirir. Yeni girişlerde eski hesap ve dinamik model kullanılmaz.

Bu model doğrusal elastik, küçük deformasyonlu, sabit E ve I içindir. Kesme
deformasyonu, beton çatlaması/sünmesi, plastisite, büyük deplasman, burkulma,
mesnet ayrılması ve darbe teması modellenmez. Sonlu elemanlar bir yakınsamalı
sayısal yaklaşımdır; bu özellik genel amaçlı doğrusal olmayan fizik motoru
veya yönetmeliğe göre tasarım/güvenlik kontrolü olarak tanımlanmamalıdır.

## Düzeltilen mevcut hesap hataları

1. Sehim/dönme sınır koşullarında C1/C2 katsayılarının işareti hatalıydı.
   L=6 m, merkezde 10 kN, E=200 GPa, I=10000 cm⁴ basit kirişin doğru orta
   sehimi 2.25 mm ve iki mesnette sehim 0 mm'dir. Eski sürüm bu mesnet koşulunu
   sağlamıyordu.
2. Kısmi üçgen yayılı yükün bitiş düzeltmesi V/M fonksiyonlarında ters
   işaretliydi; bitişten sonraki kesme/moment ve gösterilen denklem düzeltildi.
3. `ccw`/`cw` momentlerin işareti ve çizimdeki oklar ters eşlenmişti. `ccw`
   şimdi gerçekten saat yönü tersidir. Bu nedenle aynı eski moment-yönü
   girdisiyle sonuçlar değişebilir. İki eski regresyon beklentisi fiziksel
   dengeye göre güncellendi; kullanıcı bir eski modeli karşılaştırırken
   bu yön düzeltmesini dikkate almalıdır.

API sözleşmesi ve Python motorunun uygulama içindeki kullanımı korunur;
eski hatalı sayısal sonuçlar bilinçli olarak kopyalanmaz.

## Doğrulama

```sh
cd backend
python -m pytest -q
# Depo kökünde:
npx tsc --noEmit
npm run lint
npm run build
```

Regresyonlar: `PL³/(48EI)` basit merkez yükü; `5qL⁴/(384EI)` tam yayılı yük;
`PL³/(3EI)` soldan/sağdan ankastre konsol; iki yük yönü; üç yayılı yük şekli;
kuvvet çifti; bağımsız FEM statik sınırının analitik çözüme uyumu; doğal
frekansın basit/konsol kiriş formülleriyle ve 1/√kütle ilişkisiyle uyumu.

Playwright üzerinden gerçek API ile statik görünüm, negatif yük katsayısı,
ani yük uygulama, duraklatma, t=0'da sıfır sehim, ipucu butonları ve mobil
yerleşim kontrol edilir. Canlı Vercel kontrolü ayrıca `/health`, `/api/solve`,
`/api/simulate`, `/api/chimney/period` ve üretim arayüzünü kapsamalıdır.

Kaynaklar:
- https://vercel.com/academy/python-on-vercel/run-with-vercel-dev
- https://vercel.com/docs/functions/runtimes/python
- https://teachbooks.tudelft.nl/computational-modelling/structural_linear/euler_bernouilli.html
- https://teachbooks.tudelft.nl/computational-modelling/structural_linear/Exercises/Workshop_FEM_dyn_beam.html
