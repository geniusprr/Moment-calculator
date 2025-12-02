# Kiriş Analizi Algoritma Detayları ve Hesaplama Adımları

Bu doküman, uygulamanın Python backend tarafında (`beam_solver_backend`) çalışan kiriş analizi algoritmasının detaylı akışını, kullanılan formülleri ve mantıksal adımları içerir. Şema oluşturma ve dokümantasyon amacıyla hazırlanmıştır.

## 1. Genel Yaklaşım ve Yöntem

Algoritma, **Süperpozisyon (Üst Üste Bindirme) İlkesi** ve **Süreksizlik Fonksiyonları (Singularity Functions)** temeline dayanır. Analitik çözümler yerine, yüksek çözünürlüklü sayısal hesaplama (numerical computation) ve vektörel işlemler (`numpy`) kullanılarak diyagramlar oluşturulur.

### Temel Prensipler:
*   **Süperpozisyon:** Her bir yükün kiriş üzerindeki etkisi (kesme ve moment) ayrı ayrı hesaplanır ve toplanır.
    $$V_{toplam}(x) = \sum V_i(x)$$
    $$M_{toplam}(x) = \sum M_i(x)$$
*   **Ayrıklaştırma (Discretization):** Kiriş, sonlu sayıda noktaya (örneğin 401 nokta) bölünerek hesaplamalar bu noktalar üzerinde yapılır.

---

## 2. Veri Girişi ve Ayrıştırma (Input Parsing)

Kullanıcıdan gelen JSON verisi `SolveRequest` modeli ile karşılanır.

### Girdi Parametreleri:
*   **`length` ($L$):** Kirişin toplam uzunluğu (metre).
*   **`supports`:** Mesnetler.
    *   `position`: Konum ($x$).
    *   `type`: Tip (`pin`, `roller`, `fixed`).
*   **`point_loads`:** Tekil Yükler.
    *   `position`: Konum ($x$).
    *   `magnitude`: Şiddet ($P$, kN).
    *   `angle_deg`: Açı (Genellikle -90° düşey aşağı).
*   **`udls`:** Yayılı Yükler (Uniform Distributed Loads).
    *   `start`, `end`: Başlangıç ve bitiş konumları.
    *   `magnitude`: Şiddet ($w$, kN/m).
    *   `shape`: Yük profili (`uniform`, `triangular_increasing`, `triangular_decreasing`).
*   **`moment_loads`:** Tekil Momentler.
    *   `position`: Konum.
    *   `magnitude`: Şiddet ($M$, kNm).
    *   `direction`: Yön (`cw`: saat yönü, `ccw`: saat yönü tersi).

---

## 3. Adım 1: Mesnet Tepkilerinin Hesaplanması (Reaction Calculation)

Algoritma, kiriş tipine (`beam_type`) göre iki farklı yol izler.

### A. Ankastre Kiriş (Cantilever Beam)
Tek bir ankastre mesnet (genellikle $x=0$ noktasında) bulunur.

1.  **Düşey Denge ($\sum F_y = 0$):**
    Mesnet tepkisi ($R_y$), üzerindeki tüm düşey yüklerin toplamına eşit ve zıt yönlüdür.
    $$R_y = - (\sum P_{y,i} + \sum F_{udl,j})$$
    *   $P_{y,i}$: Tekil yükün düşey bileşeni ($P \cdot \sin(\theta)$).
    *   $F_{udl,j}$: Yayılı yükün eşdeğer tekil kuvveti (Alan hesabı).

2.  **Moment Dengesi ($\sum M_{mesnet} = 0$):**
    Ankastre mesnetteki tepki momenti ($M_R$), dış yüklerin mesnede göre yarattığı momentlerin toplamını dengelemelidir.
    $$M_R = - (\sum (P_{y,i} \cdot d_i) + \sum (F_{udl,j} \cdot d_{centroid,j}) + \sum M_{applied,k})$$
    *   $d_i$: Yükün mesnede olan mesafesi.
    *   $d_{centroid,j}$: Yayılı yükün ağırlık merkezinin mesnede olan mesafesi.

### B. Basit Kiriş (Simply Supported Beam)
İki mesnet bulunur (Sol mesnet A, Sağ mesnet B).

1.  **Moment Dengesi (A noktasına göre):**
    B mesnedindeki tepkiyi ($R_B$) bulmak için A noktasına göre moment alınır.
    $$\sum M_A = 0 \Rightarrow R_B \cdot L_{AB} - \sum M_{yükler@A} = 0$$
    $$R_B = \frac{\sum (P_i \cdot x_i) + \sum (F_{udl,j} \cdot x_{centroid,j}) + \sum M_{applied}}{x_B - x_A}$$

2.  **Düşey Denge ($\sum F_y = 0$):**
    A mesnedindeki tepki ($R_A$), toplam yükten $R_B$ çıkarılarak bulunur.
    $$R_A = (\sum P_{total}) - R_B$$

---

## 4. Adım 2: Ayrıklaştırma (Discretization)

Kiriş boyunca hesaplama yapılacak noktalar belirlenir.
*   **Yöntem:** `numpy.linspace(0, length, 401)`
*   **Amaç:** 0'dan $L$'ye kadar 401 adet eşit aralıklı $x$ koordinatı içeren bir dizi (array) oluşturulur. Bu dizi `x_axis` olarak adlandırılır.

---

## 5. Adım 3: Kesme Kuvveti Diyagramı Hesabı ($V(x)$)

Her bir $x$ noktası için kesme kuvveti hesaplanır. Python'da bu işlem döngü yerine vektörel (array) operasyonlarla yapılır.

**Genel Formül:**
$$V(x) = \sum V_{reaksiyon}(x) + \sum V_{tekil}(x) + \sum V_{yayılı}(x)$$

### Hesaplama Detayları:

1.  **Reaksiyonlar ve Tekil Yükler:**
    Heaviside Adım Fonksiyonu (Step Function) kullanılır. Yükün bulunduğu noktadan sonraki tüm $x$ değerleri için yük değeri eklenir.
    *   Kod Mantığı: `shear += load_value * (x_axis >= load_position)`
    *   Eğer $x < x_{yük}$ ise katkı 0, $x \ge x_{yük}$ ise katkı $P$.

2.  **Yayılı Yükler (UDL):**
    Yük fonksiyonunun integrali alınarak kesme kuvveti katkısı bulunur.
    *   **Düzgün Yayılı Yük ($w$):**
        $$V_{udl}(x) = -w \cdot (x - x_{start})$$ (Yük boyunca lineer artar)
    *   **Üçgen Yük (Artan):**
        $$V_{udl}(x) = -\frac{w \cdot (x - x_{start})^2}{2L_{udl}}$$ (Parabolik artar)

---

## 6. Adım 4: Eğilme Momenti Diyagramı Hesabı ($M(x)$)

Moment, kesme kuvvetinin integrali veya kuvvetlerin moment kollarının toplamı ile hesaplanır.

**Genel Formül:**
$$M(x) = \sum M_{reaksiyon}(x) + \sum M_{tekil}(x) + \sum M_{yayılı}(x) + \sum M_{noktasal}(x)$$

### Hesaplama Detayları:

1.  **Reaksiyonlar ve Tekil Yükler:**
    Macaulay Parantezi (Rampa Fonksiyonu) kullanılır. Kuvvet $\times$ Mesafe.
    *   Kod Mantığı: `moment += load_value * (x_axis - load_position) * (x_axis >= load_position)`

2.  **Yayılı Yükler (UDL):**
    Kesme kuvveti katkısının integrali alınır.
    *   **Düzgün Yayılı Yük:**
        $$M_{udl}(x) = -\frac{w \cdot (x - x_{start})^2}{2}$$ (2. Derece Parabol)
    *   **Üçgen Yük (Artan):**
        $$M_{udl}(x) = -\frac{w}{L_{udl}} \cdot \left( \frac{(x - x_{start})^3}{6} \right)$$ (3. Derece Eğri)

3.  **Tekil Momentler:**
    Doğrudan adım fonksiyonu olarak eklenir. Momentin uygulandığı noktada diyagramda ani bir sıçrama (jump) oluşur.
    *   Kod Mantığı: `moment += moment_magnitude * (x_axis >= moment_position)`

---

## 7. Adım 5: Kritik Noktaların Tespiti (Extrema Analysis)

Mühendislik açısından en önemli adım, maksimum momentin yerinin ve değerinin bulunmasıdır. Maksimum moment, kesme kuvvetinin sıfır olduğu ($V(x)=0$) veya işaret değiştirdiği noktalarda oluşur.

### Algoritma Akışı:
1.  **Aday Noktaların Belirlenmesi:**
    *   Kiriş uç noktaları ($x=0, x=L$).
    *   Tekil yüklerin olduğu noktalar (Kesme kuvvetinde süreksizlik yaratır).
    *   Kesme kuvvetinin işaret değiştirdiği aralıklar.

2.  **Sıfır Geçişi (Zero Crossing) Tespiti:**
    `shear` dizisi taranır. $V_i$ ve $V_{i+1}$ değerlerinin çarpımı negatifse ($V_i \cdot V_{i+1} < 0$), bu aralıkta bir kök vardır.

3.  **Kök Bulma (Root Finding):**
    İşaret değişimi tespit edilen aralıkta **Bisection Method (İkiye Bölme Yöntemi)** veya lineer interpolasyon kullanılarak $V(x)=0$ yapan hassas $x$ noktası bulunur.

4.  **Moment Hesabı ve Seçim:**
    Bulunan tüm kritik $x$ noktalarında moment değeri ($M(x)$) hesaplanır.
    *   **Max Positive:** En büyük pozitif değer.
    *   **Max Negative:** En küçük negatif değer (mutlak değerce büyük olabilir).
    *   **Max Absolute:** Mutlak değerce en büyük moment (Tasarım momenti).

---

## 8. Adım 6: Çıktı Oluşturma (Response Generation)

Hesaplanan tüm veriler `SolveResponse` modeli ile JSON formatında döndürülür.

### Çıktı İçeriği:
*   **`reactions`:** Hesaplanan mesnet tepkileri.
*   **`diagram_data`:**
    *   `x_axis`: Mesafe dizisi.
    *   `shear`: Kesme kuvveti dizisi.
    *   `bending`: Eğilme momenti dizisi.
*   **`meta`:**
    *   `max_moment`: Maksimum moment değeri ve konumu.
    *   `min_moment`: Minimum moment değeri ve konumu.
*   **`detailed_solution`:** Adım adım çözüm metinleri (Frontend'de gösterilmek üzere).
    *   Serbest Cisim Diyagramı açıklaması.
    *   Denge denklemleri.
    *   Kesme ve Alan yöntemlerinin sözel/matematiksel anlatımı.
