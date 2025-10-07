Kirişlerde Kesme (Kesit) Yöntemi – adım adım, dört işlemle

Aşağıdaki anlatım, dayanak (mesnet) tepkileri bulunmuş bir kirişte iç kuvvetleri (eksenel kuvvet N, kesme kuvveti T ve eğilme momenti M) sadece toplama–çıkarma–çarpma–bölme ile nasıl bulacağınızı gösterir.
(İstendiği gibi mesnet tepkilerini hesaplamaya girmiyoruz; onlar bu yönteme başlamadan önce hazır olmalı.)

1) Yöntemin fikri (bir cümle)

Kirişi, ilgilendiğiniz noktadan kesip (hayalî bir düzlem), kesitte doğan N, T, M iç kuvvetlerini bilinmeyen olarak koyar; sonra kestiğiniz parçanın serbest cisim diyagramını (SCD) çıkarıp denge denklemleri ile bu bilinmeyenleri bulursunuz.

2) İşaret (pozitif) kabulü

Bir kez seçin, hep aynı kullanın:

T (kesme): Sol yüzeyde yukarı (sağ yüzeyde aşağı) olan pozitif.

M (moment): Kirişi gülümseten (alt lifte çekme) moment pozitif.

N (eksenel): Çekme pozitif. (Bu örnekte düşey yükler olduğundan N = 0 çıkacak.)

3) Genel algoritma (her parça için)

Parça (bölge) seç: Yük durumunun değiştiği her aralık için ayrı çalışılır.
Örn: dağılmış yükün bittiği, noktasal yükün olduğu, çiftin (momentin) uygulandığı, mesnetlerin olduğu yerler sınırtır.

Koordinat tanımla: O parçanın sol ucundan itibaren sağa doğru x metre ölç.
(İstersen sağ uçtan sola doğru z̄ de kullanabilirsin; sonuç değişmez.)

SCD çiz: Kestiğin parçada

Dağılmış yükleri eşdeğer tek kuvvet ile göster: büyüklük = w × uzunluk, konum = uzunluk/2.

Noktasal yükleri ve çiftleri olduğu gibi koy.

Kesitte N(x), T(x), M(x) çiz (işaret kabulüne göre).

Denge yaz ve çöz:

ΣFx = 0 → N(x)

ΣFy = 0 → T(x)

Kesit noktasına göre ΣM = 0 → M(x)

Hepsi sadece toplama–çıkarma–çarpma–bölmedir.

Parça aralığını (ör. 0 ≤ x ≤ 8 m) mutlaka yanına yaz.

Komşu parçaya geç: Aralık bittiği noktada “sağdan/soladan” değer atlamalarını (mesnet, nokta yükü, çift) kontrol et.

Kısa kontrol kuralları:
• Noktasal yük → T diyagramında atlama (miktarı yük kadar).
• Dağılmış yük → T eğimi w kadar; M’nin eğimi T kadar.
• Uygulanan çift (moment) → M diyagramında atlama (miktarı çift kadar); T değişmez.

4) Uygulamalı örnek (resimdeki kiriş)

Verilen kiriş (soldan sağa):

A’da pim, B’de makara; AB = 8 m, B’nin sağında 3 m konsol (toplam 11 m).

0–8 m aralığında w = 40 kN/m dağılmış yük.

x = 11 m’de 20 kN aşağı noktasal yük.

Sağ uçta (x = 11 m) 150 kNm saat yönünde uygulanan çift (negatif).

(Hazır) mesnet tepkileri: A_y = 133.75 kN, B_y = 206.25 kN.

Aşağıda iki parçayı keserek ilerliyoruz.

Parça–1: 0 ≤ x ≤ 8 m (dağılmış yük bölgesi)

SCD (sol parça):
Solda A_y yukarı; 0–x aralığındaki dağılmış yükün eşdeğeri 40·x kN, kesite uzaklığı x/2; kesitte T₁(x), M₁(x).

Denge (dört işlemle):

ΣFy = 0:
A_y − (40·x) − T₁(x) = 0 ⟹
T₁(x) = 133.75 − 40·x  [kN]

ΣM(kesit) = 0:
(+)M₁(x) − (A_y·x) + (40·x)·(x/2) = 0 ⟹
M₁(x) = A_y·x − 20·x² = 133.75·x − 20·x²  [kNm]

Önemli sayılar (hesabı yine dört işlem):

Kesme sıfır (tepe moment yeri):
133.75 − 40·x = 0 ⟹ x = 133.75/40 = 3.34375 m
Bu noktada M_max = 133.75·3.34375 − 20·(3.34375)² ≈ +223.61 kNm (sagging).

Moment sıfır (sehim işareti değişir):
133.75·x − 20·x² = 0 ⟹ x(133.75 − 20x) = 0 ⟹
x = 0 ve x = 6.6875 m → karşı eğrilik noktası.

x = 8 m (B’nin solu):
T₁(8) = 133.75 − 320 = −186.25 kN
M₁(8) = 133.75·8 − 20·64 = −210 kNm

Parça–2: 8 ≤ x ≤ 11 m (konsol aralığı, B ile sağ uç arası)

Bu aralıkta dağılmış yük yok. Hesap için iki eşdeğer yol var; ikisini de gösterelim:

(A) Sol parça ile (x soldan ölçülüyor)

ΣFy = 0:
A_y + B_y − 320 − T₂(x) = 0 ⟹
T₂(x) = 133.75 + 206.25 − 320 = 20 kN (sabit)

ΣM(kesit) = 0 (tek tek kaldıraç kolları):
(+)
𝑀
2
(
𝑥
)
M
2
	​

(x) − A_y·x − B_y·(x−8) + 320·(x−4) = 0

Dört işlemle sadeleştir:
M₂(x) = (A_y + B_y − 320)·x − (B_y·8 − 320·4)
M₂(x) = 20·x − 370  [kNm]

x = 11 m (sağ uca gelirken): M₂(11) = 220 − 370 = −150 kNm
Bu, uçtaki +150 kNm saat yönü çiftin (negatif) değeriyle tam uyumlu.

x = 11 m’de nokta yük 20 kN → T diyagramında 20 kN aşağı sıçrama:
soldan +20 kN iken, yükten hemen sonra 0 kN ile biter.

(B) Sağ parça ile (sağ uçtan sola doğru z̄ metre)

Sağdaki kısa parçayı alın; üzerinde 20 kN aşağı yük ve 150 kNm saat yönü çift var.

ΣFy = 0: T₂ − 20 = 0 ⟹ T₂ = 20 kN (yine sabit)

ΣM(kesit) = 0: −150 − 20·z̄ − M₂ = 0 ⟹
M₂(z̄) = −150 − 20·z̄
(z̄ = 0 → uçta −150, z̄ = 3 → B’de −210)

İki yazım da aynıdır; x = 11 − z̄ dönüşümü ile M₂(x) = 20·x − 370 elde edilir.

5) Sonuçların toparlanmış hali (parçalı fonksiyonlar)

Kesme kuvveti T(x) [kN]

0 ≤ x ≤ 8: T(x) = 133.75 − 40·x

8 ≤ x ≤ 11: T(x) = +20
(x = 11’de 20 kN aşağı nokta yük ⇒ T 20 kN düşer ve 0 ile biter.)

Eğilme momenti M(x) [kNm]

0 ≤ x ≤ 8: M(x) = 133.75·x − 20·x²
(x = 0 → 0; x = 6.6875 → 0; x = 8 → −210)

8 ≤ x ≤ 11: M(x) = 20·x − 370
(x = 11 → −150; uçtaki çift nedeniyle moment sıfır değil.)

Önemli sayısal noktalar

T = 0 → x = 3.34375 m (burada M_max ≈ +223.61 kNm).

M = 0 → x = 0 m ve 6.6875 m.

B’nin solu (8⁻): T = −186.25 kN, M = −210 kNm.

B’nin sağı (8⁺): T = +20 kN (B_y kadar yukarı sıçrar).

Uç (11 m): M = −150 kNm, T = 0 kN.

6) Çizim ipuçları (SFD/BMD)

T diyagramı: 0–8 m aralığında eğim −40 kN/m (çizgi doğrusal iner); 8–11 m aralığında yatay +20 kN; x = 11’de 20 kN aşağı atlayıp 0’da biter.

M diyagramı: 0–8 m aralığında parabol (ilk başta +, sonra 6.6875 m’de 0, 8 m’de −210); 8–11 m arası doğru (−210’dan −150’ye çıkar). Sağ uçta −150 kNm’lik atlama yoktur; değer sınırda biter (çift zaten sınır koşuludur).

7) Sık yapılan 5 hata

Yanlış işaret: “Pozitif kesme, pozitif moment” kabulünü her SCD’de tutarlı kullanın.

Dağılmış yükün moment kolu: Eşdeğer kuvvet w·x, kolu x/2 (parçanın kendi uzunluğu).

Parça sınırlarını yazmamak: Her formülün hangi x aralığında geçerli olduğunu mutlaka belirtin.

Çiftin etkisi: Çift T’yi değiştirmez, M’de miktarı kadar atlama oluşturur (sınırda değer).

Kontrol yapmamak:

0–8’de M(0) = 0, M(8) = −210 çıkmalı.

8–11’de M(11) = −150 çıkmalı.

T’nin alanı = M’deki değişim: 8→11 arası T = +20 sabit ⇒ M +60 artar: −210 → −150 (doğru).

8) Kısa özet (yedek kılavuz)

Parçayı seç, x’i tanımla.

SCD çiz: Dağılmış yük → w·x ve x/2.

ΣFy = 0 → T(x)

ΣM_kesit = 0 → M(x)

Noktalar (mesnet, yük, çift) için parça değiştir ve devam et.

Atlamaları ve sınır değerlerini kontrol et, diyagramları çiz.

Bu adımlarla, kirişte iç kuvvetleri tamamen dört işlem ile, türev/integral kullanmadan bulabilirsiniz.