# Stanford veri seti sonuçları

Veri: Stanford AIMI Thyroid Ultrasound Cine-clip v1.0 · 192 biyopsi doğrulamalı nodül (17 kötü huylu) · 3254 kare örneği.
Doğrulama: 5 katlı, nodül bazında çapraz doğrulama (aynı nodülün kareleri aynı katta). Metrikler nodül düzeyindedir. Tarih: 2026-10-20.

| Yöntem | AUC |
|---|---|
| Radyoloğun TI-RADS puanı | 0.859 |
| Radyoloğun TI-RADS kategorisi (literatür riski) | 0.809 |
| ThirdEye otomatik TI-RADS önerisi (uzman maskesiyle) | 0.647 |
| ThirdEye görüntü modeli | 0.788 (%95 GA 0.693–0.876) |
| Birleşik skor: radyolog TI-RADS + model | 0.857 |

Kat bazında model AUC: 0.800 · 0.893 · 0.771 · 0.514 · 0.819.
Youden eşiğinde (0.047) model duyarlılığı %100, özgüllüğü %54.

Otomatik TI-RADS tanımlayıcılarının radyologla birebir uyumu: composition %70 · echogenicity %65 · shape %82 · margin %81 · echogenicfoci %54.
Otomatik segmentasyonun uzman maskesine uyumu (en büyük kesit, merkezden tohumlama): ortalama Dice 0.051; Dice ≥ 0,7 olan nodül oranı %2.

Notlar:
- Tek merkezli, küçük bir veri setidir; güven aralıkları geniştir. Dış doğrulama olmadan klinik performans olarak yorumlanmamalıdır.
- Model PyTorch ile eğitilip TensorFlow.js'e aktarılmıştır (tahmin farkı 3.2e-13).
- Ölçek (mm/piksel), kayıtlı en büyük nodül boyutundan kalibre edilmiştir.
- Ham görüntüler repoya eklenmemiştir; yalnızca model ağırlıkları ve bu özet paylaşılır.

## Yorum

- Görüntü modeli tek başına radyoloğun TI-RADS puanının gerisinde kalıyor (0,788'e karşı 0,859). Birleşik skor TI-RADS'a anlamlı bir katkı yapmıyor (0,857). Kat bazında AUC 0,51–0,89 arasında oynuyor; her katta yalnızca 3–4 kötü huylu nodül bulunduğu için bu beklenen bir belirsizlik.
- Otomatik TI-RADS önerisi (0,647) özellikle kompozisyon ve ekojenik odaklarda radyologdan ayrışıyor.
- Klasik (kontrasta dayalı) otomatik segmentasyon gerçek görüntülerde büyük ölçüde başarısız. Örneklenen 64 nodülün 47'sinde sınır bulunamadı; bulunanlarda ortalama Dice 0,18. Nodüllerin çoğu çevre dokuyla eş ekojenik (ekojenite oranı medyanı 1,00). Bu veri setindeki 17 bin uzman maskesiyle eğitilecek bir derin segmentasyon modeli (U-Net) bir sonraki öncelik olmalıdır.
- Sınıflandırma sonuçları uzman maskeleriyle hesaplanmıştır; uygulamada otomatik segmentasyon kullanıldığında başarı daha düşük olacaktır.

Kaynak: Stanford AIMI, Thyroid Ultrasound Cine-clip (https://doi.org/10.71718/7m5n-rh16). Veri seti Stanford Research Use Agreement kapsamında ticari olmayan araştırma amacıyla kullanılmıştır.
