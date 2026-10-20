# ThirdEye: Tiroid Nodülü Karar Destek Pilotu

**Canlı demo:** https://kadirerhan-lab.github.io/thirdeye/
**Kullanım videosu (2 dk):** https://kadirerhan-lab.github.io/thirdeye/media/thirdeye-kullanim-demosu.mp4

ThirdEye, tiroid ultrason videolarını ve görüntülerini analiz eden, nodülleri tespit edip ölçen, ACR TI-RADS tanımlayıcılarını öneren ve iyi huylu / kötü huylu ayrımı için bir risk skoru üreten bir araştırma prototipidir. Sistem, hekimin kesin tanı (sitoloji / histopatoloji) geri bildirimiyle kendini yeniden eğiten **öğrenen bir algoritmaya** sahiptir.

> **Önemli:** Bu yazılım bir araştırma ve gösterim prototipidir. Klinik tanı, tedavi veya hasta yönetimi kararlarında kullanılamaz. Demo içindeki örnek taramalar ve eğitim vakaları **sentetiktir**; gerçek hasta verisi değildir.

## Öğrenen pilot hakkında

ThirdEye'ın görüntü modeli sabit kurallara bağlı değildir. Kesin tanısı bilinen her vaka eğitim veri setine eklenebilir ve model tarayıcıda yeniden eğitilebilir. Her model sürümü, eğitimde kullanılmayan vakalarla doğrulanır (AUC, duyarlılık, özgüllük, ROC). Sürüm geçmişi kayıt altında tutulur.

Bu nedenle sistemin performansı, eğitim veri setinin **niceliği ve niteliğiyle** doğrudan ilişkilidir. Standart çekim protokolüyle elde edilmiş, doğru etiketlenmiş (Bethesda II / VI veya histopatoloji ile doğrulanmış) ve çok merkezli kaliteli vaka sayısı arttıkça modelin genelleme yeteneği ve doğruluğu artacaktır. Güvenlik amacıyla modelin nihai skordaki ağırlığı, kendi doğrulama başarısına ve klinik vaka sayısına bağlıdır. Yeterli kanıt birikene kadar skor, ACR TI-RADS'ın literatüre dayalı riskine yakın kalır.

Şu anki demo modeli yalnızca sentetik veriyle eğitilmiştir. Gösterilen metrikler, sistemin işleyişini göstermek içindir; klinik performans iddiası taşımaz.

## Ne yapar?

| Adım | İşlev |
|---|---|
| 1. Kayıt | MP4 / WebM / MOV video veya PNG / JPG görüntü dizisi yüklenir; videodan eşit aralıklı kareler alınır. Görüntü derinliği mm ölçeğini belirler. |
| 2. Ön işleme | Ultrason görüntü alanı otomatik kırpılır. Kaliper “+ / x” işaretleri, ölçüm çizgileri ve renkli yazılar tespit edilip inpainting ile temizlenir ([caliper-removal-lab](https://github.com/kadirerhan-lab/caliper-removal-lab) yaklaşımının geliştirilmiş hali). |
| 3. Tespit ve takip | Her karede parankime göre hipo/hiperekoik bölgeler bulunur ve kareler arasında izlenir. Taramada sabit kesitle görünen yapılar (karotis, trakea gölgesi) geri planda bırakılır. |
| 4. Segmentasyon | En büyük kesit karesi seçilir. Birden çok eşik denenir ve sınırı en belirgin olan en kapsayıcı kontur alınır; ince köprüyle bağlanan komşu yapılar ayrılır. Hekim tıklayarak veya elips çizerek nodülü elle de işaretleyebilir. |
| 5. Özellikler | Boyut (T, AP, en büyük çap), AP/T oranı, solidite, kenar düzensizliği ve keskinliği, ekojenite oranı, kistik oran, punktat / makro / periferik kalsifikasyon (tüm karelerde), posterior akustik bulgu ve doku (GLCM). |
| 6. ACR TI-RADS | Özellikler TI-RADS tanımlayıcılarına çevrilir; her öneri gerekçesi ve güveniyle gösterilir. Hekim onaylar veya düzeltir; puan, kategori ve İİAB / takip önerisi (boyut eşikleriyle) anında güncellenir. |
| 7. Risk skoru | `logit(nihai) = logit(TI-RADS riski) + w · model kanıtı`. TI-RADS kategori riskleri Middleton ve ark. (AJR 2017) verisidir. `w`, modelin doğrulama AUC'si ve klinik vaka sayısından hesaplanır. Modelin kararını etkileyen özellikler ve görüntü üzerindeki dikkat haritası gösterilir. |
| 8. Öğrenme | Kesin tanılı vaka veri setine eklenir, model tarayıcıda (TensorFlow.js) yeniden eğitilir. Veri seti dışa / içe aktarılarak merkezler arası birleştirilebilir. |

### Model

İki girişli küçük bir derin öğrenme ağı kullanılır. Birinci giriş 64×64 boyutunda, nodül merkezli bir görüntü yamasıdır (gri ton ve maske, 3 evrişim bloğu). İkinci giriş 20 nicel görüntü özelliğidir. Bu iki dal birleştirilir ve sigmoid çıkış üretir. Eğitimde sınıf dengeleme ve ayna artırımı uygulanır; tabakalı %20 doğrulama ayrılır. Model ve veriler yalnızca kullanıcının tarayıcısında (IndexedDB) tutulur. Hasta verisi bir sunucuya gönderilmez.

## Yerel çalıştırma

Gereksinim: [Node.js](https://nodejs.org) 18 veya üzeri. Ek paket kurulumu gerekmez.

```bash
npm start          # http://localhost:5173 adresini açar
npm test           # birim ve uçtan uca testler
```

Windows'ta `ThirdEye.bat` dosyasına çift tıklamak da yeterlidir. Uygulama internet bağlantısı olmadan da çalışır (TensorFlow.js `web/vendor` içinde gelir; yalnızca yazı tipleri çevrim içi yüklenir).

Kısa bağlantılar: `#tirads`, `#skor`, `#ogrenme` belirli bir adımı açar. `?ornek=benign` ise iyi huylu görünümlü örnekle başlatır.

Kullanım videosunu yeniden kaydetmek için (sunucu açıkken):

```bash
npm install
node tools/record-demo.mjs
```

## Proje yapısı

```
web/                 Yayınlanan statik uygulama (GitHub Pages)
  index.html, css/   Arayüz
  js/imgproc.js      Bağımlılıksız görüntü işleme çekirdeği
  js/preprocess.js   Sektör kırpma, kaliper / işaret temizleme
  js/detect.js       Tespit, takip, uyarlanabilir segmentasyon
  js/features.js     Nicel nodül özellikleri
  js/tirads.js       ACR TI-RADS puanlama, öneri ve İİAB eşikleri
  js/risk.js         TI-RADS + model birleşik skor
  js/model.js        TensorFlow.js modeli, eğitim, açıklanabilirlik
  js/store.js        Yerel vaka deposu, dışa / içe aktarma
  js/synth.js        Sentetik ultrason üreticisi (demo ve ilk eğitim)
  js/pipeline.js     Uçtan uca analiz akışı
tests/               Node testleri
tools/record-demo.mjs  Otomatik demo videosu
server.js            Bağımlılıksız yerel sunucu
```

## Yol haritası

1. **Gerçek veriyle eğitim:** Etik kurul onayıyla anonimleştirilmiş, sitoloji / patoloji doğrulamalı yerel vakalar; açık veri setleri (ör. Stanford AIMI tiroid cine-clip, DDTI, TN3K) ile ön eğitim.
2. **Derin segmentasyon:** Klasik tespit modülünün yerine U-Net / YOLO tabanlı bir model (Python / PyTorch'ta eğitilip ONNX ile tarayıcıya taşınır; mevcut ara yüz aynı kalır).
3. **DICOM desteği:** Piksel aralığı ve derinlik bilgisinin otomatik okunması.
4. **Çok merkezli öğrenme:** Veri yerine model güncellemelerinin paylaşıldığı federe öğrenme.
5. **Klinik doğrulama:** Prospektif çalışma ve okuyucu karşılaştırması.

## Bilinen sınırlamalar

- Ekstratiroidal uzanım ve kuyruklu yıldız artefaktı otomatik değerlendirilmez.
- İzoekoik, halosuz nodüller otomatik tespitte kaçabilir; elle işaretleme gerekir.
- Ölçüm doğruluğu, girilen derinlik / ölçek bilgisine bağlıdır.
- Sentetik veriyle elde edilen metrikler gerçek klinik performansı yansıtmaz.
