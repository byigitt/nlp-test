# nlp-test

Türkçe ürün yorumları için duygu analizi modeli (BERTurk tabanlı transformer model)

## model performansı

### genel doğruluk

- doğruluk oranı: %80.3 (önceki: %68)
- ortalama f1-skor: 0.80

### duygu sınıflarına göre başarı

pozitif yorumlar:

- hassasiyet: %89.7
- duyarlılık: %91.5
- f1-skor: %90.6
- örnek sayısı: 4,178

negatif yorumlar:

- hassasiyet: %79.7
- duyarlılık: %79.2
- f1-skor: %79.5
- örnek sayısı: 3,435

nötr yorumlar:

- hassasiyet: %50.9
- duyarlılık: %48.4
- f1-skor: %49.6
- örnek sayısı: 2,089

### veri seti

- eğitim seti: 47,160 yorum
- test seti: 11,790 yorum

### güçlü yönler

- yüksek genel doğruluk (%80.3)
- pozitif yorumlarda mükemmel başarı (%90.6 f1-skor)
- negatif yorumlarda tutarlı performans (%79.5 f1-skor)
- nötr yorumlarda önemli iyileşme (%49.6 f1-skor, önceki: %27)
- GPU desteği ile hızlı tahmin
- yüksek güven skorları

### teknik özellikler

- BERTurk transformer model
- karma hassasiyet eğitimi (mixed precision)
- GPU hızlandırma
- toplu tahmin desteği
- önbelleğe alınmış model yükleme

## kullanım

### komut satırından kullanım:

```powershell
python predict.py "yorumunuz buraya"
```

Örnek:

```powershell
python predict.py "ürün çok kaliteli, kesinlikle tavsiye ederim"
```

### python kodundan kullanım:

```python
from predict import predict_sentiment

# Basit tahmin
sonuc = predict_sentiment("yorumunuz buraya")
print(sonuc)  # 'positive', 'negative' veya 'neutral'

# Güven skorları ile
sonuc = predict_sentiment("yorumunuz buraya", return_proba=True)
print(sonuc['prediction'])  # Tahmin edilen duygu
print(sonuc['probabilities'])  # Her sınıf için güven skorları

# Toplu tahmin
from predict import batch_predict
yorumlar = ["çok iyi", "berbat", "fena değil"]
sonuclar = batch_predict(yorumlar)
```

### sistem gereksinimleri

- Python 3.12+
- CUDA destekli GPU (önerilen)
- 4GB+ GPU RAM (önerilen)
- requirements.txt'deki bağımlılıklar
