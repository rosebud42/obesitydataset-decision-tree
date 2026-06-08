# Obezite Seviyesi Sınıflandırma — Karar Ağacı

Bu proje, bireylerin beslenme alışkanlıkları, fiziksel aktivite düzeyleri ve demografik bilgilerini kullanarak obezite seviyelerini tahmin eden bir **Karar Ağacı (Decision Tree)** modeli geliştirmeyi amaçlamaktadır. Model, scikit-learn kütüphanesi ile oluşturulmuş olup Tkinter tabanlı bir arayüz üzerinden etkileşimli biçimde kullanılabilmektedir.

---

## Veri Seti

Projede kullanılan veri seti: [`ObesityDataSet_raw_and_data_sinthetic.csv`](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

Veri seti, Kolombiya, Peru ve Meksika'dan toplanan bireysel verileri ve sentetik olarak üretilmiş örnekleri içermektedir. Toplamda 2.111 kayıt ve 17 özellik bulunmaktadır.

### Hedef Değişken (`NObeyesdad`)

| Sınıf | Açıklama |
|---|---|
| `Insufficient_Weight` | Yetersiz Kilo |
| `Normal_Weight` | Normal Kilo |
| `Overweight_Level_I` | Fazla Kilo — Düzey I |
| `Overweight_Level_II` | Fazla Kilo — Düzey II |
| `Obesity_Type_I` | Obezite Tip I |
| `Obesity_Type_II` | Obezite Tip II |
| `Obesity_Type_III` | Obezite Tip III |

---

## Proje Yapısı

```
veri_madenciliği_proje/
│
├── main.py               # Uygulamanın giriş noktası
├── gui.py                # Tkinter arayüzü
├── preprocess.py         # Veri ön işleme (encoding, mapping)
├── model.py              # Karar Ağacı modeli, GridSearchCV, değerlendirme metrikleri
├── visualization.py      # Sınıf ve dağılım grafikleri
│
├── ObesityDataSet_raw_and_data_sinthetic.csv
├── requirements.txt
└── README.md
```

---

## Özellikler

- **Veri Ön İşleme**: İkili kategorik değişkenler için Label Encoding, çok sınıflı nominal değişkenler için One-Hot Encoding, sıralı (ordinal) değişkenler için manuel sayısal haritalama.
- **Hiperparametre Optimizasyonu**: `GridSearchCV` ile `StratifiedKFold` (5-fold) çapraz doğrulama; `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` ve `class_weight` parametreleri üzerinde arama yapılır.
- **Değerlendirme Metrikleri**: Accuracy, Sensitivity (Recall), Specificity ve F1-Score.
- **Görselleştirme**: Confusion Matrix ısı haritası, Feature Importance grafiği ve Karar Ağacı yapısı (ilk 3 katman).
- **Özellik Seçimi**: Arayüz üzerinden `Height` ve/veya `Weight` sütunları veri setinden çıkarılabilir; bu sayede modelin diğer özelliklere bağımlılığı incelenebilir.

---

## Kurulum

Python 3.8 veya üzeri bir sürüm gereklidir. Tkinter, Python'un standart kütüphanesiyle birlikte gelir; ayrıca kurulması gerekmez.

```bash
pip install -r requirements.txt
```

---

## Kullanım

```bash
python main.py
```

Uygulama açıldığında aşağıdaki adımlar izlenir:

1. **Veri Keşfi** — Obezite sınıflarının ve yaş/kilo dağılımlarının grafikleri görüntülenir.
2. **Özellik Seçimi** — İstenirse `Height` ve/veya `Weight` sütunları eğitimden çıkarılır.
3. **Veri Ayırma** — Test verisi yüzdesi girilir (varsayılan: %20).
4. **Model Eğitimi** — "Modeli Eğit" butonu ile GridSearchCV çalıştırılır; en iyi parametreler ve doğrulama skoru konsola yazdırılır.
5. **Değerlendirme ve Görselleştirme** — Metrikler arayüzde gösterilir; Confusion Matrix, Feature Importance ve Ağaç görselleştirmeleri ayrı pencerelerden açılır.

---

## Gereksinimler

| Paket | Sürüm |
|---|---|
| pandas | >= 1.5.0 |
| scikit-learn | >= 1.2.0 |
| matplotlib | >= 3.6.0 |
| seaborn | >= 0.12.0 |
| numpy | >= 1.23.0 |

---

## Lisans

Bu proje akademik amaçlarla geliştirilmiştir.
