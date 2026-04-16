# vrp-rota-optimizasyonu
Kaggle VRP dataset (4.550 örnek) ile optimal rota tahmini | Nearest Neighbor + 2-opt | Random Forest R²=0.998 | Gantt chart araç çizelgesi | Python
# Araç Rota Optimizasyonu (VRP)

## Proje Hakkında

Kaggle **Vehicle Routing Problem GA Dataset** (4,550 VRP örneği) kullanılarak iki paralel analiz gerçekleştirilmiştir:

1. **Tahmin Modeli** : Problem özelliklerinden (müşteri sayısı, mesafe, kapasite) optimal çözüm değerini tahmin eden ML modeli
2. **VRP Simülasyonu** : Nearest Neighbor heuristik + 2-opt lokal arama ile rota optimizasyonu ve Gantt chart görselleştirmesi

## Veri Seti

| Özellik | Değer |
|---------|-------|
| Kaynak | [Kaggle — VRP GA Dataset](https://www.kaggle.com/datasets/abhilashg23/vehicle-routing-problem-ga-dataset) |
| Örnek Sayısı | 4,550 VRP problemi |
| Ortalama Müşteri | 191 |
| Araç Kapasitesi | 300 / 400 / 500 |

## Model Sonuçları

| Model | R² Skoru | MAE |
|-------|----------|-----|
| Linear Regression | 0.993 | 4,700 |
| **Random Forest** | **0.998** | **2,110** |
| Gradient Boosting | 0.999 | 1,859 |

## VRP Simülasyonu (20 Müşteri)

| Yöntem | Toplam Mesafe | İyileşme |
|--------|--------------|----------|
| Nearest Neighbor | 7,748 | — |
| **2-opt Optimizasyon** | **7,714** | **%0.4** |

## En Önemli Özellikler

1. **Müşteri Sayısı** : En belirleyici faktör
2. **Ortalama Depo Mesafesi** : Rota uzunluğunu doğrudan etkiler
3. **Tahmini Araç Sayısı** : Kapasite kullanım verimliliği

## Kullanılan Yöntemler

- **Nearest Neighbor Heuristik** : Başlangıç çözümü
- **2-opt Lokal Arama** : Rota iyileştirmesi
- **Random Forest Regressor** : Çözüm kalitesi tahmini
- **Özellik Mühendisliği** : Kapasite kullanımı, müşteri başı mesafe, araç sayısı tahmini
- **Gantt Chart** : Araç çizelgeleme görselleştirmesi

## Kurulum & Çalıştırma

```bash
pip install pandas numpy matplotlib scikit-learn
python vrp_analysis.py
```

Veri dosyasını (`VRP.csv`) aynı klasöre koy.

## Kullanılan Araçlar

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---
