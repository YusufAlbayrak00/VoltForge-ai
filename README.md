# VoltForge AI ⚡🏭
**Endüstriyel Tüketim Çizelgeleme & EPİAŞ Enerji Arbitraj Kokpiti**

> VoltForge AI, organize sanayi tesislerinin makine ve vardiya çalışma planlarını EPİAŞ Gün Öncesi Piyasası saatlik elektrik fiyatlarına göre optimize ederek fabrika enerji maliyetlerini düşüren yapay zekâ destekli bir B2B karar destek platformudur.

---

## 🎯 Proje Vizyonu ve Amacı

Ağır sanayi ve üretim tesislerinde elektrik tüketimi, operasyonel giderlerin (OPEX) en büyük kalemlerinden birini oluşturur. EPİAŞ (Enerji Piyasaları İşletme A.Ş.) Gün Öncesi Piyasası'nda (GÖP) saatlik elektrik fiyatları (PTF - Piyasa Takas Fiyatı) gün içinde arz ve talebe bağlı olarak ciddi dalgalanmalar gösterir.

**VoltForge AI**, tesislerin esnek yüklerini (örneğin ergitme ocakları, kompresör sistemleri, değirmenler, pompalar vb.) ve vardiya çizelgelerini en yüksek fiyatlı pik saatlerden en ucuz vadilere kaydırarak (load shifting / arbitraj):
- Enerji maliyetlerini minimize eder,
- Üretim hedeflerinden ve kısıtlarından ödün vermez,
- Karar vericilere gerçek zamanlı interaktif simülasyon ve kokpit arayüzü sunar.

---

## 🏗️ Mimari ve Klasör Hiyerarşisi

Proje, katmanlı mimari (layered architecture) ve sorumlulukların ayrılığı (separation of concerns) ilkeleri gözetilerek yapılandırılmıştır:

```text
voltforge-ai/
├── requirements.txt         # Proje kütüphane bağımlılıkları
├── .gitignore               # Sürüm kontrolü dışlama kuralları
├── README.md                # Proje dokümantasyonu ve mimari genel bakış
├── database/                # Veri saklama, bağlantı ve şema yönetimi
│   ├── __init__.py
│   ├── connection.py        # SQLite/PostgreSQL veritabanı bağlantı yöneticisi
│   ├── schema.sql           # Tablo şemaları ve indeksler
│   ├── repository.py        # Veri erişim katmanı (Data Access Object / Repository)
│   └── seed.py              # Mock/örnek fabrika ve piyasa verisi tohumlama
├── integrations/            # Dış servis ve API entegrasyonları
│   ├── __init__.py
│   └── epias_client.py      # EPİAŞ Şeffaflık Platformu API istemcisi
├── ai_core/                 # Yapay zekâ, tahminleme ve optimizasyon motorları
│   ├── __init__.py
│   ├── price_predictor.py   # PTF (Piyasa Takas Fiyatı) tahminleme modeli
│   └── load_optimizer.py    # Kısıtlı optimizasyon ve yük çizelgeleme motoru
├── services/                # İş mantığı ve orkestrasyon katmanı
│   ├── __init__.py
│   └── energy_service.py    # Entegrasyon, veri ve AI modüllerini bağlayan servis
├── data/                    # Yerel veritabanı ve veri saklama dizini
│   └── .gitkeep
└── app.py                   # Streamlit tabanlı interaktif karar destek kokpiti
```

---

## 🧰 Teknoloji Yığını

- **Dil:** Python 3.10+
- **Veri Analitiği & İşleme:** Pandas, NumPy, python-dateutil
- **Dış Entegrasyonlar:** Requests (EPİAŞ Şeffaflık API)
- **Yapay Zekâ & Makine Öğrenimi:** Scikit-Learn
- **Dashboard & Görselleştirme:** Streamlit, Plotly
- **Veri Tabanı:** SQLite (Geliştirme) / PostgreSQL (Üretim hazır)

---

## 🚀 Kurulum ve Başlangıç

1. **Depoyu Klonlayın veya Açın:**
   ```bash
   cd voltforge-ai
   ```

2. **Sanal Ortam Oluşturun ve Aktive Edin:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # macOS / Linux
   # venv\Scripts\activate   # Windows
   ```

3. **Bağımlılıkları Yükleyin:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Uygulamayı Çalıştırın:**
   ```bash
   streamlit run app.py
   ```
