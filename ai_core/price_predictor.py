"""EPİAŞ Saatlik Fiyat Tahminleme (Price Predictor) Modülü.

Bu modül; geçmiş EPİAŞ Piyasa Takas Fiyatı (PTF) verileri, mevsimsellik,
gün içi tüketim trendleri ve takvim özniteliklerini kullanarak gelecek 24 saatlik
saatlik elektrik fiyatlarını tahminleyen makine öğrenimi modellerini (ML regression / time-series)
ve öznitelik mühendisliği (feature engineering) boru hatlarını barındırır.
"""

from typing import List, Optional

import pandas as pd


class PricePredictor:
    """EPİAŞ saatlik elektrik fiyatı tahmincisi için sınıf iskeleti.

    Bu sınıf, ileride zaman serisi öznitelikleri üzerinden saatlik fiyat
    tahmini yapacak modeli yönetmek üzere tasarlanmıştır. Şu aşamada yalnızca
    temel nitelikleri tanımlar.

    Attributes:
        model: Eğitilmiş tahmin modeli. Henüz eğitilmediği için ``None``'dır.
        features: Modelin kullanacağı takvim özniteliklerinin adları.
    """

    def __init__(self) -> None:
        """PricePredictor nesnesini başlangıç değerleriyle oluşturur.

        ``model`` niteliği henüz bir model yüklenmediği için ``None`` olarak,
        ``features`` niteliği ise kullanılacak takvim özniteliklerinin
        listesi olarak ayarlanır.
        """
        self.model: Optional[object] = None
        self.features: List[str] = ['hour', 'day_of_week', 'is_weekend']
