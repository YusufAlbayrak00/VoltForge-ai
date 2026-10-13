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

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takvim özniteliklerini (haftanın günü ve hafta sonu) üretir.

        Girdi DataFrame'inin ``target_date`` (``YYYY-MM-DD`` formatında string)
        ve ``hour`` (int) sütunlarını içerdiği varsayılır. ``target_date``
        sütunu datetime'a çevrilerek aşağıdaki sütunlar türetilir:

        - ``day_of_week``: Haftanın günü (0: Pazartesi, ..., 6: Pazar).
        - ``is_weekend``: Cumartesi/Pazar ise 1, hafta içi ise 0 (int).

        Orijinal DataFrame değiştirilmez; kopya üzerinde çalışılır.

        Args:
            df: ``target_date`` ve ``hour`` sütunlarını içeren DataFrame.

        Returns:
            ``day_of_week`` ve ``is_weekend`` sütunları eklenmiş DataFrame.
        """
        result: pd.DataFrame = df.copy()
        dates: pd.Series = pd.to_datetime(result['target_date'], format='%Y-%m-%d')
        result['day_of_week'] = dates.dt.dayofweek.astype(int)
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)
        return result
