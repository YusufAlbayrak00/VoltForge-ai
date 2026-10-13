"""EPİAŞ Saatlik Fiyat Tahminleme (Price Predictor) Modülü.

Bu modül; geçmiş EPİAŞ Piyasa Takas Fiyatı (PTF) verileri, mevsimsellik,
gün içi tüketim trendleri ve takvim özniteliklerini kullanarak gelecek 24 saatlik
saatlik elektrik fiyatlarını tahminleyen makine öğrenimi modellerini (ML regression / time-series)
ve öznitelik mühendisliği (feature engineering) boru hatlarını barındırır.
"""

from typing import List, Optional

import holidays
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
        self.features: List[str] = [
            'hour', 'day_of_week', 'is_weekend', 'is_holiday', 'is_holiday_eve'
        ]

    def _extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Takvim özniteliklerini (haftanın günü, hafta sonu, tatil, tatil arifesi) üretir.

        Girdi DataFrame'inin ``target_date`` (``YYYY-MM-DD`` formatında string)
        ve ``hour`` (int) sütunlarını içerdiği varsayılır. ``target_date``
        sütunu datetime'a çevrilerek aşağıdaki sütunlar türetilir:

        - ``day_of_week``: Haftanın günü (0: Pazartesi, ..., 6: Pazar).
        - ``is_weekend``: Cumartesi/Pazar ise 1, hafta içi ise 0 (int).
        - ``is_holiday``: Gün ``holidays.TR()`` takviminde (Türkiye resmi
          tatili) yer alıyorsa 1, aksi halde 0 (int).
        - ``is_holiday_eve``: Kendisi tatil değilken ertesi gün
          ``holidays.TR()`` takviminde yer alıyorsa (tatil arifesi) 1, aksi
          halde 0 (int). Çok günlük tatillerin ara günleri arife sayılmaz.

        Orijinal DataFrame değiştirilmez; kopya üzerinde çalışılır.

        Args:
            df: ``target_date`` ve ``hour`` sütunlarını içeren DataFrame.

        Returns:
            ``day_of_week``, ``is_weekend``, ``is_holiday`` ve
            ``is_holiday_eve`` sütunları eklenmiş DataFrame.
        """
        result: pd.DataFrame = df.copy()
        dates: pd.Series = pd.to_datetime(result['target_date'], format='%Y-%m-%d')
        result['day_of_week'] = dates.dt.dayofweek.astype(int)
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)

        tr_holidays = holidays.TR()
        result['is_holiday'] = dates.dt.date.map(lambda d: d in tr_holidays).astype(int)
        next_dates: pd.Series = dates + pd.Timedelta(days=1)
        next_is_holiday: pd.Series = next_dates.dt.date.map(lambda d: d in tr_holidays)
        result['is_holiday_eve'] = (next_is_holiday & (result['is_holiday'] == 0)).astype(int)
        return result

    def train(self, df: pd.DataFrame) -> None:
        """Fiyat tahmin modelini verilen geçmiş verilerle eğitir (şablon).

        Henüz uygulanmamıştır; ileride ``target_date``, ``hour`` ve hedef
        fiyat sütunlarını içeren geçmiş PTF verisi ile model eğitimi burada
        yapılacaktır.

        Args:
            df: Geçmiş saatlik PTF verisini içeren DataFrame.
        """
        pass

    def predict(self, target_date: str) -> dict:
        """Verilen tarih için saatlik fiyat tahminlerini döndürür (şablon).

        Henüz uygulanmamıştır; ileride ``target_date`` (``YYYY-MM-DD``) için
        24 saatlik tahmin sonuçlarını içeren bir sözlük döndürecektir.

        Args:
            target_date: Tahmin yapılacak gün (``YYYY-MM-DD`` formatında string).

        Returns:
            Saatlik tahminleri içeren sözlük.
        """
        pass
