"""EPİAŞ Saatlik Fiyat Tahminleme (Price Predictor) Modülü.

Bu modül; geçmiş EPİAŞ Piyasa Takas Fiyatı (PTF) verileri, mevsimsellik,
gün içi tüketim trendleri ve takvim özniteliklerini kullanarak gelecek 24 saatlik
saatlik elektrik fiyatlarını tahminleyen makine öğrenimi modellerini (ML regression / time-series)
ve öznitelik mühendisliği (feature engineering) boru hatlarını barındırır.
"""

from typing import List

import holidays
import lightgbm as lgb
import pandas as pd


class PricePredictor:
    """EPİAŞ saatlik elektrik fiyatı tahmincisi için sınıf iskeleti.

    Bu sınıf, ileride zaman serisi öznitelikleri üzerinden saatlik fiyat
    tahmini yapacak modeli yönetmek üzere tasarlanmıştır. Şu aşamada yalnızca
    temel nitelikleri tanımlar.

    Attributes:
        model: Hiperparametreleri yapılandırılmış, henüz eğitilmemiş
            ``lightgbm.LGBMRegressor`` nesnesi.
        features: Modelin kullanacağı takvim özniteliklerinin adları.
        is_trained: Modelin eğitilip eğitilmediğini takip eden durum değişkeni.
    """

    def __init__(self) -> None:
        """PricePredictor nesnesini başlangıç değerleriyle oluşturur.

        ``model`` niteliği, aşırı öğrenmeyi sınırlayan muhafazakâr
        hiperparametrelerle yapılandırılmış bir ``LGBMRegressor`` olarak,
        ``features`` niteliği kullanılacak takvim özniteliklerinin listesi
        olarak, ``is_trained`` ise ``False`` olarak ayarlanır.
        """
        self.model: lgb.LGBMRegressor = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbose=-1,
        )
        self.is_trained: bool = False
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

    def _validate_training_data(self, df: pd.DataFrame) -> None:
        """Eğitim verisinin modele verilmeden önce geçerliliğini doğrular.

        Kontroller sırasıyla şöyledir ve ilk başarısız kontrolde açıklayıcı
        bir Türkçe mesajla ``ValueError`` fırlatılır:

        1. ``df`` ``None`` mı veya boş mu (``df.empty``).
        2. Zorunlu sütunlardan (``target_date``, ``hour``, ``ptf_try_mwh``)
           herhangi biri eksik mi.
        3. Zorunlu sütunlarda boş (``NaN`` / ``null``) değer var mı.
        4. Toplam satır sayısı 24'ten az mı (en az 1 tam günlük veri şartı).

        Args:
            df: Doğrulanacak geçmiş saatlik PTF verisini içeren DataFrame.

        Raises:
            ValueError: Yukarıdaki kontrollerden herhangi biri başarısız
                olduğunda.
        """
        required_columns: List[str] = ['target_date', 'hour', 'ptf_try_mwh']

        if df is None or df.empty:
            raise ValueError(
                "Eğitim verisi boş veya None olamaz; en az 1 tam günlük "
                "(24 saatlik) PTF verisi gereklidir."
            )

        missing_columns: List[str] = [
            col for col in required_columns if col not in df.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Eğitim verisinde zorunlu sütun(lar) eksik: {missing_columns}. "
                f"Beklenen sütunlar: {required_columns}."
            )

        null_counts: pd.Series = df[required_columns].isna().sum()
        columns_with_nulls: pd.Series = null_counts[null_counts > 0]
        if not columns_with_nulls.empty:
            details: str = ", ".join(
                f"{col}: {int(count)} adet" for col, count in columns_with_nulls.items()
            )
            raise ValueError(
                f"Zorunlu sütunlarda boş (NaN/null) değerler bulundu -> {details}. "
                "Lütfen eksik verileri temizleyin veya doldurun."
            )

        if len(df) < 24:
            raise ValueError(
                f"Yetersiz eğitim verisi: {len(df)} satır bulundu, en az 24 satır "
                "(1 tam günlük veri) gereklidir."
            )

    def train(self, df: pd.DataFrame) -> None:
        """Fiyat tahmin modelini verilen geçmiş PTF verisiyle eğitir.

        Eğitim adımları sırasıyla şöyledir:

        1. ``_validate_training_data`` ile veri doğrulanır (boş/None, eksik
           sütun, NaN değer, 24 satırdan az veri durumlarında ``ValueError``).
        2. Orijinal DataFrame'i değiştirmemek için ``df.copy()`` üzerinden
           ``_extract_features`` çağrılarak takvim öznitelikleri
           (``day_of_week``, ``is_weekend``, ``is_holiday``,
           ``is_holiday_eve``) üretilir.
        3. ``self.features`` sütunları özellik matrisi ``X``, ``ptf_try_mwh``
           sütunu hedef değişken ``y`` olarak ayrılır.
        4. ``self.model.fit(X, y)`` ile LightGBM modeli eğitilir.
        5. Eğitim başarıyla tamamlanırsa ``self.is_trained`` ``True`` yapılır.

        Args:
            df: ``target_date``, ``hour`` ve ``ptf_try_mwh`` sütunlarını
                içeren geçmiş saatlik PTF verisi. Bu DataFrame değiştirilmez.

        Raises:
            ValueError: Eğitim verisi doğrulamayı geçemezse.
        """
        self._validate_training_data(df)

        processed_df: pd.DataFrame = self._extract_features(df.copy())

        X: pd.DataFrame = processed_df[self.features]
        y: pd.Series = processed_df['ptf_try_mwh']

        self.model.fit(X, y)
        self.is_trained = True

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
