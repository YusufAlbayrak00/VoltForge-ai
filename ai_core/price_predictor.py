"""EPİAŞ Saatlik Fiyat Tahminleme (Price Predictor) Modülü.

Bu modül; geçmiş EPİAŞ Piyasa Takas Fiyatı (PTF) verileri, mevsimsellik,
gün içi tüketim trendleri ve takvim özniteliklerini kullanarak gelecek 24 saatlik
saatlik elektrik fiyatlarını tahminleyen makine öğrenimi modellerini (ML regression / time-series)
ve öznitelik mühendisliği (feature engineering) boru hatlarını barındırır.
"""

import logging
from pathlib import Path
from typing import List

import holidays
import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, root_mean_squared_error

logger = logging.getLogger(__name__)


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
            'hour', 'day_of_week', 'is_weekend', 'is_holiday',
            'lag_24', 'lag_168', 'rolling_mean_24'
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

        Zaman serisi öznitelikleri için veri önce ``target_date`` ve ``hour``
        sütunlarına göre kronolojik olarak sıralanır (indeks sıfırlanır).
        ``ptf_try_mwh`` sütunu varsa ek olarak şunlar üretilir (satır bazlı
        kaydırma; ardışık saatlik veri varsayılır):

        - ``lag_24``: 24 saat önceki (dün aynı saat) PTF, ``shift(24)``.
        - ``lag_168``: 168 saat önceki (geçen hafta aynı gün/saat) PTF,
          ``shift(168)``.
        - ``rolling_mean_24``: Son 24 saatin hareketli PTF ortalaması,
          ``shift(1).rolling(window=24).mean()``. ``shift(1)`` ile cari
          saatin kendi fiyatı pencereye girmez (veri sızıntısı önlenir).

        ``ptf_try_mwh`` yoksa (gelecek tahmin şablonu) ve bu sütunlar henüz
        mevcut değilse ``0.0`` atanır. Kaydırmadan doğan ``NaN`` değerler
        sırasıyla ``bfill``, ``ffill`` ve ``fillna(0.0)`` ile doldurulur.

        Orijinal DataFrame değiştirilmez; kopya üzerinde çalışılır.

        Args:
            df: ``target_date`` ve ``hour`` sütunlarını içeren DataFrame.

        Returns:
            Takvim ve zaman serisi öznitelik sütunları eklenmiş, kronolojik
            olarak sıralanmış DataFrame.
        """
        result: pd.DataFrame = df.copy()
        result = result.sort_values(by=['target_date', 'hour']).reset_index(drop=True)

        dates: pd.Series = pd.to_datetime(result['target_date'], format='%Y-%m-%d')
        result['day_of_week'] = dates.dt.dayofweek.astype(int)
        result['is_weekend'] = (result['day_of_week'] >= 5).astype(int)

        tr_holidays = holidays.TR()
        result['is_holiday'] = dates.dt.date.map(lambda d: d in tr_holidays).astype(int)
        next_dates: pd.Series = dates + pd.Timedelta(days=1)
        next_is_holiday: pd.Series = next_dates.dt.date.map(lambda d: d in tr_holidays)
        result['is_holiday_eve'] = (next_is_holiday & (result['is_holiday'] == 0)).astype(int)

        lag_columns: List[str] = ['lag_24', 'lag_168', 'rolling_mean_24']
        if 'ptf_try_mwh' in result.columns:
            result['lag_24'] = result['ptf_try_mwh'].shift(24)
            result['lag_168'] = result['ptf_try_mwh'].shift(168)
            # shift(1): cari saatin fiyatı pencereye girmez -> data leakage yok.
            result['rolling_mean_24'] = (
                result['ptf_try_mwh'].shift(1).rolling(window=24).mean()
            )
        else:
            for col in lag_columns:
                if col not in result.columns:
                    result[col] = 0.0

        result[lag_columns] = (
            result[lag_columns].bfill().ffill().fillna(0.0)
        )
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

    def evaluate(self, df: pd.DataFrame) -> dict[str, float]:
        """Eğitilmiş modelin başarımını test verisi üzerinde ölçer.

        Adımlar: eğitim durumu kontrolü, ``_validate_training_data`` ile
        doğrulama, ``_extract_features`` ile öznitelik üretimi, tahmin ve
        metrik hesabı. Metrikler:

        - ``mae``: Ortalama Mutlak Hata (TL/MWh).
        - ``rmse``: Kök Ortalama Kare Hata (TL/MWh).
        - ``mape``: Ortalama Mutlak Yüzde Hata (%). Payda
          ``np.maximum(np.abs(y_true), 1e-8)`` ile sıfıra bölünmeye karşı
          korunur.

        Args:
            df: ``target_date``, ``hour`` ve ``ptf_try_mwh`` sütunlarını
                içeren test verisi. Bu DataFrame değiştirilmez.

        Returns:
            ``{"mae": ..., "rmse": ..., "mape": ...}`` sözlüğü (2 ondalık).

        Raises:
            RuntimeError: Model henüz eğitilmemişse.
            ValueError: Test verisi doğrulamayı geçemezse.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model henüz eğitilmedi; değerlendirme yapmadan önce "
                "train() metodunu çağırın."
            )

        self._validate_training_data(df)
        featured_df: pd.DataFrame = self._extract_features(df)

        X: pd.DataFrame = featured_df[self.features]
        y_true: pd.Series = featured_df['ptf_try_mwh']
        y_pred = self.model.predict(X)

        mae: float = mean_absolute_error(y_true, y_pred)
        rmse: float = root_mean_squared_error(y_true, y_pred)
        mape: float = float(
            np.mean(np.abs(y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8)) * 100
        )

        metrics: dict[str, float] = {
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "mape": round(float(mape), 2),
        }
        logger.info(
            "Model değerlendirme -> MAE: %.2f TL/MWh, RMSE: %.2f TL/MWh, MAPE: %.2f%%",
            metrics["mae"], metrics["rmse"], metrics["mape"],
        )
        return metrics

    def _create_prediction_grid(self, target_date: str) -> pd.DataFrame:
        """Verilen gün için 24 saatlik (0-23) tahmin şablonunu oluşturur.

        Args:
            target_date: Tahmin yapılacak gün (``YYYY-MM-DD`` formatında string).

        Returns:
            ``target_date`` ve ``hour`` (0-23) sütunlarını içeren 24 satırlık
            DataFrame.

        Raises:
            ValueError: ``target_date`` ``YYYY-MM-DD`` formatında değilse.
        """
        try:
            pd.to_datetime(target_date, format='%Y-%m-%d')
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Geçersiz target_date: {target_date!r}. Beklenen format: YYYY-MM-DD."
            ) from exc

        return pd.DataFrame({
            'target_date': [target_date] * 24,
            'hour': list(range(24)),
        })

    def predict(
        self, target_date: str, historical_df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """Verilen gün için 24 saatlik PTF tahmini üretir.

        ``historical_df`` verilmişse geçmiş veri ile 24 saatlik gelecek şablonu
        alt alta birleştirilir ve ``_extract_features`` birleşik tablo üzerinde
        çalıştırılır; böylece ``lag_24``, ``lag_168`` ve ``rolling_mean_24``
        gerçek geçmiş fiyatlardan hesaplanır. Geçmişten yalnızca ``target_date``
        öncesi satırlar kullanılır; tahmin satırları ``target_date`` eşleşmesiyle
        seçilir. ``historical_df`` verilmezse öznitelikler
        doğrudan şablon üzerinde üretilir (lag sütunları ``0.0`` olur).
        Tahminler negatif olamayacağı için ``np.maximum(preds, 0.0)`` ile
        sınırlanır ve 2 ondalığa yuvarlanır.

        Args:
            target_date: Tahmin yapılacak gün (``YYYY-MM-DD`` formatında string).
            historical_df: ``target_date``, ``hour`` ve ``ptf_try_mwh``
                sütunlarını içeren geçmiş veri (tercihen tahmin gününden
                önceki en az 168 ardışık saat). Değiştirilmez.

        Returns:
            ``target_date``, ``hour`` ve ``predicted_ptf_try_mwh`` sütunlarını
            içeren 24 satırlık DataFrame.

        Raises:
            RuntimeError: Model henüz eğitilmemişse.
            ValueError: ``target_date`` geçersizse.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model henüz eğitilmedi; tahmin yapmadan önce train() veya "
                "load_model() metodunu çağırın."
            )

        future_grid: pd.DataFrame = self._create_prediction_grid(target_date)
        # Tarih biçimini normalize et (ör. "2026-10-8" -> "2026-10-08"); böylece
        # birleşik tabloda tarih eşleştirmesi güvenilir olur.
        target_ts: pd.Timestamp = pd.to_datetime(target_date, format='%Y-%m-%d')
        target_date = target_ts.strftime('%Y-%m-%d')
        future_grid['target_date'] = target_date

        history: pd.DataFrame | None = None
        if historical_df is not None and not historical_df.empty:
            # Yalnızca tahmin gününden ÖNCEKİ veri kullanılır: aynı gün veya sonrası
            # (gerçekleşmiş fiyat vb.) hem veri sızıntısına hem de çift satıra yol açar.
            history_dates: pd.Series = pd.to_datetime(
                historical_df['target_date'], format='%Y-%m-%d'
            )
            history = historical_df.loc[history_dates < target_ts].copy()

        if history is not None and not history.empty:
            combined_df: pd.DataFrame = pd.concat([history, future_grid], ignore_index=True)
            featured_all: pd.DataFrame = self._extract_features(combined_df)
            # Tahmin satırları tarih eşleşmesiyle belirlenir. `_extract_features` veriyi
            # (tarih, saat) sırasına dizdiği ve geçmiş yalnızca target_date öncesi olduğundan
            # bu satırlar tablonun sonunda, saat sırasıyla (0..23) yer alır.
            future_positions: np.ndarray = np.flatnonzero(
                (featured_all['target_date'] == target_date).to_numpy()
            )
            preds_list = self._predict_autoregressive(featured_all, future_positions)
            featured_future: pd.DataFrame = featured_all.iloc[future_positions].copy()
            featured_future['predicted_ptf_try_mwh'] = np.round(preds_list, 2)
        else:
            featured_future = self._extract_features(future_grid)
            X_pred: pd.DataFrame = featured_future[self.features]
            preds = self.model.predict(X_pred)
            featured_future['predicted_ptf_try_mwh'] = np.round(np.maximum(preds, 0.0), 2)

        return featured_future[['target_date', 'hour', 'predicted_ptf_try_mwh']].reset_index(
            drop=True
        )

    def _predict_autoregressive(
        self, featured_all: pd.DataFrame, future_positions: np.ndarray
    ) -> np.ndarray:
        """Hedef günün saatlerini sırayla tahmin eder; her tahmin sonraki saati besler.

        Her saat için ``rolling_mean_24``, birleşik tablodaki ``ptf_try_mwh`` sütununun
        önceki 24 değerinden (``shift(1).rolling(24).mean()`` ile aynı mantık) yeniden
        hesaplanır; ardından o saatin tahmini (negatif olamaz) ``ptf_try_mwh`` sütununa
        yazılır. Pencerede eksik değer varsa ``_extract_features``'ın doldurduğu değer
        korunur.

        Args:
            featured_all: Geçmiş + gelecek şablonu için öznitelikleri üretilmiş tablo.
            future_positions: Tahmin edilecek satırların konumları (saat sırasıyla).

        Returns:
            Her tahmin satırı için (yuvarlanmamış, negatif olmayan) PTF tahmini.
        """
        features_df: pd.DataFrame = featured_all[self.features].copy()
        ptf: np.ndarray = featured_all['ptf_try_mwh'].astype(float).to_numpy().copy()
        rolling_col: int | None = (
            features_df.columns.get_loc('rolling_mean_24')
            if 'rolling_mean_24' in features_df.columns
            else None
        )

        predictions: List[float] = []
        for pos in future_positions:
            if rolling_col is not None and pos >= 24:
                window: np.ndarray = ptf[pos - 24:pos]
                if not np.isnan(window).any():
                    features_df.iat[pos, rolling_col] = float(window.mean())

            pred: float = max(float(self.model.predict(features_df.iloc[[pos]])[0]), 0.0)
            ptf[pos] = pred  # Sonraki saatin pencere/özniteliği için geri besle
            predictions.append(pred)

        return np.array(predictions)

    def save_model(self, filepath: str | Path) -> None:
        """Eğitilmiş modeli ``joblib`` ile diske kaydeder.

        Args:
            filepath: Kaydedilecek dosya yolu. Üst klasör yoksa oluşturulur.

        Raises:
            RuntimeError: Model henüz eğitilmemişse.
        """
        if not self.is_trained:
            raise RuntimeError(
                "Model henüz eğitilmedi; kaydedilecek eğitilmiş bir model yok."
            )

        path: Path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        logger.info("Model kaydedildi: %s", path)

    def load_model(self, filepath: str | Path) -> None:
        """``joblib`` ile kaydedilmiş modeli yükler ve ``is_trained`` yapar.

        Args:
            filepath: Yüklenecek model dosyasının yolu.

        Raises:
            FileNotFoundError: Dosya mevcut değilse.
        """
        path: Path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {path}")

        self.model = joblib.load(path)
        self.is_trained = True
        logger.info("Model yüklendi: %s", path)
