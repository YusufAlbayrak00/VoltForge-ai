"""VoltForge AI Ana Orkestrasyon Hattı (Pipeline).

Bu modül; `PricePredictor` (fiyat tahmini), `LoadOptimizer` (MILP yük optimizasyonu)
ve `DatabaseManager` (kalıcı veri katmanı) bileşenlerini tek bir akışta birbirine bağlar.
"""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ai_core.load_optimizer import LoadOptimizer
from ai_core.price_predictor import PricePredictor
from database.db_manager import DatabaseManager
from integrations.epias_client import EpiasClient

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
METER_CSV_NAME: str = "Steel_industry_data.csv"

# Eğitim için hazırlanacak geçmiş gün sayısı. lag_168 (7 gün) için en az 10 gün
# gerekir; 14 gün, lag'lerin gerçek değerle dolu olduğu 7 günlük eğitim satırı bırakır.
HISTORY_DAYS: int = 14
MIN_HISTORY_HOURS: int = 240  # 10 gün


class VoltForgePipeline:
    """Fabrika kurulumu, fiyat tahmini ve günlük optimizasyonu yöneten ana sınıf.

    Attributes:
        db: Kalıcı veri katmanı yöneticisi.
        predictor: PTF fiyat tahmincisi.
        meter_csv_path: Kalibrasyonda kullanılacak sayaç CSV yolu (bulunamazsa `None`).
    """

    def __init__(
        self,
        db_path: Optional[str] = None,
        meter_csv_path: Optional[str] = None,
    ) -> None:
        """Pipeline bileşenlerini hazırlar.

        Args:
            db_path: SQLite yolu. `None` ise `DatabaseManager` varsayılanı kullanılır;
                `":memory:"` verilebilir.
            meter_csv_path: Sayaç verisi CSV yolu. Verilmezse proje kökündeki
                `Steel_industry_data.csv`, yoksa `data/Steel_industry_data.csv` aranır.
        """
        self.db: DatabaseManager = DatabaseManager(db_path=db_path)
        self.predictor: PricePredictor = PricePredictor()
        self.epias_client: EpiasClient = EpiasClient()
        self.meter_csv_path: Optional[str] = self._resolve_meter_csv(meter_csv_path)

    @staticmethod
    def _resolve_meter_csv(meter_csv_path: Optional[str]) -> Optional[str]:
        """Sayaç CSV dosyasının yolunu belirler.

        Args:
            meter_csv_path: Kullanıcının verdiği yol (opsiyonel).

        Returns:
            Mevcut dosyanın yolu; hiçbir aday bulunamazsa `None`.
        """
        if meter_csv_path is not None:
            if Path(meter_csv_path).is_file():
                return str(meter_csv_path)
            logger.warning("Belirtilen sayaç CSV dosyası bulunamadı: %s", meter_csv_path)
            return None

        for candidate in (
            PROJECT_ROOT / METER_CSV_NAME,
            PROJECT_ROOT / "data" / METER_CSV_NAME,
        ):
            if candidate.is_file():
                return str(candidate)

        logger.warning("Varsayılan sayaç CSV dosyası bulunamadı (%s).", METER_CSV_NAME)
        return None

    def setup_factory(
        self,
        factory_name: str,
        shift_config: Optional[Dict[str, Any]] = None,
        calibrate_from_csv: bool = True,
        req_med_hours: int = 8,
        req_max_hours: int = 6,
    ) -> Dict[str, Any]:
        """Fabrikayı kurar: (opsiyonel) sayaçtan kalibre eder ve profili veritabanına yazar.

        Args:
            factory_name: Benzersiz fabrika adı.
            shift_config: Esnek vardiya konfigürasyonu. `None` ise varsayılan 3x8 şablonu.
            calibrate_from_csv: `True` ise ve CSV mevcutsa güç parametreleri sayaç
                verisinden kalibre edilir; aksi halde varsayılan kalibre değerler kullanılır.
            req_med_hours: Günlük zorunlu orta hat çalışma saati.
            req_max_hours: Günlük zorunlu ağır hat çalışma saati.

        Returns:
            Veritabanından okunan güncel fabrika profili (`shift_config` sözlük olarak).
        """
        optimizer = LoadOptimizer(shift_config=shift_config)
        optimizer.req_med_hours = req_med_hours
        optimizer.req_max_hours = req_max_hours

        if calibrate_from_csv and self.meter_csv_path is not None:
            try:
                meter_df: pd.DataFrame = pd.read_csv(self.meter_csv_path)
                calibrated = optimizer.calibrate_from_meter(meter_df)
                logger.info("Kalibrasyon tamamlandı (%s): %s", factory_name, calibrated)
            except (ValueError, OSError, pd.errors.ParserError) as exc:
                logger.warning(
                    "Kalibrasyon başarısız (%s); varsayılan değerler kullanılacak: %s",
                    factory_name,
                    exc,
                )
        elif calibrate_from_csv:
            logger.warning(
                "Sayaç CSV dosyası yok; varsayılan kalibre değerler kullanılıyor (%s).",
                factory_name,
            )

        self.db.save_factory_profile(
            factory_name=factory_name,
            p_base_kw=optimizer.p_base_kw,
            p_med_net_kw=optimizer.p_med_net_kw,
            p_max_net_kw=optimizer.p_max_net_kw,
            req_med_hours=optimizer.req_med_hours,
            req_max_hours=optimizer.req_max_hours,
            shift_config=optimizer.shift_config,
        )

        profile = self.db.get_factory_profile(factory_name)
        if profile is None:  # Teorik olarak ulaşılmaz; savunmacı kontrol
            raise RuntimeError(f"Fabrika profili kaydedilemedi: {factory_name}")
        return profile

    def _build_historical_df(self, target_date: str, days: int = HISTORY_DAYS) -> pd.DataFrame:
        """`target_date` öncesindeki `days` günlük saatlik PTF geçmişini hazırlar.

        Her gün için `EpiasClient.fetch_daily_ptf` çağrılır; canlı API'ye erişilemezse
        istemci kendisi simülasyon verisine düşer.

        Args:
            target_date: Tahmin günü (`YYYY-MM-DD`). Geçmiş bu günden bir gün öncesinde biter.
            days: Hazırlanacak geçmiş gün sayısı.

        Returns:
            `target_date`, `hour`, `ptf_try_mwh` sütunlu, kronolojik sıralı DataFrame.
        """
        target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        records: List[Dict[str, Any]] = []
        for offset in range(days, 0, -1):
            day = (target_dt - timedelta(days=offset)).strftime("%Y-%m-%d")
            records.extend(self.epias_client.fetch_daily_ptf(day))

        history = pd.DataFrame(records, columns=["target_date", "hour", "ptf_try_mwh"])
        return history.sort_values(["target_date", "hour"]).reset_index(drop=True)

    def run_price_forecast(
        self,
        target_date: str,
        historical_df: Optional[pd.DataFrame] = None,
        actual_prices: Optional[List[float]] = None,
    ) -> List[float]:
        """Hedef gün için 24 saatlik PTF tahmini üretir ve veritabanına kaydeder.

        Args:
            target_date: Tahmin günü (`YYYY-MM-DD`).
            historical_df: `target_date`, `hour`, `ptf_try_mwh` sütunlu geçmiş veri.
                `None` ise `EpiasClient` üzerinden son `HISTORY_DAYS` gün hazırlanır.
                Verilen veride `target_date` ve sonrası eğitimde kullanılmaz (sızıntı önlemi).
            actual_prices: Opsiyonel, 24 elemanlı gerçekleşen fiyatlar (geriye dönük kayıt için).

        Returns:
            24 elemanlı tahmin listesi (TL/MWh).

        Raises:
            ValueError: `target_date` geçersizse, geçmiş veri yetersizse veya
                `actual_prices` 24 elemanlı değilse.
        """
        try:
            target_dt = datetime.strptime(target_date, "%Y-%m-%d")
        except (ValueError, TypeError) as exc:
            raise ValueError(
                f"Geçersiz target_date: {target_date!r}. Beklenen format: YYYY-MM-DD."
            ) from exc
        target_date = target_dt.strftime("%Y-%m-%d")

        if actual_prices is not None and len(actual_prices) != 24:
            raise ValueError(
                f"actual_prices tam 24 elemanlı olmalıdır (gelen: {len(actual_prices)})."
            )

        if historical_df is None:
            history = self._build_historical_df(target_date)
        else:
            dates = pd.to_datetime(historical_df["target_date"], format="%Y-%m-%d")
            history = historical_df.loc[dates < pd.Timestamp(target_dt)].copy()

        if len(history) < MIN_HISTORY_HOURS:
            logger.warning(
                "Geçmiş veri %d saat; önerilen en az %d saat (lag_168 için).",
                len(history),
                MIN_HISTORY_HOURS,
            )

        self.predictor.train(history)
        forecast_df = self.predictor.predict(target_date, historical_df=history)
        predicted_prices: List[float] = [
            float(v) for v in forecast_df["predicted_ptf_try_mwh"].tolist()
        ]
        if len(predicted_prices) != 24:
            raise ValueError(f"Tahmin 24 saatlik olmalıdır (üretilen: {len(predicted_prices)}).")

        self.db.save_price_forecasts(
            target_date=target_date,
            predicted_prices=predicted_prices,
            actual_prices=actual_prices,
        )
        logger.info("PTF tahmini üretildi ve kaydedildi: %s", target_date)
        return predicted_prices
