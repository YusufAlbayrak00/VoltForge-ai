"""VoltForge AI Ana Orkestrasyon Hattı (Pipeline).

Bu modül; `PricePredictor` (fiyat tahmini), `LoadOptimizer` (MILP yük optimizasyonu)
ve `DatabaseManager` (kalıcı veri katmanı) bileşenlerini tek bir akışta birbirine bağlar.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from ai_core.load_optimizer import LoadOptimizer
from ai_core.price_predictor import PricePredictor
from database.db_manager import DatabaseManager

logger = logging.getLogger(__name__)

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
METER_CSV_NAME: str = "Steel_industry_data.csv"


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
