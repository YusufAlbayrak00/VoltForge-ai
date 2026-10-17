"""VoltForge AI Kalıcı Veri Katmanı (SQLite DatabaseManager).

Bu modül; `PricePredictor` ve `LoadOptimizer` çıktılarının kalıcı hale getirilmesi için
fabrika profilleri, fiyat tahminleri ve optimizasyon raporları tablolarını yönetir.
"""

import json
import logging
import os
import sqlite3
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class DatabaseManager:
    """SQLite tabanlı veritabanı yöneticisi.

    Her işlem kısa ömürlü bir bağlantı ile yapılır (`with sqlite3.connect(...)`);
    böylece bağlantı sızıntısı önlenir ve hata durumunda işlem otomatik geri alınır.

    Attributes:
        db_path: SQLite veritabanı dosyasının yolu.
    """

    def __init__(self, db_path: str = "data/voltforge.db") -> None:
        """Yöneticiyi başlatır, klasörü oluşturur ve tabloları hazırlar.

        Args:
            db_path: SQLite dosya yolu. Üst klasör yoksa otomatik oluşturulur.
        """
        self.db_path: str = db_path
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        os.makedirs(db_dir, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Gerekli 3 ana tabloyu (yoksa) oluşturur."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS factory_profiles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factory_name TEXT UNIQUE NOT NULL,
                    p_base_kw REAL NOT NULL DEFAULT 34.50,
                    p_med_net_kw REAL NOT NULL DEFAULT 119.28,
                    p_max_net_kw REAL NOT NULL DEFAULT 202.56,
                    req_med_hours INTEGER NOT NULL DEFAULT 8,
                    req_max_hours INTEGER NOT NULL DEFAULT 6,
                    shift_config_json TEXT NOT NULL,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS price_forecasts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    target_date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    predicted_ptf REAL NOT NULL,
                    actual_ptf REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(target_date, hour)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_reports (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    factory_name TEXT NOT NULL,
                    target_date TEXT NOT NULL,
                    baseline_cost_tl REAL,
                    optimized_cost_tl REAL,
                    daily_savings_tl REAL,
                    savings_percentage REAL,
                    annual_savings_tl REAL,
                    total_daily_energy_kwh REAL,
                    schedule_comparison_json TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
        logger.info("Veritabanı şeması hazır: %s", self.db_path)

    def save_factory_profile(
        self,
        factory_name: str,
        p_base_kw: float = 34.50,
        p_med_net_kw: float = 119.28,
        p_max_net_kw: float = 202.56,
        req_med_hours: int = 8,
        req_max_hours: int = 6,
        shift_config: Optional[Dict] = None,
    ) -> int:
        """Fabrika profilini kaydeder; aynı isim varsa günceller (upsert).

        `ON CONFLICT DO UPDATE` kullanıldığı için mevcut kaydın `id` değeri korunur
        (`INSERT OR REPLACE` satırı silip yeniden eklediğinden id değişirdi).

        Args:
            factory_name: Benzersiz fabrika adı.
            p_base_kw: Sabit baz yük (kW).
            p_med_net_kw: Orta hat net gücü (kW).
            p_max_net_kw: Ağır hat net gücü (kW).
            req_med_hours: Günlük zorunlu orta hat çalışma saati.
            req_max_hours: Günlük zorunlu ağır hat çalışma saati.
            shift_config: Vardiya sözlüğü. Verilmezse `LoadOptimizer` içindeki
                varsayılan 3x8 şablonu kullanılır.

        Returns:
            Kaydedilen/güncellenen profilin `id` değeri.
        """
        if shift_config is None:
            # Tek doğruluk kaynağı: varsayılan şablon LoadOptimizer'dan alınır.
            # Ağır bağımlılıklar (scipy/pandas) için import gecikmeli yapılır.
            from ai_core.load_optimizer import LoadOptimizer

            shift_config = LoadOptimizer().shift_config

        shift_config_json = json.dumps(shift_config, ensure_ascii=False)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO factory_profiles (
                    factory_name, p_base_kw, p_med_net_kw, p_max_net_kw,
                    req_med_hours, req_max_hours, shift_config_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(factory_name) DO UPDATE SET
                    p_base_kw = excluded.p_base_kw,
                    p_med_net_kw = excluded.p_med_net_kw,
                    p_max_net_kw = excluded.p_max_net_kw,
                    req_med_hours = excluded.req_med_hours,
                    req_max_hours = excluded.req_max_hours,
                    shift_config_json = excluded.shift_config_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (
                    factory_name,
                    p_base_kw,
                    p_med_net_kw,
                    p_max_net_kw,
                    req_med_hours,
                    req_max_hours,
                    shift_config_json,
                ),
            )
            # Güncelleme durumunda cursor.lastrowid güvenilir olmadığından id sorgulanır.
            row = conn.execute(
                "SELECT id FROM factory_profiles WHERE factory_name = ?",
                (factory_name,),
            ).fetchone()

        profile_id = int(row[0])
        logger.info("Fabrika profili kaydedildi: %s (id=%d)", factory_name, profile_id)
        return profile_id

    def get_factory_profile(self, factory_name: str) -> Optional[Dict[str, Any]]:
        """Fabrika profilini veritabanından okur.

        Args:
            factory_name: Aranacak fabrika adı.

        Returns:
            `shift_config_json` alanı `shift_config` sözlüğüne çevrilmiş profil
            sözlüğü; kayıt bulunamazsa `None`.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM factory_profiles WHERE factory_name = ?",
                (factory_name,),
            ).fetchone()

        if row is None:
            return None

        profile: Dict[str, Any] = dict(row)
        profile["shift_config"] = json.loads(profile.pop("shift_config_json"))
        return profile
