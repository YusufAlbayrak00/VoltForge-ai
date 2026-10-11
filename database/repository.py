"""Veri Erişim Katmanı (Repository) Modülü.

Bu modül, Repository tasarım kalıbını (Repository Pattern) uygulayarak;
EPİAŞ saatlik piyasa takas fiyatları (PTF), fabrika makine/yük envanteri,
vardiya çizelgeleri ve optimizasyon çıktıları için CRUD (oluşturma, okuma, güncelleme, silme)
operasyonlarını iş mantığı katmanından soyutlar.
"""

from __future__ import annotations

from pathlib import Path

from database.connection import DatabaseManager

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


class VoltForgeRepository:
    """Fabrika ve makine verilerine erişimi yöneten repository sınıfı.

    SQL sorguları yalnızca bu sınıfın içinde bulunur; üst katmanlar (servisler,
    arayüz) sadece Python metotlarını ve sözlükleri (dict) görür.
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Repository'yi hazırlar ve şemanın var olduğundan emin olur.

        Args:
            db_path: Opsiyonel veritabanı yolu. Verilmezse varsayılan
                `DatabaseManager` yolu (`data/voltforge.db`) kullanılır.
        """
        self.db_path = db_path
        self._initialize_schema()

    def _manager(self) -> DatabaseManager:
        """Her işlem için yeni bir DatabaseManager üretir."""
        return DatabaseManager(self.db_path)

    def _initialize_schema(self) -> None:
        """schema.sql dosyasını uygular (IF NOT EXISTS sayesinde güvenle tekrarlanabilir)."""
        self._manager().execute_script(SCHEMA_PATH)

    def add_factory(self, name: str, industry_type: str, contract_power_kw: float) -> int:
        """Yeni bir fabrika ekler.

        Args:
            name: Fabrika adı.
            industry_type: Sektör türü (örn. "Tekstil").
            contract_power_kw: Sözleşme gücü (kW).

        Returns:
            Eklenen fabrikanın `factory_id` değeri.
        """
        with self._manager() as conn:
            cursor = conn.execute(
                "INSERT INTO factories (name, industry_type, contract_power_kw) VALUES (?, ?, ?)",
                (name, industry_type, contract_power_kw),
            )
            return cursor.lastrowid

    def get_factory(self, factory_id: int) -> dict | None:
        """Verilen ID'ye sahip fabrikayı döndürür; yoksa `None` döner."""
        with self._manager() as conn:
            row = conn.execute(
                "SELECT * FROM factories WHERE factory_id = ?", (factory_id,)
            ).fetchone()
        return dict(row) if row else None

    def list_all_factories(self) -> list[dict]:
        """Kayıtlı tüm fabrikaları ID sırasına göre listeler."""
        with self._manager() as conn:
            rows = conn.execute("SELECT * FROM factories ORDER BY factory_id").fetchall()
        return [dict(row) for row in rows]

    def add_machine(
        self,
        factory_id: int,
        name: str,
        power_kw: float,
        daily_required_hours: int,
        can_pause: bool = True,
        min_block_hours: int = 1,
    ) -> int:
        """Bir fabrikaya yeni makine ekler.

        Args:
            factory_id: Makinenin bağlı olduğu fabrikanın ID'si (mevcut olmalı).
            name: Makine adı.
            power_kw: Saatlik çekiş gücü (kW).
            daily_required_hours: Günlük zorunlu çalışma süresi (saat).
            can_pause: Durdurulup bölünebilir mi?
            min_block_hours: Ardışık minimum çalışma saati.

        Returns:
            Eklenen makinenin `machine_id` değeri.

        Raises:
            sqlite3.IntegrityError: Fabrika yoksa (FK) veya kısıtlar ihlal edilirse.
        """
        with self._manager() as conn:
            cursor = conn.execute(
                """
                INSERT INTO machines
                    (factory_id, name, power_kw, daily_required_hours, can_pause, min_block_hours)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (factory_id, name, power_kw, daily_required_hours, int(can_pause), min_block_hours),
            )
            return cursor.lastrowid

    def get_factory_machines(self, factory_id: int) -> list[dict]:
        """Bir fabrikaya ait tüm makineleri liste olarak döndürür."""
        with self._manager() as conn:
            rows = conn.execute(
                "SELECT * FROM machines WHERE factory_id = ? ORDER BY machine_id",
                (factory_id,),
            ).fetchall()
        return [dict(row) for row in rows]
