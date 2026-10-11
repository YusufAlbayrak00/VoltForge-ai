"""Veritabanı Bağlantı Yönetimi Modülü.

Bu modül, VoltForge AI platformunun SQLite (veya gelecekteki ilişkisel veritabanı)
bağlantı yaşam döngüsünü, oturum yönetimini ve context manager yapılarını yönetmekten sorumludur.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "voltforge.db"


class DatabaseManager:
    """SQLite bağlantısını `with` bloğu ile güvenle yöneten sınıf.

    Kullanım:
        with DatabaseManager() as conn:
            conn.execute("SELECT ...")

    Davranış:
        - Blok başarıyla biterse   -> commit
        - Blok içinde hata olursa  -> rollback (hata yukarı fırlatılır)
        - Her durumda              -> bağlantı kapatılır
    """

    def __init__(self, db_path: str | Path | None = None) -> None:
        """Yöneticiyi hazırlar; bağlantı henüz açılmaz.

        Args:
            db_path: Veritabanı dosya yolu. Verilmezse `data/voltforge.db` kullanılır.
                ":memory:" verilirse geçici bellek içi veritabanı oluşturulur (testler için).
        """
        self.db_path = str(db_path) if db_path is not None else str(DEFAULT_DB_PATH)
        self.conn: sqlite3.Connection | None = None

    def _connect(self) -> sqlite3.Connection:
        """Yeni bir SQLite bağlantısı açar ve yapılandırır."""
        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        # SQLite'ta FK denetimi varsayılan KAPALIDIR; her bağlantıda açılmalıdır.
        conn.execute("PRAGMA foreign_keys = ON;")
        return conn

    def __enter__(self) -> sqlite3.Connection:
        """`with` bloğuna girerken bağlantıyı açar ve döndürür."""
        self.conn = self._connect()
        return self.conn

    def __exit__(self, exc_type, exc_value, traceback) -> bool:
        """`with` bloğundan çıkarken işlemi sonlandırır ve bağlantıyı kapatır.

        Returns:
            False: Blok içindeki hatalar bastırılmaz, çağırana iletilir.
        """
        if self.conn is None:
            return False
        try:
            if exc_type is None:
                self.conn.commit()
            else:
                self.conn.rollback()
        finally:
            self.conn.close()
            self.conn = None
        return False

    def execute_script(self, sql_file_path: str | Path) -> None:
        """Bir `.sql` dosyasındaki tüm komutları veritabanında çalıştırır.

        Yöneticinin `with` bloğu içinde (açık bağlantı varken) kullanılması durumunda
        o bağlantıyı kullanır; aksi halde kendi geçici bağlantısını açıp kapatır.

        Args:
            sql_file_path: Çalıştırılacak SQL dosyasının yolu (örn. database/schema.sql).

        Raises:
            FileNotFoundError: Dosya bulunamazsa.
            sqlite3.Error: SQL hatası olursa.
        """
        sql_path = Path(sql_file_path)
        if not sql_path.is_file():
            raise FileNotFoundError(f"SQL dosyası bulunamadı: {sql_path}")

        script = sql_path.read_text(encoding="utf-8")

        if self.conn is not None:
            self.conn.executescript(script)
        else:
            with self:
                self.conn.executescript(script)
