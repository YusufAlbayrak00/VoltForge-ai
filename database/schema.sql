CREATE TABLE IF NOT EXISTS factories (
    factory_id        INTEGER   PRIMARY KEY AUTOINCREMENT,
    name              TEXT      NOT NULL,
    industry_type     TEXT,
    contract_power_kw REAL      NOT NULL CHECK (contract_power_kw > 0),
    created_at        TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS machines (
    machine_id           INTEGER   PRIMARY KEY AUTOINCREMENT,
    factory_id           INTEGER   NOT NULL,
    name                 TEXT      NOT NULL,
    power_kw             REAL      NOT NULL CHECK (power_kw > 0), -- saatlik çekiş gücü
    daily_required_hours INTEGER   NOT NULL CHECK (daily_required_hours BETWEEN 0 AND 24), -- günlük zorunlu çalışma süresi
    can_pause            BOOLEAN   DEFAULT 1 CHECK (can_pause IN (0, 1)), -- durdurulup bölünebilir mi?
    min_block_hours      INTEGER   DEFAULT 1 CHECK (min_block_hours BETWEEN 1 AND 24), -- ardışık minimum çalışma saati
    created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (factory_id) REFERENCES factories (factory_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS hourly_tariffs (
    tariff_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    target_date     TEXT    NOT NULL,
    hour            INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    ptf_try_mwh     REAL    NOT NULL, -- TL/MWh
    renewable_ratio REAL    DEFAULT 0.0 CHECK (renewable_ratio BETWEEN 0.0 AND 1.0),
    is_actual       BOOLEAN DEFAULT 1 CHECK (is_actual IN (0, 1)), -- 1: gerçekleşen, 0: tahmin
    UNIQUE (target_date, hour)
);

CREATE TABLE IF NOT EXISTS machine_schedules (
    schedule_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id      INTEGER NOT NULL,
    target_date     TEXT    NOT NULL,
    hour            INTEGER NOT NULL CHECK (hour BETWEEN 0 AND 23),
    is_active       BOOLEAN NOT NULL CHECK (is_active IN (0, 1)),
    is_ai_optimized BOOLEAN DEFAULT 0 CHECK (is_ai_optimized IN (0, 1)),
    cost_try        REAL    DEFAULT 0.0,
    FOREIGN KEY (machine_id) REFERENCES machines (machine_id) ON DELETE CASCADE,
    UNIQUE (machine_id, target_date, hour)
);

-- Finansal kayıt: ON DELETE CASCADE yok, raporu olan fabrika silinemez.
CREATE TABLE IF NOT EXISTS cost_savings_ledger (
    ledger_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    factory_id         INTEGER NOT NULL,
    report_date        TEXT    NOT NULL,
    baseline_cost_try  REAL    NOT NULL,
    optimized_cost_try REAL    NOT NULL,
    net_savings_try    REAL    NOT NULL,
    savings_percentage REAL    NOT NULL,
    FOREIGN KEY (factory_id) REFERENCES factories (factory_id)
);

CREATE INDEX IF NOT EXISTS idx_machines_factory_id
    ON machines (factory_id);
CREATE INDEX IF NOT EXISTS idx_schedules_machine_date
    ON machine_schedules (machine_id, target_date);
CREATE INDEX IF NOT EXISTS idx_ledger_factory_date
    ON cost_savings_ledger (factory_id, report_date);
