"""Endüstriyel Yük Çizelgeleme ve Enerji Arbitraj Optimizatörü (Load Optimizer).

Bu modül; organize sanayi tesislerinin makine güç tüketimleri, kesilebilir/esnek yükleri,
asgari/azami çalışma süreleri, vardiya kısıtları ve üretim kotalarını dikkate alarak;
saatlik elektrik fiyatlarına göre fabrikanın enerji maliyetini en aza indiren
matematiksel optimizasyon ve yük kaydırma (load shifting / arbitraj) algoritmalarını yürütür.
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp


class LoadOptimizer:
    """Karışık Tamsayılı Doğrusal Programlama (MILP) tabanlı yük optimizatörü.

    Fabrikanın güç profili (baz / orta / ağır hat) ve esnek vardiya konfigürasyonu
    parametrik olarak tutulur; böylece farklı OSB tesisleri (3x8, 2x12, gece
    vardiyası yasaklı vb.) kod değiştirilmeden yalnızca konfigürasyonla modellenebilir.

    Attributes:
        p_base_kw: 24 saat kesintisiz çalışan sabit baz yük (Light_Load), kW.
        p_med_net_kw: Baz yükün üstüne binen orta üretim hattı net gücü, kW.
        p_max_net_kw: Baz yükün üstüne binen ağır üretim hattı net gücü, kW.
        req_med_hours: Günlük zorunlu orta hat çalışma saati.
        req_max_hours: Günlük zorunlu ağır hat çalışma saati.
        shift_config: Vardiya şablonu (vardiyalar, izinli vardiya sayısı, mola süresi).
    """

    def __init__(self, shift_config: Optional[Dict] = None) -> None:
        """Optimizatörü varsayılan fabrika parametreleri ve vardiya şablonuyla başlatır.

        Varsayılan güç değerleri `Steel_industry_data.csv` üzerinden kalibre edilmiş
        referans değerleridir ve `calibrate_from_meter` ile gerçek sayaç verisinden
        güncellenebilir.

        Args:
            shift_config: Opsiyonel vardiya konfigürasyonu. Verilmezse 4857 sayılı
                İş Kanunu'na uygun klasik 3x8 vardiya şablonu kullanılır. Beklenen yapı:
                `{"shifts": [{"name": str, "hours": List[int], "allowed": bool,
                "extra_cost": float (opsiyonel, TL)}, ...],
                "max_active_shifts": int, "break_hours_per_shift": int}`.
        """
        # --- Varsayılan fabrika güç parametreleri (Steel_industry_data.csv kalibre değerleri) ---
        self.p_base_kw: float = 34.50    # Light_Load baz yük
        self.p_med_net_kw: float = 119.28  # Medium_Load net ek güç (153.78 - 34.50)
        self.p_max_net_kw: float = 202.56  # Maximum_Load net ek güç (237.06 - 34.50)
        self.req_med_hours: int = 8
        self.req_max_hours: int = 6

        # --- Esnek vardiya konfigürasyonu ---
        self.shift_config: Dict = shift_config or {
            "shifts": [
                {"name": "Gece Vardiyası (00:00-08:00)", "hours": [0, 1, 2, 3, 4, 5, 6, 7], "allowed": True},
                {"name": "Gündüz Vardiyası (08:00-16:00)", "hours": [8, 9, 10, 11, 12, 13, 14, 15], "allowed": True},
                {"name": "Akşam Vardiyası (16:00-24:00)", "hours": [16, 17, 18, 19, 20, 21, 22, 23], "allowed": True}
            ],
            "max_active_shifts": 2,
            "break_hours_per_shift": 1
        }

    def calibrate_from_meter(self, df: pd.DataFrame) -> Dict[str, float]:
        """Ham sayaç verisinden fabrika güç parametrelerini otomatik kalibre eder.

        Ham (ör. 15 dakikalık) kayıtlar doğrudan `Load_Type` bazında gruplanır ve
        ortalama `Usage_kWh` değeri saatlik güce (kW) çevrilir (15 dk için x4). Bu yöntem
        vardiya/yük tipi geçiş saatlerinde veri kaybı yaşatmaz. Ardından:
        `Light_Load` -> baz yük, `Medium_Load - Light_Load` -> orta hat net gücü,
        `Maximum_Load - Light_Load` -> ağır hat net gücü hesaplanır ve 2 ondalığa
        yuvarlanarak nesne özelliklerine yazılır.

        Args:
            df: En az `Usage_kWh` ve `Load_Type` sütunlarını içeren sayaç verisi.
                `date` sütunu (gün/ay/yıl) varsa kayıt aralığı buradan çıkarılır;
                yoksa ya da çıkarılamazsa 15 dakika varsayılır.

        Returns:
            `{"p_base_kw": ..., "p_med_net_kw": ..., "p_max_net_kw": ...}` sözlüğü.

        Raises:
            ValueError: Gerekli sütunlar eksikse ya da üç yük tipinden biri için veri yoksa.
        """
        required = {"Usage_kWh", "Load_Type"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Sayaç verisinde eksik sütun(lar): {sorted(missing)}")

        data = df.copy()
        data["Usage_kWh"] = pd.to_numeric(data["Usage_kWh"], errors="coerce")
        data["Load_Type"] = data["Load_Type"].astype(str).str.strip()
        data = data.dropna(subset=["Usage_kWh"])

        # Kayıt aralığını (dakika) tespit et; varsayılan 15 dk
        interval_min: float = 15.0
        if "date" in data.columns:
            ts = pd.to_datetime(data["date"], dayfirst=True, errors="coerce").dropna()
            if len(ts) > 1:
                median_step = ts.diff().dropna().median()
                if pd.notna(median_step) and median_step > pd.Timedelta(0):
                    interval_min = median_step.total_seconds() / 60.0
        kw_factor = 60.0 / interval_min  # 15 dk -> x4

        mean_kw = data.groupby("Load_Type")["Usage_kWh"].mean() * kw_factor

        for load_type in ("Light_Load", "Medium_Load", "Maximum_Load"):
            if load_type not in mean_kw.index:
                raise ValueError(f"Sayaç verisinde '{load_type}' kaydı bulunamadı.")

        base = float(mean_kw["Light_Load"])
        self.p_base_kw = round(base, 2)
        self.p_med_net_kw = round(max(float(mean_kw["Medium_Load"]) - base, 0.0), 2)
        self.p_max_net_kw = round(max(float(mean_kw["Maximum_Load"]) - base, 0.0), 2)

        return {
            "p_base_kw": self.p_base_kw,
            "p_med_net_kw": self.p_med_net_kw,
            "p_max_net_kw": self.p_max_net_kw,
        }

    def optimize_schedule(self, ptf_prices: List[float]) -> Dict:
        """24 saatlik PTF fiyatlarına göre maliyet-minimum üretim çizelgesini hesaplar.

        SciPy HiGHS tabanlı `milp` ile ikili (binary) karar değişkenli model çözülür:
        `x_max[t]` (0..23), `x_med[t]` (24..47) ve `y_shift[s]` (48..48+S-1).
        Kısıtlar: günlük ağır/orta hat çalışma saati kotaları, aktif vardiya sayısı
        (1..`max_active_shifts`), vardiya-saat bağlantısı + hat çakışma engeli ve
        vardiya başına zorunlu mola (4857 sayılı İş Kanunu). `allowed=False` vardiyalar
        ve hiçbir vardiyanın kapsamadığı saatler üst sınır 0 ile kapatılır.

        Args:
            ptf_prices: Saatlik Piyasa Takas Fiyatı (PTF) listesi (tam 24 eleman, TL/MWh).

        Returns:
            Çözüm başarılıysa `status`, `optimized_cost_tl` (baz yük dahil), `total_energy_kwh`,
            `active_shifts`, `hourly_load_kw` ve `hourly_mode` anahtarlarını içeren sözlük.
            Çözüm bulunamazsa (infeasible vb.) `status` mesajı ile birlikte sayısal alanlar
            `None`, liste alanları boş döner.

        Raises:
            ValueError: `ptf_prices` tam 24 adet sayısal değer içermiyorsa.
        """
        # --- 1. Girdi doğrulaması ---
        if ptf_prices is None or len(ptf_prices) != 24:
            raise ValueError("ptf_prices tam olarak 24 adet saatlik fiyat içermelidir.")
        try:
            ptf_arr = np.asarray(ptf_prices, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("ptf_prices yalnızca sayısal değerler içermelidir.") from exc
        if not np.all(np.isfinite(ptf_arr)):
            raise ValueError("ptf_prices NaN veya sonsuz değer içeremez.")

        ptf_kwh = ptf_arr / 1000.0  # TL/MWh -> TL/kWh

        shifts: List[Dict] = self.shift_config["shifts"]
        n_shifts = len(shifts)
        max_active = int(self.shift_config["max_active_shifts"])
        break_hours = int(self.shift_config["break_hours_per_shift"])
        n_vars = 48 + n_shifts
        max_idx = lambda t: t          # noqa: E731  x_max[t]
        med_idx = lambda t: 24 + t     # noqa: E731  x_med[t]
        shift_idx = lambda s: 48 + s   # noqa: E731  y_shift[s]

        # --- 3. Amaç fonksiyonu (maliyet minimizasyonu) ---
        c = np.zeros(n_vars)
        c[0:24] = ptf_kwh * self.p_max_net_kw
        c[24:48] = ptf_kwh * self.p_med_net_kw
        for s, shift in enumerate(shifts):
            c[shift_idx(s)] = float(shift.get("extra_cost", 0.0))
        base_cost = float(np.sum(ptf_kwh * self.p_base_kw))  # Çözüm sonrası faturaya eklenir

        # --- 4. Kısıtlar ---
        rows: List[np.ndarray] = []
        lbs: List[float] = []
        ubs: List[float] = []

        def add_row(coefs: Dict[int, float], lb: float, ub: float) -> None:
            row = np.zeros(n_vars)
            for idx, val in coefs.items():
                row[idx] += val
            rows.append(row)
            lbs.append(lb)
            ubs.append(ub)

        # Günlük üretim hedefleri
        add_row({max_idx(t): 1.0 for t in range(24)}, self.req_max_hours, self.req_max_hours)
        add_row({med_idx(t): 1.0 for t in range(24)}, self.req_med_hours, self.req_med_hours)

        # Aktif vardiya sayısı: 1 <= sum(y) <= max_active_shifts
        add_row({shift_idx(s): 1.0 for s in range(n_shifts)}, 1, max_active)

        for s, shift in enumerate(shifts):
            s_hours = list(shift["hours"])
            # Vardiya-saat bağlantısı + hat çakışma engeli: x_max[h] + x_med[h] - y[s] <= 0
            for h in s_hours:
                add_row({max_idx(h): 1.0, med_idx(h): 1.0, shift_idx(s): -1.0}, -np.inf, 0.0)
            # Zorunlu mola: toplam çalışma <= (vardiya saati - mola) * y[s]
            coefs = {}
            for h in s_hours:
                coefs[max_idx(h)] = 1.0
                coefs[med_idx(h)] = 1.0
            coefs[shift_idx(s)] = -float(len(s_hours) - break_hours)
            add_row(coefs, -np.inf, 0.0)

        constraints = LinearConstraint(np.vstack(rows), np.array(lbs), np.array(ubs))

        # --- Sınırlar: yasaklı vardiyalar ve hiçbir vardiyanın kapsamadığı saatler ---
        lower = np.zeros(n_vars)
        upper = np.ones(n_vars)
        for s, shift in enumerate(shifts):
            if not shift.get("allowed", True):
                upper[shift_idx(s)] = 0.0
        covered_hours = {h for shift in shifts for h in shift["hours"]}
        for t in range(24):
            if t not in covered_hours:
                upper[max_idx(t)] = 0.0
                upper[med_idx(t)] = 0.0
        bounds = Bounds(lower, upper)

        # --- Çözüm (HiGHS MILP) ---
        res = milp(c=c, constraints=constraints, bounds=bounds, integrality=np.ones(n_vars))

        if not res.success or res.x is None:
            return {
                "status": str(res.message),
                "optimized_cost_tl": None,
                "total_energy_kwh": None,
                "active_shifts": [],
                "hourly_load_kw": [],
                "hourly_mode": [],
            }

        # --- 5. Çıktı ---
        x = np.round(res.x).astype(int)
        x_max = x[0:24]
        x_med = x[24:48]
        y_shift = x[48:48 + n_shifts]

        hourly_load = self.p_base_kw + x_max * self.p_max_net_kw + x_med * self.p_med_net_kw
        hourly_mode = [
            "Maximum_Load" if x_max[t] else ("Medium_Load" if x_med[t] else "Light_Load")
            for t in range(24)
        ]
        total_cost = float(res.fun) + base_cost  # Vardiya ek maliyetleri (varsa) dahil

        return {
            "status": str(res.message),
            "optimized_cost_tl": round(total_cost, 2),
            "total_energy_kwh": round(float(np.sum(hourly_load)), 2),
            "active_shifts": [shifts[s]["name"] for s in range(n_shifts) if y_shift[s] == 1],
            "hourly_load_kw": [round(float(v), 2) for v in hourly_load],
            "hourly_mode": hourly_mode,
        }

    def validate_capacity(self) -> Dict[str, Any]:
        """Vardiya kapasitesinin günlük üretim kotasını karşılayıp karşılamadığını denetler.

        İzin verilen (`allowed=True`) vardiyalar arasından net çalışma saati en yüksek olan
        `max_active_shifts` kadar vardiya seçilir. Her vardiyanın net kapasitesi
        `max(0, saat sayısı - mola saati)` olarak hesaplanır ve toplanır. Bu toplam,
        `req_max_hours + req_med_hours` kotasıyla karşılaştırılır.

        Returns:
            Şu anahtarları içeren sözlük: `is_feasible` (bool), `total_capacity_hours` (int),
            `required_quota_hours` (int), `slack_hours` (int, kapasite - kota) ve
            `warning_message` (str | None; kapasite yetersizse Türkçe açıklama, aksi halde None).
        """
        break_hours = int(self.shift_config["break_hours_per_shift"])
        max_active = int(self.shift_config["max_active_shifts"])

        net_hours = sorted(
            (
                max(0, len(s["hours"]) - break_hours)
                for s in self.shift_config["shifts"]
                if s.get("allowed", True)
            ),
            reverse=True,
        )
        total_capacity = int(sum(net_hours[:max_active]))
        required_quota = int(self.req_max_hours + self.req_med_hours)
        slack = total_capacity - required_quota
        is_feasible = total_capacity >= required_quota

        warning: Optional[str] = None
        if not is_feasible:
            warning = (
                f"Vardiya kapasitesi yetersiz: izin verilen en uygun {max_active} vardiyanın "
                f"mola düşüldükten sonraki net çalışma kapasitesi {total_capacity} saat, "
                f"günlük zorunlu üretim kotası ise {required_quota} saat "
                f"({self.req_max_hours} ağır + {self.req_med_hours} orta hat). "
                f"{-slack} saatlik açık nedeniyle optimizasyon çözümsüz olacaktır. "
                "Çözüm için aktif vardiya sayısını artırın, yasaklı vardiyaları açın, "
                "mola süresini azaltın veya üretim kotasını düşürün."
            )

        return {
            "is_feasible": is_feasible,
            "total_capacity_hours": total_capacity,
            "required_quota_hours": required_quota,
            "slack_hours": slack,
            "warning_message": warning,
        }

    def _build_baseline_schedule(self, ptf_prices: List[float]) -> Dict[str, Any]:
        """Fiyatlardan bağımsız, geleneksel (gündüz ağırlıklı) üretim çizelgesini simüle eder.

        Toplam enerji MILP çözümüyle birebir aynı kalacak şekilde `req_max_hours` kadar saat
        önce gündüz pik saatlerine (`10, 11, 13, 14, 15, 16, 9, 17, ...`), `req_med_hours`
        kadar saat ise kalan gündüz/akşam saatlerine (08:00-22:00), yetmezse kalan
        saatlere atanır. Geleneksel çizelge vardiya kısıtlarını dikkate almaz.

        Args:
            ptf_prices: Saatlik PTF listesi (24 eleman, TL/MWh).

        Returns:
            `baseline_load_kw` (24 saatlik yük listesi), `baseline_mode` (saatlik mod listesi),
            `baseline_cost_tl` (baz yük dahil günlük fatura) ve `total_energy_kwh`.

        Raises:
            ValueError: `ptf_prices` 24 eleman değilse veya kota 24 saati aşıyorsa.
        """
        if ptf_prices is None or len(ptf_prices) != 24:
            raise ValueError("ptf_prices tam olarak 24 adet saatlik fiyat içermelidir.")
        if self.req_max_hours + self.req_med_hours > 24:
            raise ValueError("Toplam üretim kotası 24 saati aşamaz; geleneksel çizelge kurulamaz.")

        ptf_kwh = np.asarray(ptf_prices, dtype=float) / 1000.0

        # Ağır hat önceliği: gündüz pik saatleri, sonra kalan gündüz/akşam, sonra gece
        max_priority: List[int] = [10, 11, 13, 14, 15, 16, 9, 17, 12, 8, 18, 19, 20, 21, 22, 23] + list(range(0, 8))
        max_hours = max_priority[: self.req_max_hours]

        # Orta hat: kalan 08:00-22:00 saatleri, yetmezse kalan diğer saatler
        remaining = [h for h in range(24) if h not in max_hours]
        med_pool = [h for h in remaining if 8 <= h < 22] + [h for h in remaining if not 8 <= h < 22]
        med_hours = med_pool[: self.req_med_hours]

        load = np.full(24, self.p_base_kw, dtype=float)
        mode = ["Light_Load"] * 24
        for h in max_hours:
            load[h] += self.p_max_net_kw
            mode[h] = "Maximum_Load"
        for h in med_hours:
            load[h] += self.p_med_net_kw
            mode[h] = "Medium_Load"

        return {
            "baseline_load_kw": [round(float(v), 2) for v in load],
            "baseline_mode": mode,
            "baseline_cost_tl": round(float(np.sum(load * ptf_kwh)), 2),
            "total_energy_kwh": round(float(np.sum(load)), 2),
        }

    def compare_with_baseline(self, ptf_prices: List[float]) -> Dict[str, Any]:
        """Optimize çizelgeyi geleneksel çizelgeyle karşılaştırıp tasarruf raporu üretir.

        Önce `validate_capacity` ve `optimize_schedule` çağrılır. Çözüm bulunamazsa
        (`optimized_cost_tl is None`) hata vermeden `status`, `capacity_check` ve
        `None` değerli maliyet/tasarruf alanlarıyla döner.

        Args:
            ptf_prices: Saatlik PTF listesi (24 eleman, TL/MWh).

        Returns:
            `status`, `baseline_cost_tl`, `optimized_cost_tl`, `daily_savings_tl`,
            `savings_percentage`, `annual_savings_tl` (260 iş günü), `total_daily_energy_kwh`,
            `energy_equal` (iki senaryonun enerjisi eşit mi), `capacity_check` ve
            `schedule_comparison` (geleneksel ve optimize saatlik yük listeleri) anahtarlarını
            içeren sözlük.

        Raises:
            ValueError: `ptf_prices` geçersizse (`optimize_schedule` tarafından).
        """
        capacity = self.validate_capacity()
        optimized = self.optimize_schedule(ptf_prices)

        if optimized.get("optimized_cost_tl") is None:
            return {
                "status": optimized.get("status"),
                "capacity_check": capacity,
                "baseline_cost_tl": None,
                "optimized_cost_tl": None,
                "daily_savings_tl": None,
                "savings_percentage": None,
                "annual_savings_tl": None,
                "total_daily_energy_kwh": None,
                "energy_equal": None,
                "schedule_comparison": None,
            }

        baseline = self._build_baseline_schedule(ptf_prices)
        baseline_cost = float(baseline["baseline_cost_tl"])
        optimized_cost = float(optimized["optimized_cost_tl"])

        daily_savings = baseline_cost - optimized_cost
        savings_pct = (daily_savings / baseline_cost * 100.0) if baseline_cost > 0 else 0.0
        energy_equal = math.isclose(
            baseline["total_energy_kwh"], optimized["total_energy_kwh"], abs_tol=0.01
        )

        return {
            "status": optimized["status"],
            "capacity_check": capacity,
            "baseline_cost_tl": round(baseline_cost, 2),
            "optimized_cost_tl": round(optimized_cost, 2),
            "daily_savings_tl": round(daily_savings, 2),
            "savings_percentage": round(savings_pct, 2),
            "annual_savings_tl": round(daily_savings * 260, 2),
            "total_daily_energy_kwh": optimized["total_energy_kwh"],
            "energy_equal": energy_equal,
            "schedule_comparison": {
                "baseline_load_kw": baseline["baseline_load_kw"],
                "optimized_load_kw": optimized["hourly_load_kw"],
            },
        }
