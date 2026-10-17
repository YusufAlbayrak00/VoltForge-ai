"""VoltForge AI - Komut satırı ana çalıştırma noktası.

Örnek bir fabrika için kurulum, 24 saatlik LightGBM PTF tahmini ve MILP vardiya
optimizasyonunu uçtan uca çalıştırır; sonuçları özet tablo olarak ekrana yazar.

Kullanım:
    python main.py
    python main.py --factory Daewoo_Steel_OSB --date 2026-10-18 --db :memory:
"""

import argparse
import logging
from datetime import date, timedelta
from typing import Any, Dict, Optional

from ai_core.pipeline import VoltForgePipeline

DEFAULT_FACTORY: str = "Daewoo_Steel_OSB"
LINE_WIDTH: int = 62


def format_tl(value: Optional[float]) -> str:
    """TL tutarını binlik ayraçlı biçimde döndürür (`None` ise `-`)."""
    return "-" if value is None else f"{value:,.2f} TL"


def print_summary(result: Dict[str, Any]) -> None:
    """Optimizasyon sonucunu özet tablo olarak yazdırır.

    Args:
        result: `VoltForgePipeline.run_daily_optimization` çıktısı.
    """
    prices = result["ptf_prices"]
    shifts = result.get("active_shifts") or []
    pct = result.get("savings_percentage")
    energy = result.get("total_daily_energy_kwh")

    rows = [
        ("Fabrika", result["factory_name"]),
        ("Hedef Gün", result["target_date"]),
        ("Rapor ID", str(result["report_id"])),
        ("PTF Aralığı (TL/MWh)", f"{min(prices):,.2f} - {max(prices):,.2f}"),
        ("Günlük Enerji", "-" if energy is None else f"{energy:,.2f} kWh"),
        ("Geleneksel Fatura", format_tl(result.get("baseline_cost_tl"))),
        ("Optimize Fatura", format_tl(result.get("optimized_cost_tl"))),
        ("Günlük Tasarruf", format_tl(result.get("daily_savings_tl"))),
        ("Yıllık Tasarruf (260 gün)", format_tl(result.get("annual_savings_tl"))),
        ("Yüzdesel Kazanç", "-" if pct is None else f"% {pct:.2f}"),
    ]

    print("=" * LINE_WIDTH)
    print(" VoltForge AI - Günlük Enerji Optimizasyon Özeti".center(LINE_WIDTH))
    print("=" * LINE_WIDTH)
    for label, value in rows:
        print(f" {label:<27}| {value}")
    print("-" * LINE_WIDTH)
    print(" Aktif Vardiyalar:")
    if shifts:
        for name in shifts:
            print(f"   - {name}")
    else:
        print(f"   (çözüm bulunamadı: {result.get('status')})")
    print("=" * LINE_WIDTH)


def main() -> None:
    """Komut satırı argümanlarını okuyup pipeline'ı çalıştırır."""
    parser = argparse.ArgumentParser(description="VoltForge AI günlük optimizasyon")
    parser.add_argument("--factory", default=DEFAULT_FACTORY, help="Fabrika adı")
    parser.add_argument(
        "--date",
        default=(date.today() + timedelta(days=1)).isoformat(),
        help="Hedef gün (YYYY-MM-DD); varsayılan: yarın",
    )
    parser.add_argument(
        "--db", default=None, help="SQLite yolu (varsayılan: data/voltforge.db, ':memory:' olabilir)"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    pipeline = VoltForgePipeline(db_path=args.db)
    try:
        profile = pipeline.setup_factory(args.factory)
        print(
            f"Fabrika kuruldu: {profile['factory_name']} "
            f"(baz {profile['p_base_kw']} kW, orta +{profile['p_med_net_kw']} kW, "
            f"ağır +{profile['p_max_net_kw']} kW)"
        )
        pipeline.run_price_forecast(args.date)
        result = pipeline.run_daily_optimization(args.factory, args.date)
        print_summary(result)
    finally:
        pipeline.db.close()


if __name__ == "__main__":
    main()
