"""EPİAŞ Şeffaflık Platformu API İstemcisi Modülü.

Bu modül, Enerji Piyasaları İşletme A.Ş. (EPİAŞ) Şeffaflık Platformu API'si ile
güvenli iletişim kurarak; Gün Öncesi Piyasası (GÖP) saatlik Piyasa Takas Fiyatı (PTF)
ve ilgili şebeke parametrelerini çekmek, yanıtları normalize etmek ve olası ağ/veri
hatalarını yönetmekten sorumludur.
"""

import random
from datetime import datetime


class EpiasClient:
    """EPİAŞ Şeffaflık Platformu için istemci sınıfı (iskelet).

    Şu aşamada yalnızca canlı API'ye erişilemediği durumlarda kullanılacak
    yedek (fallback) sahte veri üretici içerir. Gerçek API istekleri sonraki
    adımlarda eklenecektir.
    """

    # Saatlik baz PTF profili (TL/MWh), indeks = saat (00-23).
    # Gece vadisi (~03:00), sabah piki (~08-09) ve akşam piki (~19:00).
    _BASE_PRICE_PROFILE: list[float] = [
        1900.0, 1750.0, 1650.0, 1600.0, 1650.0, 1800.0,  # 00-05
        2200.0, 2700.0, 3000.0, 2900.0, 2600.0, 2450.0,  # 06-11
        2400.0, 2350.0, 2300.0, 2400.0, 2600.0, 2900.0,  # 12-17
        3200.0, 3400.0, 3250.0, 2900.0, 2500.0, 2150.0,  # 18-23
    ]

    # Gürültü genliği: baz fiyatın ± %8'i kadar rastgele sapma.
    _NOISE_RATIO: float = 0.08

    def _generate_mock_data(self, target_date: str) -> list[dict]:
        """Belirtilen tarih için 24 saatlik gerçekçi sahte PTF verisi üretir.

        Canlı API'ye bağlanılamadığında yedek veri kaynağı olarak kullanılır.
        Fiyatlar gece 03:00 civarında en düşük, sabah (08:00-09:00) ve akşam
        (19:00) saatlerinde pik yapacak şekilde simüle edilir. Aynı tarih için
        her çağrıda aynı sonuç üretilir (tarih, rastgele üreteç tohumu olarak
        kullanılır), böylece testler tekrarlanabilir olur.

        Args:
            target_date: Hedef tarih, ``YYYY-MM-DD`` formatında (ör. ``"2026-10-12"``).

        Returns:
            24 elemanlı bir liste. Her eleman şu anahtarları içeren sözlüktür:
                - ``date`` (str): ``YYYY-MM-DD`` formatında tarih.
                - ``hour`` (str): ``HH:00`` formatında saat.
                - ``ptf`` (float): Piyasa Takas Fiyatı (TL/MWh), 2 basamağa yuvarlanmış.

        Raises:
            ValueError: ``target_date`` beklenen formatta değilse.
        """
        parsed_date = datetime.strptime(target_date, "%Y-%m-%d").date()
        rng = random.Random(parsed_date.isoformat())

        mock_data: list[dict] = []
        for hour, base_price in enumerate(self._BASE_PRICE_PROFILE):
            noise = rng.uniform(-self._NOISE_RATIO, self._NOISE_RATIO)
            price = round(base_price * (1 + noise), 2)
            mock_data.append(
                {
                    "date": parsed_date.isoformat(),
                    "hour": f"{hour:02d}:00",
                    "ptf": price,
                }
            )
        return mock_data
