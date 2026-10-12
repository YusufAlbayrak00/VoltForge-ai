"""EPİAŞ Şeffaflık Platformu API İstemcisi Modülü.

Bu modül, Enerji Piyasaları İşletme A.Ş. (EPİAŞ) Şeffaflık Platformu API'si ile
güvenli iletişim kurarak; Gün Öncesi Piyasası (GÖP) saatlik Piyasa Takas Fiyatı (PTF)
ve ilgili şebeke parametrelerini çekmek, yanıtları normalize etmek ve olası ağ/veri
hatalarını yönetmekten sorumludur.
"""

import random
from datetime import datetime

import requests


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

    def fetch_daily_ptf(self, target_date: str) -> list[dict]:
        """Belirtilen tarih için saatlik PTF verisini EPİAŞ GÖP API'sinden çeker.

        Önce EPİAŞ Şeffaflık Platformu GÖP uç noktasına istek atmayı dener.
        Zaman aşımı, bağlantı hatası, HTTP hata kodu (4xx/5xx), geçersiz JSON
        veya boş/beklenmeyen yanıt gibi herhangi bir sorunda program çökmez;
        konsola uyarı basılır ve ``_generate_mock_data`` ile üretilen
        simülasyon verisi döndürülür.

        Not:
            Kullanılan URL şimdilik temsilidir; gerçek uç nokta ve yanıt şeması
            sonraki adımlarda netleştirilip yanıt normalize edilecektir.

        Args:
            target_date: Hedef tarih, ``YYYY-MM-DD`` formatında (ör. ``"2026-10-12"``).

        Returns:
            Saatlik PTF kayıtlarını içeren sözlük listesi. Canlı veri alınırsa
            API yanıtındaki kayıtlar, alınamazsa 24 elemanlı simülasyon verisi
            (``date``, ``hour``, ``ptf`` anahtarlı) döner.

        Raises:
            ValueError: ``target_date`` ``YYYY-MM-DD`` formatında değilse
                (bu bir çağıran taraf hatasıdır, yedek veriye düşülmez).
        """
        # Hatalı tarih formatını ağ isteği atmadan önce yakala.
        datetime.strptime(target_date, "%Y-%m-%d")

        api_url = "https://seffaflik.epias.com.tr/api/market/gop"
        timeout_seconds = 10

        try:
            response = requests.get(
                api_url,
                params={"date": target_date},
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()

            if not isinstance(data, list) or not data:
                raise ValueError("API yanıtı beklenen formatta değil veya boş.")

            return data
        except (requests.exceptions.RequestException, ValueError) as error:
            print(
                "Canlı veri alınamadı, simülasyon verisi kullanılıyor... "
                f"(Sebep: {type(error).__name__}: {error})"
            )
            return self._generate_mock_data(target_date)
