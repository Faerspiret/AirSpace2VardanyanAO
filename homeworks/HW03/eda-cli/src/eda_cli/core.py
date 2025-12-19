from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
from pandas.api import types as ptypes


@dataclass
class SvodkaKolonki:
    """Сводная информация по одной колонке данных"""
    nazvanie: str
    tip_dannyh: str
    zapolnennyh: int
    propuskov: int
    dolya_propuskov: float
    unikalnyh: int
    primer_znacheniy: List[Any]
    chislovaya: bool
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    srednee: Optional[float] = None
    standartnoe_otklonenie: Optional[float] = None

    def v_slovar(self) -> Dict[str, Any]:
        """Преобразовать в словарь для удобства"""
        return asdict(self)


@dataclass
class SvodkaDannyh:
    """Общая сводка по всем данным"""
    kolichestvo_strok: int
    kolichestvo_kolonok: int
    kolonki: List[SvodkaKolonki]

    def v_slovar(self) -> Dict[str, Any]:
        """Преобразовать в словарь"""
        return {
            "kolichestvo_strok": self.kolichestvo_strok,
            "kolichestvo_kolonok": self.kolichestvo_kolonok,
            "kolonki": [k.v_slovar() for k in self.kolonki],
        }


def sozdat_svodku(
    dannye: pd.DataFrame,
    kolichestvo_primerov_na_kolonku: int = 3,
) -> SvodkaDannyh:
    """
    Создать полную сводку по набору данных
    """
    kolichestvo_strok, kolichestvo_kolonok = dannye.shape
    spisok_kolonok: List[SvodkaKolonki] = []

    for nazvanie_kolonki in dannye.columns:
        kolonka = dannye[nazvanie_kolonki]
        strokovoe_predstavlenie_tipa = str(kolonka.dtype)

        zapolnennyh = int(kolonka.notna().sum())
        propuskov = kolichestvo_strok - zapolnennyh
        dolya_propuskov = float(propuskov / kolichestvo_strok) if kolichestvo_strok > 0 else 0.0
        unikalnyh = int(kolonka.nunique(dropna=True))

        # Примерные значения
        primery = (
            kolonka.dropna().astype(str).unique()[:kolichestvo_primerov_na_kolonku].tolist()
            if zapolnennyh > 0
            else []
        )

        chislovaya = bool(ptypes.is_numeric_dtype(kolonka))
        minimum: Optional[float] = None
        maximum: Optional[float] = None
        srednee: Optional[float] = None
        standartnoe_otklonenie: Optional[float] = None

        if chislovaya and zapolnennyh > 0:
            minimum = float(kolonka.min())
            maximum = float(kolonka.max())
            srednee = float(kolonka.mean())
            standartnoe_otklonenie = float(kolonka.std())

        spisok_kolonok.append(
            SvodkaKolonki(
                nazvanie=nazvanie_kolonki,
                tip_dannyh=strokovoe_predstavlenie_tipa,
                zapolnennyh=zapolnennyh,
                propuskov=propuskov,
                dolya_propuskov=dolya_propuskov,
                unikalnyh=unikalnyh,
                primer_znacheniy=primery,
                chislovaya=chislovaya,
                minimum=minimum,
                maximum=maximum,
                srednee=srednee,
                standartnoe_otklonenie=standartnoe_otklonenie,
            )
        )

    return SvodkaDannyh(
        kolichestvo_strok=kolichestvo_strok,
        kolichestvo_kolonok=kolichestvo_kolonok,
        kolonki=spisok_kolonok,
    )


def tablitsa_propuskov(dannye: pd.DataFrame) -> pd.DataFrame:
    """
    Таблица пропусков по колонкам: количество и доля
    """
    if dannye.empty:
        return pd.DataFrame(columns=["kolichestvo_propuskov", "dolya_propuskov"])

    vsego_propuskov = dannye.isna().sum()
    dolya = vsego_propuskov / len(dannye)
    rezultat = (
        pd.DataFrame(
            {
                "kolichestvo_propuskov": vsego_propuskov,
                "dolya_propuskov": dolya,
            }
        )
        .sort_values("dolya_propuskov", ascending=False)
    )
    return rezultat


def matritsa_korrelyatsii(dannye: pd.DataFrame) -> pd.DataFrame:
    """
    Матрица корреляции Пирсона для числовых колонок
    """
    chislovye_kolonki = dannye.select_dtypes(include="number")
    if chislovye_kolonki.empty:
        return pd.DataFrame()
    return chislovye_kolonki.corr(numeric_only=True)


def top_kategorii(
    dannye: pd.DataFrame,
    maksimalno_kolonok: int = 5,
    kolichestvo_v_top: int = 5,
) -> Dict[str, pd.DataFrame]:
    """
    Для категориальных колонок найти топ значений
    """
    rezultat: Dict[str, pd.DataFrame] = {}
    potentsialnye_kolonki: List[str] = []

    for nazvanie in dannye.columns:
        kolonka = dannye[nazvanie]
        if ptypes.is_object_dtype(kolonka) or isinstance(kolonka.dtype, pd.CategoricalDtype):
            potentsialnye_kolonki.append(nazvanie)

    for nazvanie in potentsialnye_kolonki[:maksimalno_kolonok]:
        kolonka = dannye[nazvanie]
        chastota_znacheniy = kolonka.value_counts(dropna=True).head(kolichestvo_v_top)
        if chastota_znacheniy.empty:
            continue
        dolya = chastota_znacheniy / chastota_znacheniy.sum()
        tablitsa = pd.DataFrame(
            {
                "znachenie": chastota_znacheniy.index.astype(str),
                "kolichestvo": chastota_znacheniy.values,
                "dolya": dolya.values,
            }
        )
        rezultat[nazvanie] = tablitsa

    return rezultat


def proverit_kachestvo(svodka: SvodkaDannyh, tablitsa_propuskov: pd.DataFrame) -> Dict[str, Any]:
    """
    Проверка качества данных по эвристикам
    """
    flagi: Dict[str, Any] = {}
    flagi["slishkom_malo_strok"] = svodka.kolichestvo_strok < 100
    flagi["slishkom_mnogo_kolonok"] = svodka.kolichestvo_kolonok > 100
    
    # Проверка на много нулей в числовых колонках
    porog_nulej = 0.1
    mnogo_nulej = False
    for kolonka in svodka.kolonki:
        if kolonka.chislovaya and kolonka.zapolnennyh > 0:
            # Простая эвристика - если среднее близко к нулю и мин=макс=0
            if kolonka.minimum == 0 and kolonka.maximum == 0:
                mnogo_nulej = True
    flagi["mnogo_nulej"] = mnogo_nulej
    
    # Проверка на константные колонки
    est_konstantnye_kolonki = any(kol.unikalnyh == 1 for kol in svodka.kolonki)
    flagi["est_konstantnye_kolonki"] = est_konstantnye_kolonki
    
    # Проверка на дубликаты ID
    podozritelnye_id = next((kol for kol in svodka.kolonki if 'id' in kol.nazvanie.lower()), None)
    if podozritelnye_id is not None:
        flagi["est_podozritelnye_id"] = podozritelnye_id.unikalnyh < svodka.kolichestvo_strok
    else:
        flagi["est_podozritelnye_id"] = False
    
    maksimalnaya_dolya_propuskov = float(tablitsa_propuskov["dolya_propuskov"].max()) if not tablitsa_propuskov.empty else 0.0
    flagi["maksimalnaya_dolya_propuskov"] = maksimalnaya_dolya_propuskov
    flagi["slishkom_mnogo_propuskov"] = maksimalnaya_dolya_propuskov > 0.5
    
    # Простой расчет оценки качества
    ball_kachestva = 1.0
    ball_kachestva -= maksimalnaya_dolya_propuskov  # чем больше пропусков, тем хуже
    if svodka.kolichestvo_strok < 100:
        ball_kachestva -= 0.2
    if svodka.kolichestvo_kolonok > 100:
        ball_kachestva -= 0.1
    if mnogo_nulej:
        ball_kachestva -= 0.1
    if est_konstantnye_kolonki:
        ball_kachestva -= 0.1
    
    ball_kachestva = max(0.0, min(1.0, ball_kachestva))
    flagi["ball_kachestva"] = ball_kachestva

    return flagi


def podgotovit_dlya_pechati(svodka: SvodkaDannyh) -> pd.DataFrame:
    """
    Преобразовать сводку в таблицу для печати
    """
    stroki: List[Dict[str, Any]] = []
    for kolonka in svodka.kolonki:
        stroki.append(
            {
                "nazvanie": kolonka.nazvanie,
                "tip": kolonka.tip_dannyh,
                "zapolnennyh": kolonka.zapolnennyh,
                "propuskov": kolonka.propuskov,
                "dolya_propuskov": kolonka.dolya_propuskov,
                "unikalnyh": kolonka.unikalnyh,
                "chislovaya": kolonka.chislovaya,
                "minimum": kolonka.minimum,
                "maximum": kolonka.maximum,
                "srednee": kolonka.srednee,
                "standartnoe_otklonenie": kolonka.standartnoe_otklonenie,
            }
        )
    return pd.DataFrame(stroki)
