from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

Puti = Union[str, Path]


def _sozdat_papku(put: Puti) -> Path:
    """Создать папку если её нет"""
    papka = Path(put)
    papka.mkdir(parents=True, exist_ok=True)
    return papka


def narisovat_gistogrammy_po_kolonkam(
    dannye: pd.DataFrame,
    papka_dlya_sohraneniya: Puti,
    maksimalno_kolonok: int = 6,
    kolichestvo_binov: int = 20,
) -> List[Path]:
    """
    Построить гистограммы для числовых колонок
    """
    papka_dlya_sohraneniya = _sozdat_papku(papka_dlya_sohraneniya)
    chislovye_dannye = dannye.select_dtypes(include="number")

    spisok_putey: List[Path] = []
    for nomer, nazvanie in enumerate(chislovye_dannye.columns[:maksimalno_kolonok]):
        znacheniya = chislovye_dannye[nazvanie].dropna()
        if znacheniya.empty:
            continue

        figura, osi = plt.subplots()
        osi.hist(znacheniya.values, bins=kolichestvo_binov)
        osi.set_title(f"Гистограмма для {nazvanie}")
        osi.set_xlabel(nazvanie)
        osi.set_ylabel("Количество")
        figura.tight_layout()

        put_k_fajlu = papka_dlya_sohraneniya / f"gistogramma_{nomer+1}_{nazvanie}.png"
        figura.savefig(put_k_fajlu)
        plt.close(figura)

        spisok_putey.append(put_k_fajlu)

    return spisok_putey


def narisovat_matritsu_propuskov(dannye: pd.DataFrame, put_k_fajlu: Puti) -> Path:
    """
    Визуализация пропусков в данных
    """
    put_k_fajlu = Path(put_k_fajlu)
    put_k_fajlu.parent.mkdir(parents=True, exist_ok=True)

    if dannye.empty:
        # Пустой график для пустых данных
        figura, osi = plt.subplots()
        osi.text(0.5, 0.5, "Нет данных", ha="center", va="center")
        osi.axis("off")
    else:
        maska_propuskov = dannye.isna().values
        figura, osi = plt.subplots(figsize=(min(12, dannye.shape[1] * 0.4), 4))
        osi.imshow(maska_propuskov, aspect="auto", interpolation="none")
        osi.set_xlabel("Колонки")
        osi.set_ylabel("Строки")
        osi.set_title("Матрица пропусков")
        osi.set_xticks(range(dannye.shape[1]))
        osi.set_xticklabels(dannye.columns, rotation=90, fontsize=8)
        osi.set_yticks([])

    figura.tight_layout()
    figura.savefig(put_k_fajlu)
    plt.close(figura)
    return put_k_fajlu


def narisovat_teplovuyu_kartu_korrelyatsii(dannye: pd.DataFrame, put_k_fajlu: Puti) -> Path:
    """
    Тепловая карта корреляций между числовыми признаками
    """
    put_k_fajlu = Path(put_k_fajlu)
    put_k_fajlu.parent.mkdir(parents=True, exist_ok=True)

    chislovye_dannye = dannye.select_dtypes(include="number")
    if chislovye_dannye.shape[1] < 2:
        figura, osi = plt.subplots()
        osi.text(0.5, 0.5, "Недостаточно числовых колонок", ha="center", va="center")
        osi.axis("off")
    else:
        korrelyatsii = chislovye_dannye.corr(numeric_only=True)
        figura, osi = plt.subplots(figsize=(min(10, korrelyatsii.shape[1]), min(8, korrelyatsii.shape[0])))
        izobrazhenie = osi.imshow(korrelyatsii.values, vmin=-1, vmax=1, cmap="coolwarm", aspect="auto")
        osi.set_xticks(range(korrelyatsii.shape[1]))
        osi.set_xticklabels(korrelyatsii.columns, rotation=90, fontsize=8)
        osi.set_yticks(range(korrelyatsii.shape[0]))
        osi.set_yticklabels(korrelyatsii.index, fontsize=8)
        osi.set_title("Тепловая карта корреляций")
        figura.colorbar(izobrazhenie, ax=osi, label="Коэффициент корреляции")

    figura.tight_layout()
    figura.savefig(put_k_fajlu)
    plt.close(figura)
    return put_k_fajlu


def sohranit_tablitsy_top_kategorij(
    top_kategorii: Dict[str, pd.DataFrame],
    papka_dlya_sohraneniya: Puti,
) -> List[Path]:
    """
    Сохранить таблицы с топ категориями в CSV файлы
    """
    papka_dlya_sohraneniya = _sozdat_papku(papka_dlya_sohraneniya)
    spisok_putey: List[Path] = []
    
    for nazvanie_kolonki, tablitsa in top_kategorii.items():
        put_k_fajlu = papka_dlya_sohraneniya / f"top_znachenij_{nazvanie_kolonki}.csv"
        tablitsa.to_csv(put_k_fajlu, index=False)
        spisok_putey.append(put_k_fajlu)
        
    return spisok_putey
