from __future__ import annotations

import pandas as pd

from eda_cli.core import (
    proverit_kachestvo,
    matritsa_korrelyatsii,
    podgotovit_dlya_pechati,
    tablitsa_propuskov,
    sozdat_svodku,
    top_kategorii,
)


def _primer_dannyh() -> pd.DataFrame:
    """Пример данных для тестирования"""
    return pd.DataFrame(
        {
            "vozrast": [10, 20, 30, None],
            "rost": [140, 150, 160, 170],
            "gorod": ["A", "B", "A", None],
        }
    )


def _dannye_s_konstantami() -> pd.DataFrame:
    """Данные с константными колонками"""
    return pd.DataFrame({
        "Imya": ["Anna", "Ivan", "Olga"],
        "Vozrast": [25, 30, 35],
        "Konstantnaya": [1, 1, 1],
    })


def test_konstantnye_kolonki():
    """Тест для проверки константных колонок"""
    dannye = _dannye_s_konstantami()
    tablitsa_propuskov_dannyh = tablitsa_propuskov(dannye)
    svodka = sozdat_svodku(dannye)
    flagi_kachestva = proverit_kachestvo(svodka, tablitsa_propuskov_dannyh)
    
    assert flagi_kachestva["est_konstantnye_kolonki"] is True


def test_sozdanie_svodki():
    """Тест создания сводки данных"""
    dannye = _primer_dannyh()
    svodka = sozdat_svodku(dannye)

    assert svodka.kolichestvo_strok == 4
    assert svodka.kolichestvo_kolonok == 3
    assert any(k.nazvanie == "vozrast" for k in svodka.kolonki)
    assert any(k.nazvanie == "gorod" for k in svodka.kolonki)

    tablitsa_svodki = podgotovit_dlya_pechati(svodka)
    assert "nazvanie" in tablitsa_svodki.columns
    assert "dolya_propuskov" in tablitsa_svodki.columns


def test_tablitsa_propuskov_i_kachestvo():
    """Тест таблицы пропусков и качества данных"""
    dannye = _primer_dannyh()
    tablitsa_propuskov_dannyh = tablitsa_propuskov(dannye)

    assert "kolichestvo_propuskov" in tablitsa_propuskov_dannyh.columns
    assert tablitsa_propuskov_dannyh.loc["vozrast", "kolichestvo_propuskov"] == 1

    svodka = sozdat_svodku(dannye)
    flagi_kachestva = proverit_kachestvo(svodka, tablitsa_propuskov_dannyh)
    assert 0.0 <= flagi_kachestva["ball_kachestva"] <= 1.0


def test_korrelyatsii_i_top_kategorii():
    """Тест корреляций и топ категорий"""
    dannye = _primer_dannyh()
    korrelyatsii = matritsa_korrelyatsii(dannye)
    
    # Проверяем, что есть корреляции или матрица пуста
    assert "vozrast" in korrelyatsii.columns or korrelyatsii.empty is False

    top_kategorii_dannyh = top_kategorii(dannye, maksimalno_kolonok=5, kolichestvo_v_top=2)
    assert "gorod" in top_kategorii_dannyh
    tablitsa_gorodov = top_kategorii_dannyh["gorod"]
    assert "znachenie" in tablitsa_gorodov.columns
    assert len(tablitsa_gorodov) <= 2
