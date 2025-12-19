from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .core import (
    SvodkaDannyh,
    proverit_kachestvo,
    matritsa_korrelyatsii,
    podgotovit_dlya_pechati,
    tablitsa_propuskov,
    sozdat_svodku,
    top_kategorii,
)
from .viz import (
    narisovat_teplovuyu_kartu_korrelyatsii,
    narisovat_matritsu_propuskov,
    narisovat_gistogrammy_po_kolonkam,
    sohranit_tablitsy_top_kategorij,
)

prilozhenie = typer.Typer(help="Мини-CLI для анализа CSV файлов")


def _zagruzit_csv(
    put: Path,
    razdelyatel: str = ",",
    kodirovka: str = "utf-8",
) -> pd.DataFrame:
    """Загрузить CSV файл"""
    if not put.exists():
        raise typer.BadParameter(f"Файл '{put}' не найден")
    try:
        return pd.read_csv(put, sep=razdelyatel, encoding=kodirovka)
    except Exception as oshibka:
        raise typer.BadParameter(f"Не удалось прочитать CSV: {oshibka}") from oshibka


@prilozhenie.command()
def obzor(
    put: str = typer.Argument(..., help="Путь к CSV файлу."),
    razdelyatel: str = typer.Option(",", help="Разделитель в CSV."),
    kodirovka: str = typer.Option("utf-8", help="Кодировка файла."),
) -> None:
    """
    Краткий обзор набора данных
    """
    dannye = _zagruzit_csv(Path(put), sep=razdelyatel, encoding=kodirovka)
    svodka: SvodkaDannyh = sozdat_svodku(dannye)
    tablitsa_svodki = podgotovit_dlya_pechati(svodka)

    typer.echo(f"Количество строк: {svodka.kolichestvo_strok}")
    typer.echo(f"Количество колонок: {svodka.kolichestvo_kolonok}")
    typer.echo("\nКолонки:")
    typer.echo(tablitsa_svodki.to_string(index=False))


@prilozhenie.command()
def nachalo(
    put: str = typer.Argument(..., help="Путь к CSV файлу."),
    kolichestvo_strok: int = typer.Option(5, help="Количество выводимых строк"),
    razdelyatel: str = typer.Option(",", help="Разделитель в CSV."),
    kodirovka: str = typer.Option("utf-8", help="Кодировка файла"),
) -> None:
    """
    Показать начало файла
    """
    dannye = _zagruzit_csv(Path(put), sep=razdelyatel, encoding=kodirovka)
    svodka: SvodkaDannyh = sozdat_svodku(dannye)
    tablitsa_svodki = podgotovit_dlya_pechati(svodka)
    dlya_pechati = tablitsa_svodki.head(kolichestvo_strok)
    
    typer.echo(f"Первые {kolichestvo_strok} строк:\n")
    typer.echo(dlya_pechati.to_string(index=False))


@prilozhenie.command()
def otchyot(
    put: str = typer.Argument(..., help="Путь к CSV файлу."),
    papka_dlya_otchyota: str = typer.Option("otchyoty", help="Папка для сохранения отчёта."),
    razdelyatel: str = typer.Option(",", help="Разделитель в CSV."),
    kodirovka: str = typer.Option("utf-8", help="Кодировка файла."),
    maks_gistogramm: int = typer.Option(6, help="Максимум гистограмм для числовых колонок."),
    zagolovok: str = typer.Option("Отчёт анализа данных", help="Заголовок отчета"),
    minimalnaya_dolya_propuskov: float = typer.Option(0.1, help="Порог доли пропусков", min=0.0, max=1.0),
) -> None:
    """
    Сгенерировать полный отчёт анализа данных
    """
    osnovnaya_papka = Path(papka_dlya_otchyota)
    osnovnaya_papka.mkdir(parents=True, exist_ok=True)

    dannye = _zagruzit_csv(Path(put), sep=razdelyatel, encoding=kodirovka)
    
    # 1. Создание сводок
    svodka = sozdat_svodku(dannye)
    tablitsa_svodki = podgotovit_dlya_pechati(svodka)
    tablitsa_propuskov_dannyh = tablitsa_propuskov(dannye)
    matritsa_korr = matritsa_korrelyatsii(dannye)
    top_kategorii_dannyh = top_kategorii(dannye)

    # 2. Проверка качества
    flagi_kachestva = proverit_kachestvo(svodka, tablitsa_propuskov_dannyh)

    # 3. Сохранение таблиц
    tablitsa_svodki.to_csv(osnovnaya_papka / "svodka.csv", index=False)
    if not tablitsa_propuskov_dannyh.empty:
        tablitsa_propuskov_dannyh.to_csv(osnovnaya_papka / "propuski.csv", index=True)
    if not matritsa_korr.empty:
        matritsa_korr.to_csv(osnovnaya_papka / "korrelyatsii.csv", index=True)
    sohranit_tablitsy_top_kategorij(top_kategorii_dannyh, osnovnaya_papka / "top_kategorii")

    # 4. Создание markdown отчёта
    put_k_markdown = osnovnaya_papka / "otchyot.md"
    with put_k_markdown.open("w", encoding="utf-8") as fajl:
        fajl.write(f"# {zagolovok}\n\n")
        fajl.write(f"Исходный файл: `{Path(put).name}`\n\n")
        fajl.write(f"Строк: **{svodka.kolichestvo_strok}**, колонок: **{svodka.kolichestvo_kolonok}**\n\n")

        fajl.write("## Качество данных (эвристики)\n\n")
        fajl.write(f"- Оценка качества: **{flagi_kachestva['ball_kachestva']:.2f}**\n")
        fajl.write(f"- Максимальная доля пропусков: **{flagi_kachestva['maksimalnaya_dolya_propuskov']:.2%}**\n")
        fajl.write(f"- Слишком мало строк: **{flagi_kachestva['slishkom_malo_strok']}**\n")
        fajl.write(f"- Слишком много колонок: **{flagi_kachestva['slishkom_mnogo_kolonok']}**\n")
        fajl.write(f"- Слишком много пропусков: **{flagi_kachestva['slishkom_mnogo_propuskov']}**\n")
        fajl.write(f"- Есть подозрительные ID: **{flagi_kachestva['est_podozritelnye_id']}**\n")
        fajl.write(f"- Много нулевых значений: **{flagi_kachestva['mnogo_nulej']}**\n")
        fajl.write(f"- Есть константные колонки: **{flagi_kachestva['est_konstantnye_kolonki']}**\n\n")
        
        fajl.write("## Колонки\n\n")
        fajl.write("Подробности в файле `svodka.csv`.\n\n")

        fajl.write("## Пропуски\n\n")
        if tablitsa_propuskov_dannyh.empty:
            fajl.write("Пропусков нет или данные пусты.\n\n")
        else:
            fajl.write("Смотрите файлы `propuski.csv` и `matritsa_propuskov.png`.\n\n")

        # Колонки с пропусками выше порога
        kolonki_s_propuskami = tablitsa_propuskov_dannyh[tablitsa_propuskov_dannyh["dolya_propuskov"] > minimalnaya_dolya_propuskov]
        if not kolonki_s_propuskami.empty:
            fajl.write("## Колонки с большим количеством пропусков\n\n")
            fajl.write(f"Доля пропусков > {minimalnaya_dolya_propuskov:.1%}\n\n")
            for kolonka, stroka in kolonki_s_propuskami.iterrows():
                fajl.write(f"- **{kolonka}**: {stroka['dolya_propuskov']:.2%}\n")
            fajl.write("\n")

        fajl.write("## Корреляции числовых признаков\n\n")
        if matritsa_korr.empty:
            fajl.write("Недостаточно числовых колонок для анализа корреляций.\n\n")
        else:
            fajl.write("Смотрите `korrelyatsii.csv` и `teplovaya_karta_korrelyatsij.png`.\n\n")

        fajl.write("## Категориальные признаки\n\n")
        if not top_kategorii_dannyh:
            fajl.write("Категориальные признаки не найдены.\n\n")
        else:
            fajl.write("Топ значения в папке `top_kategorii/`.\n\n")

        fajl.write("## Гистограммы числовых колонок\n\n")
        fajl.write("Смотрите файлы `gistogramma_*.png`.\n")

    # 5. Создание графиков
    narisovat_gistogrammy_po_kolonkam(dannye, osnovnaya_papka, maksimalno_kolonok=maks_gistogramm)
    narisovat_matritsu_propuskov(dannye, osnovnaya_papka / "matritsa_propuskov.png")
    narisovat_teplovuyu_kartu_korrelyatsii(dannye, osnovnaya_papka / "teplovaya_karta_korrelyatsij.png")

    typer.echo(f"Отчёт сохранён в: {osnovnaya_papka}")
    typer.echo(f"- Markdown отчёт: {put_k_markdown}")
    typer.echo("- Табличные файлы: svodka.csv, propuski.csv, korrelyatsii.csv, top_kategorii/*.csv")
    typer.echo("- Графики: gistogramma_*.png, matritsa_propuskov.png, teplovaya_karta_korrelyatsij.png")


if __name__ == "__main__":
    prilozhenie()
