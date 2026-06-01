"""Сводит скан множителей в таблицу Δ(MULT) по доменам и подсвечивает лучшую точку."""
import glob
import json
import os
import pandas as pd


def main():
    files = sorted(glob.glob("results/scan/*_scan.jsonl"))
    if not files:
        print("Нет файлов скана в results/scan/")
        return
    rows = {}
    mults = set()
    for fp in files:
        subj = os.path.basename(fp).replace("_scan.jsonl", "")
        rows[subj] = {}
        with open(fp) as f:
            for line in f:
                r = json.loads(line)
                rows[subj][r["multiplier"]] = r["delta"]
                mults.add(r["multiplier"])
    mults = sorted(mults)

    print("\n" + "=" * 80)
    print("MULTIPLIER SCAN — Δ accuracy (vs baseline) по множителям")
    print("Ищем множитель с Δ >= 0 (или наименее отрицательной).")
    print("=" * 80)
    table = []
    for subj, dd in rows.items():
        row = {"Subject": subj}
        best_m, best_d = None, None
        for m in mults:
            d = dd.get(m)
            row[f"x{m}"] = d
            if d is not None and (best_d is None or d > best_d):
                best_d, best_m = d, m
        row["BEST mult"] = best_m
        row["BEST Δ"] = best_d
        table.append(row)
    df = pd.DataFrame(table)
    try:
        print(df.to_markdown(index=False))
    except Exception:
        print(df.to_string(index=False))
    print("=" * 80)
    print("Если BEST Δ >= 0 на каком-то домене при малом множителе — это положительный сигнал.")
    print("Если Δ монотонно падает с ростом множителя — подтверждает: большой bias ломает routing.\n")


if __name__ == "__main__":
    main()
