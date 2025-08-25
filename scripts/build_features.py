#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# >>> Lê DIRETO dos CSVs que você gerou:
IN_DIR  = ROOT / "data" / "limit_test_nacional"
OUT_DIR = ROOT / "data" / "build"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeia nomes de arquivo -> nome canônico da coluna no dataset final
FILE_TO_CANON = {
    "pib_nominal_brl":        "pib_nominal_brl",
    "receita_total_pct_pib":  "receita_pct_pib",
    "despesa_total_pct_pib":  "despesa_pct_pib",
    "divida_bruta_pct_pib":   "divida_pct_pib",
    "ipca_indice":            "ipca_indice",
    "selic_meta":             "selic_meta",
    "cambio_usd":             "cambio_usd",
    "producao_industrial":    "producao_industrial",
}

# Ordem de merge: começamos pelo PIB e vamos agregando
MERGE_ORDER = [
    "pib_nominal_brl",
    "receita_total_pct_pib",
    "despesa_total_pct_pib",
    "divida_bruta_pct_pib",
    "ipca_indice",
    "selic_meta",
    "cambio_usd",
    "producao_industrial",
]

def _infer_date_col(df: pd.DataFrame) -> str | None:
    cands = ["date","data","periodo","time","ano","year","mes","month"]
    low = {c.lower(): c for c in df.columns}
    for k in cands:
        if k in low:
            return low[k]
    return None

def ensure_date(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normaliza coluna temporal para 'date' em datetime64[ns] NAIVE (sem timezone).
    Resolve misturas de datas com e sem timezone (ex.: IPEA vem com '-02:00').
    """
    df = df.copy()
    dcol = _infer_date_col(df)
    if dcol is None:
        raise RuntimeError(f"não achei coluna de data. colunas: {list(df.columns)}")

    # 1) parse em UTC para lidar com offsets mistos sem warning/erro
    s = pd.to_datetime(df[dcol], errors="coerce", dayfirst=True, utc=True)

    # 2) remover timezone (ficar NAIVE) – padroniza com fontes sem tz
    s = s.dt.tz_convert("UTC").dt.tz_localize(None)

    df["date"] = s
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    return df

def load_feature_csv(file_stem: str) -> pd.DataFrame | None:
    path = IN_DIR / f"{file_stem}.csv"
    if not path.exists() or path.stat().st_size == 0:
        print(f"[warn] {path} ausente ou vazio – pulando")
        return None

    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"[warn] falha lendo {path}: {e} – pulando")
        return None

    # Normaliza data
    try:
        df = ensure_date(df)
    except Exception as e:
        print(f"[warn] não consegui normalizar 'date' em {path}: {e} – pulando")
        return None

    # Descobre coluna de valores e renomeia p/ canônico
    canon = FILE_TO_CANON[file_stem]
    if canon in df.columns:
        val_col = canon
    else:
        # pega a primeira coluna numérica (não-'date')
        numcols = [c for c in df.columns if c != "date" and pd.api.types.is_numeric_dtype(df[c])]
        if numcols:
            val_col = numcols[0]
            df = df.rename(columns={val_col: canon})
        else:
            # tenta converter segunda coluna como numérica
            othercols = [c for c in df.columns if c != "date"]
            if othercols:
                df[othercols[0]] = pd.to_numeric(df[othercols[0]], errors="coerce")
                df = df.rename(columns={othercols[0]: canon})
                val_col = canon
            else:
                print(f"[warn] sem coluna numérica em {path} – pulando")
                return None

    # Mantém só date + valor canônico
    out = df[["date", canon]].copy()
    return out

def nearest_merge(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    """
    merge_asof, mantendo a 'date' do lado esquerdo; right é casado pelo mais próximo.
    Pré-condição: ambos com 'date' NAIVE e ordenados.
    """
    L = ensure_date(left)
    R = ensure_date(right)

    L = L.sort_values("date")
    R = R.sort_values("date")

    out = pd.merge_asof(L, R, on="date", direction="nearest")
    return out

def main():
    # 1) carrega datasets existentes (ordem definida)
    frames: list[pd.DataFrame] = []
    for stem in MERGE_ORDER:
        df = load_feature_csv(stem)
        if df is None:
            continue
        frames.append(df)

    if not frames:
        raise RuntimeError("Nenhum dataset válido encontrado em data/limit_test_nacional/.")

    # 2) merge incremental
    base = frames[0]
    for df in frames[1:]:
        base = nearest_merge(base, df)

    # 3) salvar
    out_path = OUT_DIR / "features_nacional.csv"
    base.to_csv(out_path, index=False)
    print(f"[ok] features → {out_path}  ({len(base)} linhas, {len(base.columns)} colunas)")
    print("[cols]", ", ".join(base.columns))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[build_features error] {e}")
        sys.exit(1)
