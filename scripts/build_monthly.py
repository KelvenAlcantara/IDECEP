#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable, List, Tuple, Set

import numpy as np
import pandas as pd


# ==============================
# CLI
# ==============================

def parse_args():
    p = argparse.ArgumentParser(
        description="Mensaliza (anual→mensal) a base features_nacional.csv usando Denton–Cholette (aditivo) + correção de não-negatividade."
    )
    p.add_argument("--in", dest="inp", required=True,
                   help="CSV anual de entrada (ex.: data/build/features_nacional.csv). Deve ter coluna 'date'.")
    p.add_argument("--out", dest="out", required=True,
                   help="CSV mensal de saída (ex.: data/build/features_monthly.csv).")
    p.add_argument("--start", default=None,
                   help="(opcional) recorte inicial YYYY-MM-DD (aplicado após a mensalização).")
    p.add_argument("--end", default=None,
                   help="(opcional) recorte final YYYY-MM-DD (aplicado após a mensalização).")
    p.add_argument("--flows", nargs="*", default=["pib_nominal_brl"],
                   help="Colunas tratadas como FLOW (soma dos 12 meses = anual). Default: pib_nominal_brl")
    p.add_argument("--avgs", nargs="*", default=[],
                   help="Colunas tratadas como AVG/STOCK (média dos 12 meses = anual). As demais entram aqui por padrão.")
    p.add_argument("--verbose", action="store_true",
                   help="Imprime logs de progresso.")
    return p.parse_args()


def log(msg: str, verbose: bool):
    if verbose:
        print(msg)


# ==============================
# Utilitários temporais
# ==============================

def ensure_datetime_col(df: pd.DataFrame, col="date") -> pd.DataFrame:
    df = df.copy()
    df[col] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
    if df[col].isna().any():
        raise ValueError("Coluna 'date' inválida ou com valores não parseáveis.")
    return df.sort_values(col).reset_index(drop=True)


def month_end_index(ymin: int, ymax: int) -> pd.DatetimeIndex:
    """Índice no FIM do mês (MonthEnd)."""
    return pd.period_range(start=f"{ymin}-01", end=f"{ymax}-12", freq="M").to_timestamp(how="end")


# ==============================
# Denton–Cholette (aditivo)
# ==============================

def denton_additive_annual_to_monthly(y_annual: np.ndarray, constraint: str) -> np.ndarray:
    """
    Annual -> Monthly com Denton-Cholette (1ª diferença).
    constraint:
      - 'flow' : soma dos 12 meses = anual
      - 'avg'  : média dos 12 meses = anual
    Retorna vetor mensal (12*N).
    """
    y = np.asarray(y_annual, dtype=float)
    N = len(y)
    T = 12 * N

    # Matriz de agregação A (N x T)
    A = np.zeros((N, T))
    for i in range(N):
        A[i, 12*i:12*i+12] = 1.0 if constraint == "flow" else (1.0 / 12.0)

    # Operador de primeira diferença D (T-1 x T)
    D = np.zeros((T - 1, T))
    for t in range(T - 1):
        D[t, t] = -1.0
        D[t, t + 1] = 1.0

    Q = D.T @ D  # penaliza variação mensal
    K = np.block([[Q, A.T],
                  [A, np.zeros((N, N))]])
    rhs = np.concatenate([np.zeros(T), y])

    sol = np.linalg.lstsq(K, rhs, rcond=None)[0]
    x = sol[:T]
    return x


# ==============================
# Correção de não-negatividade por ano
# ==============================

# Por padrão, todas as 8 colunas do seu dataset são não-negativas
NON_NEG_DEFAULT = {
    "pib_nominal_brl",
    "receita_pct_pib",
    "despesa_pct_pib",
    "divida_pct_pib",
    "ipca_indice",
    "selic_meta",
    "cambio_usd",
    "producao_industrial",
}

def fix_non_negative_yearwise_monthly(m_values: np.ndarray,
                                      y_annual: np.ndarray,
                                      years: np.ndarray,
                                      constraint: str,
                                      eps: float = 1e-9) -> np.ndarray:
    """
    Se aparecer negativo no vetor mensal m_values, corrige ano a ano:
      1) clip para >= eps
      2) reescala dentro do ano para respeitar a restrição anual:
         - flow: sum(m_i*) == y_i
         - avg : mean(m_i*) == y_i
    Mantém mínimos ajustes necessários.
    """
    m = m_values.copy()
    if not (m < 0).any():
        return m

    T = len(m)
    N = len(y_annual)
    assert T == 12 * N, "Dimensão inconsistente em fix_non_negative_yearwise_monthly."

    for i in range(N):
        sl = slice(12*i, 12*i+12)
        mi = m[sl].astype(float)
        yi = float(y_annual[i])

        # 1) clip
        mi = np.where(mi < eps, eps, mi)

        # 2) reescala
        if constraint == "flow":
            s = mi.sum()
            if s > 0:
                mi = mi * (yi / s)
            else:
                mi = np.full(12, yi / 12.0)
        else:  # avg
            av = mi.mean()
            if av > 0:
                mi = mi * (yi / av)
            else:
                mi = np.full(12, yi)

        m[sl] = mi

    return m


# ==============================
# Mensalização
# ==============================

DEFAULT_COLUMNS_ORDER = [
    "pib_nominal_brl",
    "receita_pct_pib",
    "despesa_pct_pib",
    "divida_pct_pib",
    "ipca_indice",
    "selic_meta",
    "cambio_usd",
    "producao_industrial",
]

def split_flows_avgs(all_cols: Iterable[str], flows_cli: List[str], avgs_cli: List[str]) -> Tuple[Set[str], Set[str]]:
    flows = set()
    avgs  = set()
    # 1) aplica escolhas do usuário
    flows.update(flows_cli or [])
    avgs.update(avgs_cli or [])
    # 2) qualquer outra coluna numérica que não entrou em flows → avgs
    for c in all_cols:
        if c in flows:
            continue
        if c not in avgs:
            avgs.add(c)
    return flows, avgs


def make_monthly_from_annual(df_annual: pd.DataFrame,
                             date_col: str = "date",
                             flows_cols: List[str] | None = None,
                             avgs_cols: List[str] | None = None,
                             non_negative_cols: Set[str] = NON_NEG_DEFAULT,
                             verbose: bool = False) -> pd.DataFrame:
    df = ensure_datetime_col(df_annual, date_col)
    df["year"] = df[date_col].dt.year

    # pega somente colunas numéricas (exceto 'date' e 'year')
    num_cols = [c for c in df.columns if c not in [date_col, "year"] and pd.api.types.is_numeric_dtype(df[c])]
    # ordena as colunas no output (se existirem), mantendo extras ao final
    ordered = [c for c in DEFAULT_COLUMNS_ORDER if c in num_cols] + [c for c in num_cols if c not in DEFAULT_COLUMNS_ORDER]

    flows_set, avgs_set = split_flows_avgs(ordered, flows_cols or [], avgs_cols or [])

    years = df["year"].to_numpy()
    m_index = month_end_index(int(years.min()), int(years.max()))
    out = pd.DataFrame({"date": m_index})

    # valores anuais alinhados
    y_by_col = {col: df[col].to_numpy() for col in ordered}

    for col in ordered:
        y = y_by_col[col]
        constraint = "flow" if col in flows_set else "avg"
        m = denton_additive_annual_to_monthly(y, constraint=constraint)

        # correção de não-negatividade (se necessário)
        if col in non_negative_cols and (m < 0).any():
            m = fix_non_negative_yearwise_monthly(m, y, years, constraint=constraint, eps=1e-9)
            if verbose:
                nneg = (m < 0).sum()
                log(f"[Fix+Rescale] {col}: negativos corrigidos; remanescentes={nneg}", verbose)

        out[col] = m
        log(f"[Denton] {col}: {constraint}", verbose)

    # garante ordenação das colunas no arquivo final
    out = out[["date"] + ordered]
    return out


# ==============================
# Main
# ==============================

def main():
    args = parse_args()
    inp = Path(args.inp)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df_annual = pd.read_csv(inp)
    if "date" not in df_annual.columns:
        raise SystemExit("CSV de entrada precisa ter a coluna 'date'.")

    mdf = make_monthly_from_annual(
        df_annual,
        date_col="date",
        flows_cols=args.flows,
        avgs_cols=args.avgs,
        verbose=args.verbose
    )

    # recorte opcional após mensalização
    if args.start or args.end:
        mdf["date"] = pd.to_datetime(mdf["date"], errors="coerce")
        if args.start:
            mdf = mdf[mdf["date"] >= pd.to_datetime(args.start)]
        if args.end:
            mdf = mdf[mdf["date"] <= pd.to_datetime(args.end)]
        mdf = mdf.reset_index(drop=True)

    # >>> força formato YYYY-MM-DD
    mdf["date"] = pd.to_datetime(mdf["date"]).dt.strftime("%Y-%m-%d")

    mdf.to_csv(out, index=False)
    print(f"OK: salvo mensal em {out}  ({len(mdf)} linhas, {len(mdf.columns)} colunas)")
    if args.verbose:
        print("[cols]", ", ".join(mdf.columns))


if __name__ == "__main__":
    main()