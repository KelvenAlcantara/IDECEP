#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalize_csv_idecep_latex.py — LaTeX-safe + diagnósticos completos

Gera artefatos prontos para LaTeX SEM tocar nas pastas de validação do projeto.
Saídas (fixas, sem timestamp) em: <RAIZ>/files_to_latex/idecep/

Arquivos gerados:
  - idecep_stationarity_checks.csv
  - idecep_rolling_logdiff.csv
  - idecep_ecm_diagnostics.csv   <-- agora com: ADF p, KPSS p, LB(12), LB(24), BG(12), ARCH LM p, JB p, DW
  - idecep_gate_report.csv
  (+) opcional: copia/saneia CSVs de idecep_uses

Leitura:
  - Usa outputs/idecep/idecep_processed.csv se existir;
  - Caso contrário, requer --raw.

Uso:
  python scripts/normalize_csv_idecep_latex.py
  python scripts/normalize_csv_idecep_latex.py --raw data/seu.csv --date-col date
  python scripts/normalize_csv_idecep_latex.py --extra-from outputs/idecep_uses
"""

from __future__ import annotations
import argparse
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant
from statsmodels.stats.diagnostic import (
    acorr_ljungbox,
    acorr_breusch_godfrey,
    het_arch,
    het_breuschpagan
)
from statsmodels.stats.stattools import jarque_bera, durbin_watson

# =======================
# Caminhos robustos
# =======================
ROOT = Path(__file__).resolve().parents[1]
OUT_BASE = ROOT / "outputs"
IDECEP_DIR = OUT_BASE / "idecep"
PROCESSED_CSV = IDECEP_DIR / "idecep_processed.csv"

LATEX_DIR = ROOT / "files_to_latex" / "idecep"
STATION_CSV = LATEX_DIR / "idecep_stationarity_checks.csv"
ROLLING_CSV = LATEX_DIR / "idecep_rolling_logdiff.csv"
DIAG_CSV    = LATEX_DIR / "idecep_ecm_diagnostics.csv"
GATE_CSV    = LATEX_DIR / "idecep_gate_report.csv"

def ensure_dirs():
    LATEX_DIR.mkdir(parents=True, exist_ok=True)

# ============ formatação/padronização ============
def _bool_to_int_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    if s.dtype == object:
        lower = s.astype(str).str.lower()
        mask_t = lower.isin(["true", "1", "yes", "sim"])
        mask_f = lower.isin(["false", "0", "no", "nao", "não"])
        if (mask_t | mask_f).any():
            out = pd.Series(np.nan, index=s.index, dtype="float64")
            out[mask_t] = 1
            out[mask_f] = 0
            try:
                rest = pd.to_numeric(s[~(mask_t | mask_f)], errors="coerce")
                out[~(mask_t | mask_f)] = rest
            except Exception:
                pass
            return out.fillna(0).astype(int)
    return s

def _latex_clean_df(df: pd.DataFrame, float_ndigits: int = 6) -> pd.DataFrame:
    df = df.copy()
    for c in df.columns:
        df[c] = _bool_to_int_series(df[c])
    df = df.replace([np.inf, -np.inf], np.nan)

    # arredonda floats "grandes", mas não mexe em strings já formatadas (p-values)
    for c in df.select_dtypes(include=["float64", "float32"]).columns:
        df[c] = df[c].round(float_ndigits)

    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).str.strip()

    df = df.where(pd.notna(df), "")
    return df

def _to_csv_latex(df: pd.DataFrame, path: Path, float_ndigits: int = 6, index: bool = False):
    df = _latex_clean_df(df, float_ndigits=float_ndigits)
    path.parent.mkdir(parents=True, exist_ok=True)
    # não forçar float_format aqui, pois algumas colunas já são strings formatadas (p-values)
    df.to_csv(path, index=index)

def _format_p(x: float) -> str:
    """
    Retorna p-value em formato legível a LaTeX sem virar 0.000000:
    - usa formato 'g' com até 6 sig. figs (ex: 6.373e-19).
    - se NaN, retorna vazio.
    """
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return ""
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)

def _copy_and_clean_all_csvs(src_dir: Path, dst_dir: Path, float_ndigits: int = 6):
    if not src_dir or not src_dir.exists():
        return
    for p in src_dir.glob("*.csv"):
        try:
            df = pd.read_csv(p)
            _to_csv_latex(df, dst_dir / p.name, float_ndigits=float_ndigits, index=False)
            print(f"[extras] Copiado/saneado: {p.name}")
        except Exception as e:
            print(f"[extras] Falhou {p.name}: {e}")

# ============ helpers de detecção ============
def _pick(df: pd.DataFrame, candidates) -> str | None:
    cols = {c.lower(): c for c in df.columns}
    for name in candidates:
        if name in cols:
            return cols[name]
    return None

def _reconstruct_from_diffs(df: pd.DataFrame, dln_y_col: str, dln_x_col: str) -> pd.DataFrame:
    tmp = df.copy()
    tmp["ln_y"] = tmp[dln_y_col].cumsum()
    tmp["ln_x"] = tmp[dln_x_col].cumsum()
    return tmp[["ln_y", "ln_x"]].dropna()

def _ensure_ln_columns(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()

    ln_y = _pick(tmp, ["ln_y", "log_y", "lny"])
    ln_x = _pick(tmp, ["ln_x", "log_x", "lnx"])
    if ln_y and ln_x:
        out = tmp.rename(columns={ln_y: "ln_y", ln_x: "ln_x"})
        return out[["ln_y", "ln_x"]].replace([np.inf, -np.inf], np.nan).dropna()

    y_pc = _pick(tmp, ["pib_pc_real", "pib_pc", "y_level", "pib_real_pc", "pib_real_brl_pc"])
    x_pc = _pick(tmp, ["receita_pc_real", "receita_pc", "x_level", "receita_real_pc", "receita_real_brl_pc"])
    if y_pc and x_pc:
        tmp["ln_y"] = np.log(pd.to_numeric(tmp[y_pc], errors="coerce"))
        tmp["ln_x"] = np.log(pd.to_numeric(tmp[x_pc], errors="coerce"))
        return tmp[["ln_y", "ln_x"]].replace([np.inf, -np.inf], np.nan).dropna()

    pib_total = _pick(tmp, ["pib_real_brl", "pib_real", "y"])
    rec_total = _pick(tmp, ["receita_real_brl", "receita_real", "x", "arrecadacao_real_brl"])
    pop_col   = _pick(tmp, ["pop", "population"])
    if pib_total and rec_total and pop_col:
        tmp["pib_pc_real__auto"] = pd.to_numeric(tmp[pib_total], errors="coerce") / pd.to_numeric(tmp[pop_col], errors="coerce")
        tmp["receita_pc_real__auto"] = pd.to_numeric(tmp[rec_total], errors="coerce") / pd.to_numeric(tmp[pop_col], errors="coerce")
        tmp["ln_y"] = np.log(tmp["pib_pc_real__auto"])
        tmp["ln_x"] = np.log(tmp["receita_pc_real__auto"])
        return tmp[["ln_y", "ln_x"]].replace([np.inf, -np.inf], np.nan).dropna()

    if pib_total and rec_total:
        tmp["ln_y"] = np.log(pd.to_numeric(tmp[pib_total], errors="coerce"))
        tmp["ln_x"] = np.log(pd.to_numeric(tmp[rec_total], errors="coerce"))
        return tmp[["ln_y", "ln_x"]].replace([np.inf, -np.inf], np.nan).dropna()

    y_reb = _pick(tmp, ["pib_real_rebased_2008"])
    x_reb = _pick(tmp, ["receita_real_rebased_2008"])
    if y_reb and x_reb:
        tmp["ln_y"] = np.log(pd.to_numeric(tmp[y_reb], errors="coerce"))
        tmp["ln_x"] = np.log(pd.to_numeric(tmp[x_reb], errors="coerce"))
        return tmp[["ln_y", "ln_x"]].replace([np.inf, -np.inf], np.nan).dropna()

    dln_y = _pick(tmp, ["dln_pib_real", "dln_y"])
    dln_x = _pick(tmp, ["dln_receita_real", "dln_x"])
    if dln_y and dln_x:
        rec = _reconstruct_from_diffs(tmp[[dln_y, dln_x]].apply(pd.to_numeric, errors="coerce"), dln_y, dln_x)
        return rec

    raise KeyError(f"Não encontrei forma de criar ln_y/ln_x. Colunas: {list(df.columns)}")

def _read_csv_any(path: Path, date_col_guess="date", freq=None) -> pd.DataFrame:
    df = pd.read_csv(path)
    if date_col_guess in df.columns:
        df[date_col_guess] = pd.to_datetime(df[date_col_guess], errors="coerce")
        df = df.sort_values(date_col_guess).set_index(date_col_guess)
    else:
        try:
            df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
            df = df.sort_values(df.columns[0]).set_index(df.columns[0])
        except Exception:
            pass
    df.index.name = "date"
    if freq:
        try:
            df = df.asfreq(freq)
        except Exception:
            pass
    return df

def load_or_create_processed(args) -> pd.DataFrame:
    if PROCESSED_CSV.exists():
        base = _read_csv_any(PROCESSED_CSV)
        return _ensure_ln_columns(base)
    if not args.raw:
        raise AssertionError("idecep_processed.csv não existe. Informe --raw CSV.")
    raw = _read_csv_any(Path(args.raw), date_col_guess=args.date_col, freq=args.freq)
    return _ensure_ln_columns(raw)

# ============ métricas / modelos ============
def adf_series(x: pd.Series):
    x = x.dropna()
    stat = adfuller(x, autolag="AIC")
    return {"ADF stat": float(stat[0]), "ADF p": float(stat[1]), "nobs": int(stat[3])}

def kpss_series(x: pd.Series):
    x = x.dropna()
    # H0 da KPSS: série é estacionária (ao redor de nível). Usar regression="c" por padrão.
    stat, p, *_ = kpss(x, regression="c", nlags="auto")
    return {"KPSS stat": float(stat), "KPSS p": float(p)}

def diff_log(df: pd.DataFrame):
    return df["ln_y"].diff(), df["ln_x"].diff()

def engle_granger_ecm(df: pd.DataFrame, p_lags=1, q_lags=1):
    y = df["ln_y"]
    X = add_constant(df["ln_x"])
    ols_coint = OLS(y, X, missing="drop").fit()
    resid = ols_coint.resid

    ecm = pd.DataFrame({
        "dln_y": y.diff(),
        "dln_x": df["ln_x"].diff(),
        "ect": resid.shift(1)
    }, index=df.index)

    for i in range(1, p_lags + 1):
        ecm[f"dln_y_lag{i}"] = ecm["dln_y"].shift(i)
    for j in range(1, q_lags + 1):
        ecm[f"dln_x_lag{j}"] = ecm["dln_x"].shift(j)

    ecm = ecm.dropna()
    Y = ecm["dln_y"]
    Z = add_constant(ecm.drop(columns=["dln_y"]))
    mdl = OLS(Y, Z, missing="drop").fit()

    return {
        "coint_beta": float(ols_coint.params.get("ln_x", np.nan)),
        "coint_beta_p": float(ols_coint.pvalues.get("ln_x", np.nan)),
        "lambda_coef": float(mdl.params.get("ect", np.nan)),
        "lambda_p": float(mdl.pvalues.get("ect", np.nan)),
        "model": mdl,
        "resid": mdl.resid
    }

def run_diagnostics_full(model, resid, lj_lags_1=12, lj_lags_2=24, bg_lags=12):
    # Ljung-Box
    lb1 = acorr_ljungbox(resid, lags=[lj_lags_1], return_df=True)
    lb2 = acorr_ljungbox(resid, lags=[lj_lags_2], return_df=True)
    lb_p1 = float(lb1["lb_pvalue"].iloc[0])
    lb_p2 = float(lb2["lb_pvalue"].iloc[0])

    # Breusch-Godfrey
    _, bg_p, _, _ = acorr_breusch_godfrey(model, nlags=bg_lags)

    # ARCH LM e F
    arch_out = het_arch(resid, maxlag=lj_lags_1)  # (lm_stat, lm_p, f_stat, f_p)
    arch_lm_p = float(arch_out[1]) if len(arch_out) > 1 else float("nan")
    arch_f_p  = float(arch_out[3]) if len(arch_out) > 3 else float("nan")

    # Jarque-Bera (normalidade)
    jb_stat, jb_p, _, _ = jarque_bera(resid)

    # Durbin-Watson
    dw = float(durbin_watson(resid))

    return {
        "Ljung-Box p (12)": lb_p1,
        "Ljung-Box p (24)": lb_p2,
        "BG p (12)": float(bg_p),
        "ARCH LM p": arch_lm_p,
        "ARCH F p": arch_f_p,
        "Jarque-Bera p": float(jb_p),
        "Durbin-Watson": dw,
    }

# ============ validação ============
def validate_outputs():
    checks = []
    for p in [STATION_CSV, ROLLING_CSV, DIAG_CSV, GATE_CSV]:
        ok = p.exists()
        nrows = ncols = nan_any = "NA"
        if ok:
            try:
                df = pd.read_csv(p)
                nrows, ncols = df.shape
                nan_any = bool(df.isna().any().any())
                ok = ok and (nrows > 0) and (ncols > 0)
            except Exception:
                ok = False
        checks.append((p.name, ok, nrows, ncols, nan_any))

    print("\n[Validação] Arquivos gerados em files_to_latex/idecep/")
    for name, ok, nrows, ncols, nan_any in checks:
        status = "OK" if ok else "FALHOU"
        print(f" - {name:32s}  {status:7s}  linhas={nrows}  colunas={ncols}  temNaN={nan_any}")

    try:
        diag = pd.read_csv(DIAG_CSV)
        expected_cols = {
            "ADF p","KPSS p","Ljung-Box p (12)","Ljung-Box p (24)",
            "BG p (12)","ARCH LM p","Jarque-Bera p","Durbin-Watson"
        }
        missing = expected_cols - set(diag.columns)
        if missing:
            print(f"[Validação] Aviso: colunas ausentes em {DIAG_CSV.name}: {missing}")
    except Exception as e:
        print(f"[Validação] Não foi possível validar {DIAG_CSV.name}: {e}")

# ============ CLI ============
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=str, default=None,
                    help="CSV bruto (usado se idecep_processed.csv NÃO existir).")
    ap.add_argument("--date-col", type=str, default="date",
                    help="Nome da coluna de data no CSV bruto (default: date).")
    ap.add_argument("--freq", type=str, default=None,
                    help="Frequência para reamostrar (ex.: 'Q','M').")
    ap.add_argument("--rolling", type=int, default=4,
                    help="Janela para médias móveis nas difs (default=4).")
    ap.add_argument("--no-validate", action="store_true",
                    help="Se passado, não executa a rotina de validação ao final.")
    ap.add_argument("--extra-from", type=str, default=None,
                    help="Diretório com CSVs extras (ex.: outputs/idecep_uses) para copiar/sanear.")
    ap.add_argument("--float-digits", type=int, default=6,
                    help="Precisão de floats em colunas não formatadas (default=6).")
    return ap.parse_args()

# ============ Main ============
def main():
    ensure_dirs()
    args = parse_args()

    # 1) dados básicos
    df = load_or_create_processed(args)
    dln_y, dln_x = diff_log(df)

    # 2) stationarity (agora com KPSS também)
    station_rows = [
        {"series": "ln_y",  **adf_series(df["ln_y"]),  **kpss_series(df["ln_y"])},
        {"series": "ln_x",  **adf_series(df["ln_x"]),  **kpss_series(df["ln_x"])},
        {"series": "dln_y", **adf_series(dln_y),       **kpss_series(dln_y.dropna())},
        {"series": "dln_x", **adf_series(dln_x),       **kpss_series(dln_x.dropna())},
    ]
    station_df = pd.DataFrame(station_rows)
    _to_csv_latex(station_df, STATION_CSV, float_ndigits=args.float_digits, index=False)

    # 3) rolling diffs (com coluna date)
    roll_w = args.rolling
    roll_df = pd.DataFrame({
        "date": df.index,
        "dln_y": dln_y.values,
        "dln_x": dln_x.values,
        f"rolling_mean_dln_y_{roll_w}": dln_y.rolling(roll_w).mean().values,
        f"rolling_mean_dln_x_{roll_w}": dln_x.rolling(roll_w).mean().values
    })
    _to_csv_latex(roll_df, ROLLING_CSV, float_ndigits=args.float_digits, index=False)

    # 4) ECM + diagnósticos (completos nos resíduos)
    ecm = engle_granger_ecm(df, p_lags=1, q_lags=1)
    diag = run_diagnostics_full(ecm["model"], ecm["resid"], lj_lags_1=12, lj_lags_2=24, bg_lags=12)

    # ADF/KPSS nos resíduos também (para manter o cabeçalho "ANTERIOR")
    resid = ecm["resid"].dropna()
    adf_res = adfuller(resid, autolag="AIC")
    kpss_res = kpss(resid, regression="c", nlags="auto")

    diag_row = {
        # p-values formatados (string) para não virar 0.000000
        "ADF p":            _format_p(adf_res[1]),
        "KPSS p":           _format_p(kpss_res[1]),
        "Ljung-Box p (12)": _format_p(diag["Ljung-Box p (12)"]),
        "Ljung-Box p (24)": _format_p(diag["Ljung-Box p (24)"]),
        "BG p (12)":        _format_p(diag["BG p (12)"]),
        "ARCH LM p":        _format_p(diag["ARCH LM p"]),
        "Jarque-Bera p":    _format_p(diag["Jarque-Bera p"]),
        "Durbin-Watson":    _format_p(diag["Durbin-Watson"]),
    }
    diag_df = pd.DataFrame([diag_row])
    _to_csv_latex(diag_df, DIAG_CSV, float_ndigits=args.float_digits, index=False)

    # 5) Gate com flags numéricas 0/1
    ok_stationarity = int((station_df.loc[station_df.series == "dln_y", "ADF p"].astype(float).values[0] < 0.05) and
                          (station_df.loc[station_df.series == "dln_x", "ADF p"].astype(float).values[0] < 0.05))
    ect_sig = int((not np.isnan(ecm["lambda_coef"])) and (ecm["lambda_coef"] < 0) and (ecm["lambda_p"] < 0.10))

    # para gate, usar os valores numéricos originais (não formatados)
    lb12 = float(diag["Ljung-Box p (12)"])
    bg12 = float(diag["BG p (12)"])
    arch_lm = float(diag["ARCH LM p"]) if diag["ARCH LM p"] == diag["ARCH LM p"] else 1.0

    auto_ok = int((lb12 > 0.05) and (bg12 > 0.05))
    arch_ok = int(arch_lm > 0.05)

    gate = pd.DataFrame([{
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "Estacionariedade_dln": ok_stationarity,
        "ECT_significativo_neg": ect_sig,
        "Sem_autocorrelacao": auto_ok,
        "Sem_ARCH": arch_ok,
        "lambda_coef": ecm["lambda_coef"],
        "lambda_p": ecm["lambda_p"],
        "coint_beta": ecm["coint_beta"],
        "coint_beta_p": ecm["coint_beta_p"],
        "rolling_window": int(roll_w)
    }])
    _to_csv_latex(gate, GATE_CSV, float_ndigits=args.float_digits, index=False)

    # 6) Extras idecep_uses (auto-descoberta se --extra-from não for passada)
    extra_dirs = []
    if args.extra_from:
        extra_dirs.append(Path(args.extra_from))
    else:
        # tenta caminhos comuns com e sem 's' e absolutos
        extra_dirs.extend([
            ROOT / "outputs" / "idecep_uses",
            ROOT / "output"  / "idecep_uses",
            Path("/outputs/idecep_uses"),
            Path("/output/idecep_uses"),
        ])

    for d in extra_dirs:
        _copy_and_clean_all_csvs(d, LATEX_DIR, float_ndigits=args.float_digits)

    print("\n[IDECEP → LaTeX] ✅ Concluído")
    print(f"Saídas (sem timestamp): {LATEX_DIR}")
    for f in [STATION_CSV, ROLLING_CSV, DIAG_CSV, GATE_CSV]:
        print(" -", f)

    if not args.no_validate:
        validate_outputs()

if __name__ == "__main__":
    main()
