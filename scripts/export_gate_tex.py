#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Exporta .tex e gráficos .pdf para o "gate" do IDECEP.
- Lê CSVs em outputs/idecep_validation/
- Extrai métricas: ADF, KPSS, Ljung-Box, BG, ARCH, JB, DW, ECT significativo, Estacionariedade
- Tenta extrair lambda (ECT) se disponível
- Gera PDF: observado_vs_previsto.pdf e residuos.pdf a partir de idecep_rolling_logdiff.csv
- Escreve .tex: GATE_1_pretests.tex, GATE_2_ecm.tex e GATE_include.tex

Uso:
    python scripts/export_gate_tex.py
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ---------- Utilidades ----------
def find_root(start: Path | None = None) -> Path:
    start = Path(start or Path.cwd()).resolve()
    markers = {"configs", "scripts", "src", "requirements.txt", "README.md", ".git"}
    for p in [start, *start.parents]:
        if any((p / m).exists() for m in markers):
            return p
    return start

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def read_csv_if_exists(p: Path) -> pd.DataFrame | None:
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception as e:
            print(f"[WARN] Falha lendo {p}: {e}", file=sys.stderr)
    return None

def find_first_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns: return c
    # try case-insensitive
    lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in lower: return lower[c.lower()]
    return None

# ---------- Geração de gráficos ----------
def plot_obs_pred_pdf(df: pd.DataFrame, out_pdf: Path) -> None:
    """
    Espera colunas: date (opcional), y_true, y_pred
    """
    # normaliza data
    date_col = None
    for c in ["date","Date","DATA","data"]:
        if c in df.columns:
            date_col = c; break

    x = pd.to_datetime(df[date_col]) if date_col else pd.RangeIndex(len(df))
    y_true_col = find_first_col(df, ["y_true","y","dln_y","dln_pib_real"])
    y_pred_col = find_first_col(df, ["y_pred","yhat","pred"])

    if not y_pred_col:
        raise ValueError("Coluna de previsão não encontrada (y_pred / yhat / pred).")

    plt.figure(figsize=(9,5))
    plt.plot(x, df[y_pred_col], label="Previsto", linewidth=1.5)
    if y_true_col:
        plt.plot(x, df[y_true_col], label="Observado", linewidth=1.0)
        plt.title("Observado vs Previsto (escala do alvo usado no treino)")
    else:
        plt.title("Previsto (sem observado disponível)")
    plt.xlabel("date" if date_col else "t")
    plt.ylabel(y_true_col or y_pred_col)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=200)
    plt.close()

def plot_residuos_pdf(df: pd.DataFrame, out_pdf: Path) -> None:
    y_true_col = find_first_col(df, ["y_true","y","dln_y","dln_pib_real"])
    y_pred_col = find_first_col(df, ["y_pred","yhat","pred"])
    if not (y_true_col and y_pred_col):
        # Sem observado ou previsto, não dá para fazer resíduo
        print("[INFO] Não foi possível gerar resíduos (faltam colunas y_true/y_pred).")
        return
    resid = df[y_true_col] - df[y_pred_col]
    plt.figure(figsize=(9,5))
    plt.plot(resid.values, linewidth=1.0)
    plt.title("Resíduos (y - yhat)")
    plt.xlabel("t")
    plt.ylabel("resíduo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_pdf, dpi=200)
    plt.close()

# ---------- LaTeX ----------
def tex_escape(s: str) -> str:
    # Escapa caracteres típicos do LaTeX
    if s is None: return ""
    rep = {
        "&":"\\&","%":"\\%","$":"\\$","#":"\\#","_":"\\_",
        "{":"\\{","}":"\\}","~":"\\textasciitilde{}","^":"\\textasciicircum{}",
        "\\":"\\textbackslash{}"
    }
    for k,v in rep.items():
        s = s.replace(k,v)
    return s

def write_gate1_pretests(df_stat: pd.DataFrame, out_tex: Path) -> None:
    """
    Espera colunas em `idecep_stationarity_checks.csv` e/ou `idecep_ecm_diagnostics.csv`.
    Produz uma tabela enxuta com p-valores de ADF, KPSS e testes de autocorrelação/normalidade.
    """
    lines = []
    lines.append("% Auto-gerado por export_gate_tex.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{IDECEP – Pré-testes de Estacionariedade e Diagnósticos}")
    lines.append("\\label{tab:idecep_pretests}")
    lines.append("\\begin{tabular}{lcccccccc}")
    lines.append("\\toprule")
    lines.append("Série/Teste & ADF $p$ & KPSS $p$ & Ljung–Box $p$ (12) & Ljung–Box $p$ (24) & BG $p$ (12) & ARCH LM $p$ & Jarque–Bera $p$ & DW \\\\")
    lines.append("\\midrule")

    # Tenta agregar informações de stationarity + diagnostics em uma linha
    # Preferência: se df_stat tem múltiplas linhas/series, sumariza linha a linha
    # Nomes prováveis:
    cols = {
        "adf":"ADF p",
        "kpss":"KPSS p",
        "lj12":"Ljung-Box p (12)",
        "lj24":"Ljung-Box p (24)",
        "bg12":"BG p (12)",
        "arch":"ARCH LM p",
        "jb":"Jarque-Bera p",
        "dw":"Durbin-Watson",
        "series":"series"
    }
    # Renomeia case-insensitive
    rename_map = {}
    for c in df_stat.columns:
        lc = c.lower().strip()
        if "adf" in lc and "p" in lc: rename_map[c] = "ADF p"
        elif "kpss" in lc and "p" in lc: rename_map[c] = "KPSS p"
        elif "ljung" in lc and "12" in lc: rename_map[c] = "Ljung-Box p (12)"
        elif "ljung" in lc and "24" in lc: rename_map[c] = "Ljung-Box p (24)"
        elif (("bg" in lc) or ("breusch" in lc)) and "12" in lc: rename_map[c] = "BG p (12)"
        elif "arch" in lc and "p" in lc: rename_map[c] = "ARCH LM p"
        elif ("jarque" in lc) or ("jb" in lc): rename_map[c] = "Jarque-Bera p"
        elif ("durbin" in lc) or (lc.startswith("dw")): rename_map[c] = "Durbin-Watson"
        elif lc.startswith("series"): rename_map[c] = "series"
    df_r = df_stat.rename(columns=rename_map).copy()

    # Se não há colunas de testes completos, tenta fallback vazio
    wanted = ["series","ADF p","KPSS p","Ljung-Box p (12)","Ljung-Box p (24)","BG p (12)","ARCH LM p","Jarque-Bera p","Durbin-Watson"]
    for _, row in df_r.iterrows():
        vals = [tex_escape(str(row.get("series","-")))]
        for k in wanted[1:]:
            v = row.get(k, "")
            vals.append(f"{v:.4g}" if isinstance(v, (int,float,np.floating)) and np.isfinite(v) else tex_escape(str(v)) )
        lines.append(" {} \\\\".format(" & ".join(vals)))

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Escrito {out_tex}")

def write_gate2_ecm(df_diag: pd.DataFrame, df_gate: pd.DataFrame | None, out_tex: Path) -> None:
    """
    Usa idecep_ecm_diagnostics.csv + idecep_gate_report.csv
    Inclui lambda (ECT) se disponível em alguma coluna; caso contrário, marca N/A.
    """
    # Detecta lambda em colunas possíveis
    lambda_candidates = ["lambda_ect","ect_lambda","lambda","ECT","ect","lambda (ECT)"]
    lambda_val = None
    for c in df_diag.columns:
        if c.lower().strip() in [lc.lower() for lc in lambda_candidates]:
            try:
                lambda_val = float(df_diag.iloc[0][c])
                break
            except Exception:
                pass

    # ECT significativo
    ect_sig = None
    if df_gate is not None:
        # tenta achar coluna que seja booleana/True/False
        for c in df_gate.columns:
            lc = c.lower()
            if "ect" in lc and ("sig" in lc or "<0" in lc or "signific" in lc):
                ect_sig = str(df_gate.iloc[0][c])
                break

    lines = []
    lines.append("% Auto-gerado por export_gate_tex.py")
    lines.append("\\begin{table}[H]")
    lines.append("\\centering")
    lines.append("\\caption{IDECEP – Diagnósticos do ECM e Correção de Erro}")
    lines.append("\\label{tab:idecep_ecm}")
    lines.append("\\begin{tabular}{lc}")
    lines.append("\\toprule")
    lines.append("Métrica & Valor \\\\")
    lines.append("\\midrule")

    # Inclui principais métricas do df_diag (p-valores)
    rename_map = {}
    for c in df_diag.columns:
        lc = c.lower()
        if "adf" in lc and "p" in lc: rename_map[c] = "ADF $p$"
        elif "kpss" in lc and "p" in lc: rename_map[c] = "KPSS $p$"
        elif "ljung" in lc and "12" in lc: rename_map[c] = "Ljung--Box $p$ (12)"
        elif "ljung" in lc and "24" in lc: rename_map[c] = "Ljung--Box $p$ (24)"
        elif (("bg" in lc) or ("breusch" in lc)) and "12" in lc: rename_map[c] = "BG $p$ (12)"
        elif "arch" in lc and "p" in lc: rename_map[c] = "ARCH LM $p$"
        elif ("jarque" in lc) or ("jb" in lc): rename_map[c] = "Jarque--Bera $p$"
        elif ("durbin" in lc) or (lc.startswith("dw")): rename_map[c] = "Durbin--Watson"

    df_show = df_diag.rename(columns=rename_map).copy()
    for k in ["ADF $p$","KPSS $p$","Ljung--Box $p$ (12)","Ljung--Box $p$ (24)","BG $p$ (12)","ARCH LM $p$","Jarque--Bera $p$","Durbin--Watson"]:
        if k in df_show.columns:
            v = df_show.iloc[0][k]
            val = f"{v:.4g}" if isinstance(v, (int,float,np.floating)) and np.isfinite(v) else tex_escape(str(v))
            lines.append(f"{k} & {val} \\\\")

    # Lambda e ECT
    lines.append(f"$\\lambda$ (ECT) & {('%.4f' % lambda_val) if isinstance(lambda_val,(int,float,np.floating)) else 'N/A'} \\\\")
    if ect_sig is not None:
        lines.append(f"ECT significativo & {tex_escape(str(ect_sig))} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Escrito {out_tex}")

def write_include_snippet(out_dir: Path, figs: dict[str, Path]) -> None:
    """
    Cria um snippet LaTeX com \input dos .tex e \includegraphics dos PDFs.
    """
    lines = []
    lines.append("% Auto-gerado por export_gate_tex.py")
    lines.append("\\section{Gate de Validação do IDECEP}")
    lines.append("\\input{files_to_latex/idecep/GATE_1_pretests}")
    lines.append("\\input{files_to_latex/idecep/GATE_2_ecm}")
    lines.append("")
    lines.append("\\begin{figure}[H]")
    lines.append("\\centering")
    if "obs_pred" in figs:
        lines.append(f"\\includegraphics[width=.9\\textwidth]{{{figs['obs_pred'].as_posix()}}}")
        lines.append("\\caption{Observado vs. Previsto (ECM).}")
        lines.append("\\end{figure}")
        lines.append("")
        lines.append("\\begin{figure}[H]")
        lines.append("\\centering")
    if "residuos" in figs:
        lines.append(f"\\includegraphics[width=.9\\textwidth]{{{figs['residuos'].as_posix()}}}")
        lines.append("\\caption{Resíduos do ECM.}")
    lines.append("\\end{figure}")

    (out_dir / "GATE_include.tex").write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] Escrito {out_dir / 'GATE_include.tex'}")

# ---------- MAIN ----------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=None, help="Forçar raiz do projeto (opcional)")
    args = parser.parse_args()

    ROOT = find_root(Path(args.root) if args.root else None)
    val_dir = ROOT / "outputs" / "idecep_validation"
    latex_dir = ensure_dir(ROOT / "files_to_latex" / "idecep")

    # CSVs esperados
    p_stat = val_dir / "idecep_stationarity_checks.csv"
    p_diag = val_dir / "idecep_ecm_diagnostics.csv"
    p_gate = val_dir / "idecep_gate_report.csv"
    p_roll = val_dir / "idecep_rolling_logdiff.csv"

    df_stat = read_csv_if_exists(p_stat)
    df_diag = read_csv_if_exists(p_diag)
    df_gate = read_csv_if_exists(p_gate)
    df_roll = read_csv_if_exists(p_roll)

    if df_stat is None and df_diag is None:
        print("[ERRO] Nenhum CSV de diagnósticos encontrado em outputs/idecep_validation/", file=sys.stderr)
        sys.exit(1)

    # ---------- Tabelas .tex ----------
    if df_stat is None:
        # se não houver stationarity, tenta usar df_diag como insumo mínimo
        write_gate1_pretests(df_diag.copy(), latex_dir / "GATE_1_pretests.tex")
    else:
        write_gate1_pretests(df_stat.copy(), latex_dir / "GATE_1_pretests.tex")

    if df_diag is None:
        # fallback: usa df_stat
        write_gate2_ecm(df_stat.copy(), df_gate, latex_dir / "GATE_2_ecm.tex")
    else:
        write_gate2_ecm(df_diag.copy(), df_gate, latex_dir / "GATE_2_ecm.tex")

    # ---------- Gráficos .pdf ----------
    figs = {}
    if df_roll is not None:
        obs_pred_pdf = latex_dir / "observado_vs_previsto.pdf"
        resid_pdf    = latex_dir / "residuos.pdf"
        try:
            plot_obs_pred_pdf(df_roll.copy(), obs_pred_pdf)
            figs["obs_pred"] = Path("files_to_latex/idecep/observado_vs_previsto.pdf")
        except Exception as e:
            print(f"[WARN] Falha gerando observado_vs_previsto.pdf: {e}", file=sys.stderr)
        try:
            plot_residuos_pdf(df_roll.copy(), resid_pdf)
            figs["residuos"] = Path("files_to_latex/idecep/residuos.pdf")
        except Exception as e:
            print(f"[WARN] Falha gerando residuos.pdf: {e}", file=sys.stderr)
    else:
        print("[INFO] idecep_rolling_logdiff.csv não encontrado; gráficos não serão gerados.")

    # ---------- Snippet include ----------
    write_include_snippet(latex_dir, figs)

    print("\n[OK] Export finalizado.")
    print(f"- Tabelas: {latex_dir / 'GATE_1_pretests.tex'}, {latex_dir / 'GATE_2_ecm.tex'}")
    print(f"- Gráficos: {latex_dir / 'observado_vs_previsto.pdf'}, {latex_dir / 'residuos.pdf'}")
    print(f"- Snippet:  {latex_dir / 'GATE_include.tex'}")
    print("\nNo LaTeX, use: \\input{{files_to_latex/idecep/GATE_include}}")

if __name__ == "__main__":
    main()
