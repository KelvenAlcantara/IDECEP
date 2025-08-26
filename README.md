# IDECEP – Índice de Desenvolvimento Econômico pela Capacidade de Execução Pública

## 📌 Descrição
O **IDECEP** é um índice desenvolvido para avaliar a relação entre desenvolvimento econômico e a capacidade de execução pública.  
O modelo busca medir, de forma quantitativa, o impacto da arrecadação e das despesas públicas sobre o crescimento econômico, com ênfase em sustentabilidade fiscal e eficiência na alocação de recursos.

Este repositório contém os scripts, dados e relatórios do projeto, com foco em análise estatística e econometria aplicada ao contexto brasileiro.

---

## ⚙️ Metodologia
O índice é calculado a partir de:
- **PIB real per capita (ln y)** e **Receita pública per capita (ln x)**  
- Testes de estacionariedade (ADF)  
- Modelos de cointegração e correção de erro (ECM)  
- Critérios de diagnóstico: autocorrelação, heterocedasticidade e significância do ECT  

Formas do índice:
1. **IDECEP Simples** – baseado em variações de arrecadação e PIB.  
2. **IDECEP de Regressão** – inclui testes de cointegração e ajuste de resíduos.  

---

## 📊 Resultados
- Gráficos de séries temporais comparando **y_real vs y_pred**.  
- Relatórios de diagnóstico econométrico (gate de aprovação do modelo).  
- Tabelas de resíduos, correlação e testes estatísticos.

---

## 📐 Matemática

O cálculo do **IDECEP** é fundamentado em modelos de séries temporais, integrando PIB real per capita e Receita pública per capita.  

### Variáveis principais
- \( P_t \): PIB real per capita em logaritmo natural (proxy da capacidade produtiva da população)  
- \( E_t \): Receita pública per capita em logaritmo natural (proxy da capacidade de execução do Estado)  
- \( \Delta \): operador de diferença (variação)  
- \( \lambda \): coeficiente de correção de erro (ECT)  

### Forma Simples

A forma simples do índice captura a razão entre variações da execução estatal e variações da produção populacional:

$$
\mathrm{IDECEP}_t = \frac{\Delta E_t}{\Delta P_t} \times 100
$$

### Forma de Regressão (ECM)

A versão mais robusta utiliza um modelo de correção de erro:

$$
\Delta P_t = \alpha + \beta\,\Delta E_t + \lambda\big(P_{t-1} - \gamma E_{t-1}\big) + \varepsilon_t
$$

**Onde:**
- $\beta$: impacto de curto prazo da receita sobre o PIB  
- $\gamma$: relação de longo prazo (cointegração)  
- $\lambda$: velocidade de ajuste para o equilíbrio de longo prazo


### Critérios de Diagnóstico
O modelo é validado através de:
- **Teste ADF** → estacionariedade  
- **Ljung-Box e Breusch-Godfrey** → autocorrelação  
- **ARCH LM** → heterocedasticidade  

Esses testes compõem o **"gate" de aprovação** que define se o modelo é estatisticamente válido.



# 📡 Coleta Nacional via APIs Públicas (BCB, IPEA, SIDRA)

O script `src/push_national_data.sh` baixa e padroniza automaticamente as séries nacionais usadas no IDECEP.  
Ele cria `data/push_nacional/` com CSVs brutos e copia versões padronizadas para `data/processed/nacional/`, incluindo **renomeações compatíveis** com o pipeline do projeto.

## ✅ Pré‑requisitos
- **bash**, **curl**, **awk**, **sed**
- **jq** (para processar JSON)
- Acesso à internet

Instalação rápida (Ubuntu/Debian):
```bash
sudo apt update
sudo apt install -y curl jq
```

## ▶️ Como executar coleta
```bash
chmod +x src/push_national_data.sh
./src/push_national_data.sh
```

Saídas principais:
- Brutos: `data/push_nacional/*.csv` e arquivos de intervalo `*.years`
- Padronizados: `data/processed/nacional/*.csv` (com nomes esperados, ex: `*_pct_pib.csv`)

## 🌐 Fontes e Séries coletadas
- **BCB (SGS)**: PIB nominal, dívida/PIB, câmbio, IPCA, Selic (IDs 1207, 4505, 10813, 433, 432).
- **IPEA (OData API)**: Receita total/PIB e Despesa total/PIB (IDs WEO_RECGGWEOBRA, WEO_DESPGGWEOBRA).
- **IBGE SIDRA**: Produção industrial (Agregado 3653, variável 3135).

## 🔗 Integração no pipeline
Exemplo:
```bash
# 1) Coleta nacional
./src/push_national_data.sh

# 2) Build anual unificado
python scripts/build_features.py --in-dir data/processed/nacional --out data/build/features_nacional.csv

# 3) Trimestral
python scripts/build_quarterly.py --in data/build/features_nacional.csv --out data/splited/features_quarterly.csv --flows pib_nominal_brl --verbose

# 4) Mensal
python scripts/build_monthly.py --in data/build/features_nacional.csv --out data/splited/features_monthly.csv --flows pib_nominal_brl --verbose

# 5) Artefatos LaTeX
python scripts/normalize_csv_idecep_latex.py --raw data/splited/features_quarterly.csv --date-col date
```

---

# 🚀 Execução do Pipeline

> Pré-requisitos: Python 3.10+ (recomendado), `pip install -r requirements.txt`.

## 1) Construir dataset anual unificado
```bash
python scripts/build_features.py
```
Saída: `data/build/features_nacional.csv`.

## 2) Derivar periodicidades
**Trimestral:**
```bash
python scripts/build_quarterly.py --in data/build/features_nacional.csv --out data/splited/features_quarterly.csv --flows pib_nominal_brl --verbose
```
**Mensal:**
```bash
python scripts/build_monthly.py --in data/build/features_nacional.csv --out data/splited/features_monthly.csv --flows pib_nominal_brl --verbose
```

## 3) Artefatos para LaTeX
```bash
python scripts/normalize_csv_idecep_latex.py
# ou
python scripts/normalize_csv_idecep_latex.py --raw data/splited/features_quarterly.csv --date-col date
```
Arquivos:  
`idecep_stationarity_checks.csv`, `idecep_rolling_logdiff.csv`, `idecep_ecm_diagnostics.csv`, `idecep_gate_report.csv`.

---

# 📓 Notebooks
- **`notebooks/idecep.ipynb`** — exploração, checagens e iteração de modelo.  
Abrir com:
```bash
jupyter lab
# ou
jupyter notebook
```
Certifique-se de ativar o mesmo ambiente virtual antes de rodar o notebook.

---

## 📜 Licença
O código-fonte deste projeto está licenciado sob a **GNU GPL v3** – veja o arquivo `LICENSE` para mais detalhes.  

A documentação, relatórios e textos analíticos (incluindo arquivos LaTeX e README) estão licenciados sob a **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** – veja o arquivo `LICENSE_DOCS`.  

---

✍️ **Autor:** Kelven de Alcantara Bonfim  
📍 Londrina – Paraná – Brasil
