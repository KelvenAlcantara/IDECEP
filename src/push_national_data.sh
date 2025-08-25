#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT="${ROOT}/data/push_nacional"
mkdir -p "$OUT"

UA="idecep-cli/1.0"

echo ">> Salvando arquivos em: $OUT"
echo

# =============== Funções utilitárias ==================

# BCB com FATIA de 5 em 5 anos (robusto p/ 10813/432)
sgs_chunk_csv () {
  local series_id="$1"; local colname="$2"; local start_year="$3"; local end_year="$4"; local out_csv="$5"; local span_years=5
  : > "$out_csv"; local wrote_header=0
  echo "[BCB] série ${series_id} (${colname}) – janelas ${start_year}-${end_year} (5y)"
  local year="$start_year"
  while [ "$year" -le "$end_year" ]; do
    local endy=$(( year + span_years - 1 )); [ "$endy" -gt "$end_year" ] && endy="$end_year"
    local di="$(printf "%02d/%02d/%04d" 01 01 "$year")"
    local df="$(printf "%02d/%02d/%04d" 31 12 "$endy")"
    local chunk="$(mktemp)"
    curl -sS --retry 3 --retry-delay 1 -H "User-Agent: ${UA}" \
      "https://api.bcb.gov.br/dados/serie/bcdata.sgs.${series_id}/dados?formato=csv&dataInicial=${di}&dataFinal=${df}" > "$chunk" || true
    if [ -s "$chunk" ]; then
      local sep=';'; head -n1 "$chunk" | grep -q ',' && sep=','
      if [ "$wrote_header" -eq 0 ]; then
        awk -v FS="$sep" -v OFS=',' -v COL="$colname" 'NR==1{$1="date";$2=COL;print;next}{gsub(",",".",$2);print $1,$2}' "$chunk" >> "$out_csv"
        wrote_header=1
      else
        tail -n +2 "$chunk" | awk -v FS="$sep" -v OFS=',' '{gsub(",",".",$2);print $1,$2}' >> "$out_csv"
      fi
    fi
    rm -f "$chunk"; year=$(( endy + 1 ))
  done
  [ -s "$out_csv" ] || echo "date,${colname}" > "$out_csv"
}

ipea_to_csv () {
  local ser="$1"; local colname="$2"; local out_csv="$3"
  echo "[IPEA] série ${ser} (${colname})"
  curl -sS --retry 3 --retry-delay 1 -H "Accept: application/json" \
    "http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='${ser}')" \
  | jq -r '.value[] | [.VALDATA, .VALVALOR] | @csv' \
  | sed '1i"date","'"$colname"'"' \
  > "$out_csv" || echo "\"date\",\"${colname}\"" > "$out_csv"
}

# SIDRA — PIM (produção industrial)
sidra_pim_to_csv () {
  local out_csv="$1"
  echo "[SIDRA] producao_industrial (agregado 3653)"
  curl -sS --retry 4 --retry-delay 1 --globoff -H "User-Agent: ${UA}" \
    "https://servicodados.ibge.gov.br/api/v3/agregados/3653/periodos/all/variaveis/3135?localidades=N1[all]" \
  | jq -r '
      if (type=="array") then
        .[]
        | select(type=="object" and has("resultados") and (.resultados|type)=="array" and ((.resultados|length)>0))
        | .resultados[0].series[0].serie
        | to_entries
        | .[]
        | [ (if ((.key|type)=="string" and (.key|length)==6)
              then (.key[0:4] + "-" + .key[4:6] + "-01")
              else (.key + "-01-01") end),
            (.value|tostring|gsub(",";".")) ]
        | @csv
      else empty end' \
  | sed '1i"date","producao_industrial"' > "$out_csv" \
  || echo "\"date\",\"producao_industrial\"" > "$out_csv"
}

years_from_csv () {
  local infile="$1"; local outfile="$2"
  awk -F',' 'NR>1{
                split($1,d,"-"); y=d[1];
                if(min==""||y<min)min=y;
                if(max==""||y>max)max=y
              }
              END{if(min!="")print min"-"max; else print "NA"}' "$infile" > "$outfile"
}

# =============== Séries ===============

# BCB — todas em janelas de 5 anos
sgs_chunk_csv "1207"  "pib_nominal_brl"      1980 2025 "${OUT}/pib_nominal_brl.csv"
years_from_csv "${OUT}/pib_nominal_brl.csv"  "${OUT}/pib_nominal_brl.years"

sgs_chunk_csv "4505"  "divida_bruta_pct_pib" 2000 2025 "${OUT}/divida_bruta_pct_pib.csv"
years_from_csv "${OUT}/divida_bruta_pct_pib.csv" "${OUT}/divida_bruta_pct_pib.years"

sgs_chunk_csv "10813" "cambio_usd"           2000 2025 "${OUT}/cambio_usd.csv"
years_from_csv "${OUT}/cambio_usd.csv"       "${OUT}/cambio_usd.years"

sgs_chunk_csv "433"   "ipca_indice"          1980 2025 "${OUT}/ipca_indice.csv"
years_from_csv "${OUT}/ipca_indice.csv"      "${OUT}/ipca_indice.years"

sgs_chunk_csv "432"   "selic_meta"           1980 2025 "${OUT}/selic_meta.csv"
years_from_csv "${OUT}/selic_meta.csv"       "${OUT}/selic_meta.years"

# IPEA — % do PIB
ipea_to_csv "WEO_RECGGWEOBRA" "receita_total_pct_pib" "${OUT}/receita_total_pct_pib.csv"
years_from_csv "${OUT}/receita_total_pct_pib.csv" "${OUT}/receita_total_pct_pib.years"

ipea_to_csv "WEO_DESPGGWEOBRA" "despesa_total_pct_pib" "${OUT}/despesa_total_pct_pib.csv"
years_from_csv "${OUT}/despesa_total_pct_pib.csv" "${OUT}/despesa_total_pct_pib.years"

# SIDRA — PIM (apenas)
sidra_pim_to_csv "${OUT}/producao_industrial.csv"
years_from_csv "${OUT}/producao_industrial.csv" "${OUT}/producao_industrial.years"

# =============== Cópia p/ processed/nacional (com renomeação p/ o build) ===============
echo
echo ">> Copiando dados para processed/nacional/"
PROC="${ROOT}/data/processed/nacional"
mkdir -p "$PROC"

cp "${OUT}/pib_nominal_brl.csv"          "${PROC}/pib_nominal_brl.csv"
cp "${OUT}/receita_total_pct_pib.csv"    "${PROC}/receita_total_pct_pib.csv"
cp "${OUT}/despesa_total_pct_pib.csv"    "${PROC}/despesa_total_pct_pib.csv"
cp "${OUT}/divida_bruta_pct_pib.csv"     "${PROC}/divida_bruta_pct_pib.csv"
cp "${OUT}/cambio_usd.csv"               "${PROC}/cambio_usd.csv"
cp "${OUT}/ipca_indice.csv"              "${PROC}/ipca_indice.csv"
cp "${OUT}/selic_meta.csv"               "${PROC}/selic_meta.csv"
cp "${OUT}/producao_industrial.csv"      "${PROC}/producao_industrial.csv"

# nomes esperados pelo build_features/project.yaml
cp "${PROC}/receita_total_pct_pib.csv" "${PROC}/receita_pct_pib.csv"
cp "${PROC}/despesa_total_pct_pib.csv" "${PROC}/despesa_pct_pib.csv"
cp "${PROC}/divida_bruta_pct_pib.csv"  "${PROC}/divida_pct_pib.csv"

echo
echo "✅ Pronto. Resultados em $OUT e copiados/renomeados para $PROC"
ls -lh "$OUT"