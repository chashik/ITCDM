#!/bin/bash
# Downloads the Belarusian data that cannot be reached from the analysis session (www.nbrb.by, www.bcse.by time out
# there): NBRB interest-rate statistics, banking-statistics bulletins and securities statistics; BCSE pages of
# Ministry of Finance and Belarusbank bonds with their trading results (prices, yields).
# Also the monthly average consumer prices of Russia from Rosstat (rosstat.gov.ru does not answer there either).
#
# Run from a Belarusian connection WITHOUT VPN, from the repository root:
#     bash code/lab10_economies/collect/fetch_blocked.sh
# Everything goes to code/data/raw/d9/manual/<site>/ ; nothing outside that folder is changed; no sudo needed.
# It takes a few minutes; a second run skips files that are already there.  At the end it prints a summary.

set -u
trap '' HUP          # keep going if the terminal is closed (the first run was killed by a hang-up)
ROOT="$(cd "$(dirname "$0")/../../data/raw/d9" 2>/dev/null && pwd || (mkdir -p "$(dirname "$0")/../../data/raw/d9" && cd "$(dirname "$0")/../../data/raw/d9" && pwd))"
OUT="$ROOT/manual"
LOG="$OUT/fetch_log.txt"
UA="Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
mkdir -p "$OUT/nbrb" "$OUT/bcse"
echo "=== $(date --iso-8601=seconds) fetch_blocked.sh" >> "$LOG"

reachable() {   # $1 = url
    curl -s -o /dev/null -m 20 -A "$UA" -w "%{http_code}" "$1" | grep -qE "^(2|3)"
}

for site in https://www.nbrb.by/ https://www.bcse.by/; do
    if reachable "$site"; then
        echo "OK   $site" | tee -a "$LOG"
    else
        echo "FAIL $site is not reachable — switch the VPN off (or use a Belarusian connection) and run again." | tee -a "$LOG"
        exit 1
    fi
done

# ---------------------------------------------------------------- NBRB
# Exactly the files that hold the data (found by the first, broad crawl): average interest rates of the credit-deposit
# market by year, rates on new loans by lending conditions, corporate securities market, the daily yield indicators
# of the market portfolio of government bonds (one page per month) and the monthly statistical bulletins.
echo "--- NBRB (www.nbrb.by)" | tee -a "$LOG"
N="$OUT/nbrb/www.nbrb.by"
get() {   # $1 = url, $2 = target file; an existing non-empty file is kept
    [ -s "$2" ] && return 0
    mkdir -p "$(dirname "$2")"
    if curl -sfL -m 300 -A "$UA" -o "$2" "$1"; then echo "OK   $1" | tee -a "$LOG"; else rm -f "$2"; echo "miss $1" | tee -a "$LOG"; fi
    sleep 0.5
}
S=https://www.nbrb.by/statistics
for y in 2021 2022 2023 2024 2025 2026; do
    get "$S/financialmarkets/avgintratesdyn/${y}_ru.xlsx" "$N/statistics/financialmarkets/avgintratesdyn/${y}_ru.xlsx"
done
get "$S/financialmarkets/avgintratesnewcredit/archive_ru.xlsx" "$N/statistics/financialmarkets/avgintratesnewcredit/archive_ru.xlsx"
for f in AvgIntRates_ru.xlsx AvgIntRates2000-2020_ru.xlsx; do
    get "$S/CreditDepositMarketRates/$f" "$N/statistics/CreditDepositMarketRates/$f"
done
for f in corpsecurmarketbonds_ru.xlsx corpsecurmarketbondsa_ru.xlsx corpsecurmarketdynamics_ru.xlsx; do
    get "$S/financialmarkets/corpsecurmarket/$f" "$N/statistics/financialmarkets/corpsecurmarket/$f"
done
for y in 2021 2022 2023 2024 2025; do
    for m in 1 2 3 4 5 6 7 8 9 10 11 12; do
        get "$S/indicators?year=$y&month=$m" "$N/statistics/indicators/$y-$(printf %02d $m).html"
    done
done

# Monthly statistical bulletins: those listed on the bulletin page (the last two years) and the January and December
# issues of 2021-2024 (a January bulletin contains the whole previous year month by month).
BUL="$N/publications/bulletin"
get "https://www.nbrb.by/publications/bulletin/" "$BUL/index.html"
for b in $(grep -o 'stat_bulletin_[0-9]\{4\}_[0-9]\{2\}\.pdf' "$BUL/index.html" 2>/dev/null | sort -u); do
    pdfinfo "$BUL/$b" >/dev/null 2>&1 || rm -f "$BUL/$b"          # an incomplete earlier download is fetched again
    get "https://www.nbrb.by/publications/bulletin/$b" "$BUL/$b"
done
for y in 2021 2022 2023 2024; do
    for m in 01 12; do
        get "https://www.nbrb.by/publications/bulletin/stat_bulletin_${y}_${m}.pdf" "$BUL/stat_bulletin_${y}_${m}.pdf"
    done
done

# ---------------------------------------------------------------- BCSE
# The security directory is a JavaScript search, so its pages cannot be crawled.  Fetched instead: the pages
# "Средневзвешенная доходность по облигациям", "Рыночная цена ценных бумаг на биржевом рынке", the directory page
# and the site map, with the spreadsheets and PDFs they link to.
echo "--- BCSE (www.bcse.by)" | tee -a "$LOG"
B="$OUT/bcse/www.bcse.by"
for p in for_investors/weighted-average-yield-on-bonds for_investors/securities_market_price \
         for_investors/securities_directory SiteMap/Map; do
    get "https://www.bcse.by/$p" "$B/$p.html"
    for u in $(grep -oE "href=[\"'][^\"']+\.(xlsx?|csv|pdf|zip)[\"']" "$B/$p.html" 2>/dev/null | sed -E "s/^href=[\"']//; s/[\"']$//" | sort -u); do
        case "$u" in http*) url="$u";; /*) url="https://www.bcse.by$u";; *) url="https://www.bcse.by/$u";; esac
        get "$url" "$B/files/$(basename "${u%%\?*}")"
    done
done

# ---------------------------------------------------------------- Rosstat
# Monthly average consumer prices of Russia (sugar, petrol and the rest): the price page of Rosstat and the
# spreadsheets it links to.  Rosstat does not answer from the analysis session; if it does not answer here either,
# this part is skipped and the yearly figures of the Eurasian Economic Commission stay in use.
echo "--- Rosstat (rosstat.gov.ru)" | tee -a "$LOG"
R="$OUT/rosstat"
if reachable "https://rosstat.gov.ru/"; then
    get "https://rosstat.gov.ru/statistics/price" "$R/price.html"
    for u in $(grep -oE "href=[\"'][^\"']+\.(xlsx?|zip)[\"']" "$R/price.html" 2>/dev/null | sed -E "s/^href=[\"']//; s/[\"']$//" | sort -u); do
        case "$u" in http*) url="$u";; /*) url="https://rosstat.gov.ru$u";; *) url="https://rosstat.gov.ru/$u";; esac
        get "$url" "$R/files/$(basename "${u%%\?*}")"
    done
else
    echo "skip rosstat.gov.ru is not reachable from this connection" | tee -a "$LOG"
fi

# ---------------------------------------------------------------- summary
echo "--- summary" | tee -a "$LOG"
for d in nbrb bcse rosstat; do
    [ -d "$OUT/$d" ] || continue
    n_html=$(find "$OUT/$d" -type f -name '*.htm*' | wc -l)
    n_data=$(find "$OUT/$d" -type f \( -iname '*.xls' -o -iname '*.xlsx' -o -iname '*.csv' -o -iname '*.zip' -o -iname '*.pdf' \) | wc -l)
    size=$(du -sh "$OUT/$d" | cut -f1)
    echo "$d: $n_html pages, $n_data data files, $size" | tee -a "$LOG"
done
echo "Done. Files are in $OUT." | tee -a "$LOG"
