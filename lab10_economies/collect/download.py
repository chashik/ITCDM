"""Download of the data set D9 into data/raw/d9 (only what is missing; a second run retries failures).

Sources reachable from anywhere are fetched here.  The National Bank of Belarus (www.nbrb.by), the Belarusian Currency
and Stock Exchange (www.bcse.by) and Rosstat answer only from a Belarusian connection: their files are fetched by
collect/fetch_blocked.sh into data/raw/d9/manual.  Statistics Lithuania (osp.stat.gov.lt) is behind a browser check;
Lithuanian prices come from Eurostat.  Downloads are untrusted: they are only parsed, never executed.

Run (from code/):  python -m lab10_economies.collect.download
"""
from __future__ import annotations

import time
import urllib.request
from pathlib import Path

from ..paths import RAW

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
HERE = Path(__file__).resolve().parent
EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data"
ECB = "https://data-api.ecb.europa.eu/service/data"
OECD = "https://sdmx.oecd.org/public/rest/data/OECD.SDD.TPS,DSD_PRICES@DF_PRICES_ALL,1.0"
IMF2 = "https://api.imf.org/external/sdmx/2.1/data/IMF.STA"
IMF3 = "https://api.imf.org/external/sdmx/3.0/data/dataflow/IMF.STA"
FAO = "https://fpma.fao.org/giews/v4/global/price_module/api/v1/FpmaSeriePrice"
CBR = "https://www.cbr.ru"
MOEX = "https://iss.moex.com/iss/history/engines/stock/markets/index/securities"
EEC = "https://eec.eaeunion.org/upload/files/dep_stat/econstat"
BELSTAT = "https://www.belstat.gov.by/upload-belstat/upload-belstat-excel/Oficial_statistika"
GUS_YEARS = "&".join(f"year={y}" for y in range(2019, 2026))

FILES = {
    # ------------------------------------------------------------------------------------------------ Belarus
    # Belstat CPI, 2010 = 100 (the file is renamed with every monthly release: IPC-MMYY.xlsx)
    "belstat/IPC-0926.xlsx": f"{BELSTAT}/2026/IPC-0926.xlsx",
    "nbrb_api/refinancingrate.json": "https://api.nbrb.by/refinancingrate",
    "fao/series_9b5697fb-4ad4-4a15-a6e8-23ad23157ab3.json": f"{FAO}/9b5697fb-4ad4-4a15-a6e8-23ad23157ab3/",
    "imf/er_BLR.xml": f"{IMF3}/ER/+/BLR.*.*.M?startPeriod=2020-12&endPeriod=2025-12",
    "imf/mfs_ir_BLR_long.xml": f"{IMF3}/MFS_IR/+/BLR.MFS166_RT_PT_A_PT.M?c%5BTIME_PERIOD%5D=ge:2012-M01",
    # ------------------------------------------------------------------------------------------------ Poland
    "oecd/cpi_pol.csv": f"{OECD}/POL.M.N.CPI.PA._T.N.GY?startPeriod=2020-01&endPeriod=2025-12&format=csvfilewithlabels",
    "oecd/cpi_pol_IX__Z.csv": f"{OECD}/POL.M.N.CPI.IX._T.N._Z?startPeriod=2005-01&format=csvfilewithlabels",
    "eurostat/prc_hicp_manr_PL.json": f"{EUROSTAT}/prc_hicp_manr?geo=PL&coicop=CP00&sinceTimePeriod=2020-01",
    "eurostat/prc_hicp_mmor_PL.json": f"{EUROSTAT}/prc_hicp_mmor?geo=PL&coicop=CP00&sinceTimePeriod=2020-01",
    "eurostat/irt_lt_mcby_m_PL.json": f"{EUROSTAT}/irt_lt_mcby_m?geo=PL&sinceTimePeriod=2020-01",
    **{f"eurostat/midx_{c}.json": f"{EUROSTAT}/prc_hicp_midx?geo=PL&coicop={c}&unit=I15&sinceTimePeriod=2020-01"
       for c in ("CP01112", "CP01141", "CP01181", "CP07222")},
    **{f"gus/annual_{v}.json": f"https://bdl.stat.gov.pl/api/v1/data/by-variable/{v}?format=json&unit-level=0&{GUS_YEARS}"
       for v in ("4953", "1749185", "4975", "4994", "196398")},     # annual average prices: flour, milk, sugar, petrol
    "nbp/stopy_archiwum.xml": "https://static.nbp.pl/dane/stopy/stopy_procentowe_archiwum.xml",
    "nbp/mir/stopy_proc_pl_srdW.xlsx": "https://static.nbp.pl/dane/statystyka/inne/stopy_proc_pl_srdW.xlsx",
    "ecb/mir_pl.csv": f"{ECB}/MIR/M.PL.B.A2C.A.R.A.2250.PLN.N?startPeriod=2020-12&format=csvdata",
    "bis/cbpol.csv": "https://stats.bis.org/api/v2/data/dataflow/BIS/WS_CBPOL/1.0/M.PL+BY?startPeriod=2020-01&format=csv",
    "imf/er_POL.xml": f"{IMF3}/ER/+/POL.*.*.M?startPeriod=2020-12&endPeriod=2025-12",
    # Belarus and Poland together (IMF)
    "imf/cpi.json": f"{IMF2},CPI/BLR+POL.CPI._T.YOY_PCH_PA_PT.M?startPeriod=2021-01&endPeriod=2025-12",
    "imf/mfs_ir.xml": f"{IMF2},MFS_IR/BLR+POL..M?startPeriod=2021-01&endPeriod=2025-12",
    # EU Weekly Oil Bulletin (Poland and Lithuania)
    "ec_oil/history.xlsx": "https://energy.ec.europa.eu/document/download/906e60ca-8b6a-44e7-8589-652854d2fd3f_en"
                           "?filename=Weekly_Oil_Bulletin_Prices_History_maticni_4web.xlsx",
    # ------------------------------------------------------------------------------------------------ Russia
    "cbr/keyrate.html": f"{CBR}/hd_base/KeyRate/?UniDbQuery.Posted=True&UniDbQuery.From=01.12.2020&UniDbQuery.To=31.12.2025",
    "cbr/infl.html": f"{CBR}/hd_base/infl/?UniDbQuery.Posted=True&UniDbQuery.From=01.12.2020&UniDbQuery.To=31.12.2025",
    "cbr/zcyc.html": f"{CBR}/hd_base/zcyc_params/?UniDbQuery.Posted=True&UniDbQuery.From=01.12.2020&UniDbQuery.To=31.12.2025",
    "cbr/02_13_Rates_mortgage.xlsx": f"{CBR}/vfs/statistics/BankSector/Mortgage/02_13_Rates_mortgage.xlsx",
    "cbr/02_08_Rates_housing.xlsx": f"{CBR}/vfs/statistics/BankSector/Mortgage/02_08_Rates_housing.xlsx",
    "cbr/usd_rub.xml": f"{CBR}/scripts/XML_dynamic.asp?date_req1=01/12/2020&date_req2=31/12/2025&VAL_NM_RQ=R01235",
    "eec/CPI_1.xls": f"{EEC}/Series_CPI/CPI_1.xls",
    "eec/CPI_3.xls": f"{EEC}/Series_CPI/CPI_3.xls",
    "eec/InfoCPI202112.pdf": f"{EEC}/InfoBlocks/Info_CPI/InfoCPI202112.pdf",
    "eec/InfoCPI202212.pdf": f"{EEC}/InfoBlocks/Info_CPI/InfoCPI202212.pdf",
    "eec/InfoCPI202312.pdf": f"{EEC}/InfoBlocks/Info_CPI/InfoCPI202312.pdf",
    "eec/InfoCPI202412.pdf": f"{EEC}/InfoBlocks/Info_CPI/%D0%94%D0%B5%D0%BA%D0%B0%D0%B1%D1%80%D1%8C_2024.pdf",
    "eec/InfoCPI202512.pdf": f"{EEC}/InfoBlocks/Info_CPI/InfoCPI2025_1.pdf",
    "fao/series_RUS_flour.json": f"{FAO}/51e217ea-0635-4028-8e27-5c11c36508c2/",
    "fao/series_RUS_milk.json": f"{FAO}/2daade04-cf07-4d68-8333-7ea6804774bb/",
    "imf/er_RUS.xml": f"{IMF3}/ER/+/RUS.*.*.M?startPeriod=2020-12&endPeriod=2025-12",
    "imf/mfs_ir_RUS_long.xml": f"{IMF3}/MFS_IR/+/RUS.MFS166_RT_PT_A_PT.M?c%5BTIME_PERIOD%5D=ge:2012-M01",
    # ------------------------------------------------------------------------------------------------ Lithuania
    "eurostat/prc_hicp_manr_LT.json": f"{EUROSTAT}/prc_hicp_manr?geo=LT&coicop=CP00&sinceTimePeriod=2020-01",
    "eurostat/prc_hicp_mmor_LT.json": f"{EUROSTAT}/prc_hicp_mmor?geo=LT&coicop=CP00&sinceTimePeriod=2020-01",
    **{f"eurostat/midx_{c}_LT.json": f"{EUROSTAT}/prc_hicp_midx?geo=LT&coicop={c}&unit=I15&sinceTimePeriod=2020-12"
       for c in ("CP01112", "CP01141", "CP01181", "CP07222")},
    "eurostat/irt_lt_mcby_m_LT.json": f"{EUROSTAT}/irt_lt_mcby_m?geo=LT&sinceTimePeriod=2020-01",
    # euro-area 10-year yield: the market yield for the price of Lithuanian bonds (the Lithuanian series is that of
    # the last issue and does not move)
    "eurostat/irt_lt_mcby_m_EA.json": f"{EUROSTAT}/irt_lt_mcby_m?geo=EA&sinceTimePeriod=2020-01",
    "eurostat/prc_hicp_manr_LT_2011.json": f"{EUROSTAT}/prc_hicp_manr?geo=LT&coicop=CP00&sinceTimePeriod=2011-01",
    "eurostat/prc_hicp_manr_EA.json": f"{EUROSTAT}/prc_hicp_manr?geo=EA&coicop=CP00&sinceTimePeriod=2020-01",
    "oecd/cpi_ltu.csv": f"{OECD}/LTU.M.N.CPI.PA._T.N.GY?startPeriod=2020-01&endPeriod=2025-12&format=csvfile",
    "ecb/dfr.csv": f"{ECB}/FM/D.U2.EUR.4F.KR.DFR.LEV?format=csvdata&startPeriod=2020-12-01&endPeriod=2025-12-31",
    "ecb/mro.csv": f"{ECB}/FM/D.U2.EUR.4F.KR.MRR_FR.LEV?format=csvdata&startPeriod=2020-12-01&endPeriod=2025-12-31",
    "ecb/mir_lt.csv": f"{ECB}/MIR/M.LT.B.A2C.A.R.A.2250.EUR.N?format=csvdata&startPeriod=2020-12&endPeriod=2025-12",
    "ecb/exr_usd.csv": f"{ECB}/EXR/M.USD.EUR.SP00.A?format=csvdata&startPeriod=2020-12&endPeriod=2025-12",
    "ecb/dfr_long.csv": f"{ECB}/FM/D.U2.EUR.4F.KR.DFR.LEV?format=csvdata&startPeriod=2011-12-01&endPeriod=2025-12-31",
    "ecb/exr_usd_long.csv": f"{ECB}/EXR/M.USD.EUR.SP00.A?format=csvdata&startPeriod=2011-01&endPeriod=2025-12",
}


def belstat_files() -> dict:
    """Monthly average retail prices of Belstat (one workbook per month; irregular names, so the links are listed)."""
    out = {}
    for line in (HERE / "belstat_links.tsv").read_text().splitlines():
        month, url = line.split("\t")
        out[f"belstat/goods/goods_{month}{Path(url).suffix}"] = url
    return out


def minfin_files() -> dict:
    """Ministry of Finance of Belarus: register of bonds in circulation on the first day of each month."""
    months = [(m, y) for y in range(21, 26) for m in range(1, 13)] + [(1, 26)]
    return {f"minfin_by/register/obl_obr_{m:02d}{y}.xlsx":
            f"https://www.minfin.gov.by/upload/depcen/obligacii/obl_obr_{m:02d}{y}.xlsx" for m, y in months}


def moex_files() -> dict:
    """Moscow Exchange: total-return index of federal loan bonds (RGBITR), daily closes, 100 days per page."""
    return {f"moex/rgbitr_{s:04d}.csv": f"{MOEX}/RGBITR.csv?from=2020-12-01&till=2025-12-31&start={s}"
            "&iss.only=history&history.columns=TRADEDATE,CLOSE" for s in range(0, 1300, 100)}


def get(rel: str, url: str) -> str:
    path = RAW / rel
    if path.exists() and path.stat().st_size > 0:
        return "have"
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": UA}
    if "api.imf.org" in url:                                 # SDMX-ML (the IMF answers with JSON otherwise)
        headers["Accept"] = "application/xml"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=180) as r:
            path.write_bytes(r.read())
        time.sleep(0.3)
        return "got"
    except Exception as e:                                   # report and go on; a second run retries
        return f"FAIL ({e})"


def main():
    jobs = {**FILES, **belstat_files(), **minfin_files(), **moex_files()}
    status = {rel: get(rel, url) for rel, url in jobs.items()}
    for rel, s in status.items():
        if s != "have":
            print(f"{s:5s} {rel}")
    from .pl_retail_bonds import download as obligacje       # pages of the Polish savings bonds (one per issue)
    obligacje()
    fails = [r for r, s in status.items() if s.startswith("FAIL")]
    print(f"{len(jobs)} files: {sum(s == 'have' for s in status.values())} present, "
          f"{sum(s == 'got' for s in status.values())} downloaded, {len(fails)} failed")
    if not (RAW / "manual" / "nbrb").exists():
        print("NBRB, BCSE and Rosstat files are missing: run lab10_economies/collect/fetch_blocked.sh "
              "from a Belarusian connection")


if __name__ == "__main__":
    main()
