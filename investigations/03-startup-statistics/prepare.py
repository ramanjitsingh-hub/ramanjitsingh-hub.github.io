"""Turn UCI Online Retail II into a customer-by-month activity matrix.

Source: D. Chen, Online Retail II, UCI Machine Learning Repository (2019),
doi:10.24432/C5CG6D, licence CC BY 4.0. A UK online retailer's transactions,
1 Dec 2009 to 9 Dec 2011.

Output data/raw/activity.npz:
  first[i]   index of customer i's first purchase month (0 = Dec 2009)
  active[i,m] True if customer i bought anything in calendar month m
  country[i] most frequent country
Cancelled invoices (prefix C), rows without a customer id, and non-positive
quantities or prices are dropped.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"


def main():
    pq = RAW / "transactions.parquet"
    if pq.exists():
        t = pd.read_parquet(pq)
    else:
        xl = next(RAW.glob("*.xlsx"))
        sheets = pd.read_excel(xl, sheet_name=None, dtype={"Invoice": str, "StockCode": str})
        t = pd.concat(sheets.values(), ignore_index=True)
        for c in ["Invoice", "StockCode", "Description", "Country"]:
            t[c] = t[c].astype(str)
        t.to_parquet(pq)
    n0 = len(t)
    t = t[t["Customer ID"].notna()]
    t = t[~t["Invoice"].astype(str).str.startswith("C")]
    t = t[(t["Quantity"] > 0) & (t["Price"] > 0)]
    t = t.drop_duplicates()
    t["month"] = (t["InvoiceDate"].dt.year - 2009) * 12 + t["InvoiceDate"].dt.month - 12
    cust = t["Customer ID"].astype(int)
    ids, inv = np.unique(cust.to_numpy(), return_inverse=True)
    M = int(t["month"].max()) + 1
    active = np.zeros((len(ids), M), bool)
    active[inv, t["month"].to_numpy()] = True
    first = active.argmax(1)
    country = t.groupby(cust)["Country"].agg(lambda s: s.mode().iloc[0]).reindex(ids).to_numpy().astype(str)
    np.savez_compressed(RAW / "activity.npz", first=first, active=active, country=country, ids=ids)
    print(f"{n0} rows -> {len(t)} clean purchase lines, {len(ids)} customers, {M} months")
    sizes = np.bincount(first, minlength=M)
    print("cohort sizes by first month:", sizes.tolist())


if __name__ == "__main__":
    main()
