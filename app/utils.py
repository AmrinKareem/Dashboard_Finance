import os
from pathlib import Path
import calendar
from typing import Any, Dict, List, Tuple, Iterable, Union, Optional
import json
import pandas as pd
from collections import defaultdict
from sqlalchemy import text
from datetime import datetime


def _load_rows(path: str) -> Tuple[List[Dict[str, Any]], str]:
    """Load the array of rows under the (unknown) top-level period key, or the file is already an array."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data, "unknown"

    if isinstance(data, dict):
        # Try direct period-style keys first (e.g., 'jun-2025' or 'june-2025'); otherwise take first key
        for k in list(data.keys()):
            v = data.get(k)
            if isinstance(v, list):
                return v, k
        # Fallback: no list found -> malformed
        raise ValueError(f"{os.path.basename(path)}: expected an array under a top-level period key.")
    raise ValueError(f"{os.path.basename(path)}: unsupported JSON structure.")


def _group_by_job(rows: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    by_job: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        job = str(r.get("job_no") or r.get("no") or "").strip()
        if job:
            by_job[job].append(r)
    return by_job

def _most_frequent(values: List[Any]) -> Any:
    counts: Dict[str, int] = defaultdict(int)
    for v in values:
        s = "" if v is None else str(v).strip()
        if not s:
            continue
        counts[s] += 1
    if not counts:
        return None
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]

def _to_number(x: Any) -> float:
    try:
        if isinstance(x, str):
            x = x.strip()
        n = float(x)
        if n != n:  # NaN
            return 0.0
        return n
    except Exception:
        return 0.0
def _longest_nonempty(values: Union[pd.Series, Iterable]) -> Optional[str]:
    s = pd.Series(values)  # <-- key change
    s = (
        s.dropna()
         .astype(str)
         .map(str.strip)
         .replace("", pd.NA)
         .dropna()
    )
    if s.empty:
        return None
    return s.loc[s.str.len().idxmax()]

def _first_nonempty(series: pd.Series):
    s = series.dropna()
    if s.empty:
        return None
    # treat "" as empty
    s = s.astype(str).map(lambda x: x.strip()).replace("", pd.NA).dropna()
    return None if s.empty else s.iloc[0]


def safe_str(x: Any) -> str:
    return "" if pd.isna(x) else str(x).strip()


def safe_num(x: Any) -> float:
    if pd.isna(x) or x == "":
        return 0.0
    try:
        return float(x)
    except Exception:
        return 0.0
    
def period_to_label(cperiod: Any) -> str:
    """
    Converts cPeriod like 201901 (YYYYMM) -> 'january-2019'
    """
    if pd.isna(cperiod):
        return "unknown-period"
    s = str(int(cperiod))
    if len(s) != 6:
        return s.lower()
    year = int(s[:4])
    month = int(s[4:6])
    month_name = calendar.month_name[month].lower()
    return f"{month_name}-{year}"



def make_periods(start_ym="201901", end_ym=None):
    """Returns list like ['201901','201902',...], inclusive."""
    if end_ym is None:
        end_ym = datetime.today().strftime("%Y%m")

    start = datetime.strptime(start_ym, "%Y%m")
    end = datetime.strptime(end_ym, "%Y%m")

    periods = []
    y, m = start.year, start.month
    while (y < end.year) or (y == end.year and m <= end.month):
        periods.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            m = 1
            y += 1
    return periods

def get_filter_options(db):

    try:
        periods_sql = text("""
            SELECT DISTINCT cPeriod
            FROM Nibis.dbo.AC_JcPackageDt WITH (NOLOCK)
        """)
        periods = [row[0] for row in db.execute(periods_sql).fetchall()]

        projects_sql = text("""
            SELECT DISTINCT iProjNo
            FROM Nibis.dbo.AC_JcPackageDt WITH (NOLOCK)
        """)
        projects = [row[0] for row in db.execute(projects_sql).fetchall()]
        db.commit()  
    except Exception as e:
        db.rollback()
    return periods, projects




def _normalize_project_groups(project_groups: dict) -> dict[int, str]:
    proj_to_label, seen_groups = {}, set()

    for members in project_groups.values():
        if members is None:
            continue
        mem = members if isinstance(members, (list, tuple, set)) else [members]
        mem_sorted = tuple(sorted({int(x) for x in mem}))
        if not mem_sorted or mem_sorted in seen_groups:
            continue
        seen_groups.add(mem_sorted)

        label = " & ".join(map(str, mem_sorted))
        for p in mem_sorted:
            prev = proj_to_label.get(p)
            if prev is not None and prev != label:
                raise ValueError(f"Project {p} appears in multiple groups: {prev} vs {label}")
            proj_to_label[p] = label

    return proj_to_label


def merge_or_longest(s: pd.Series):
    vals = (
        s.dropna().astype(str).str.strip()
         .replace("", pd.NA).dropna()
         .unique()
    )
    if len(vals) == 0:
        return pd.NA
    if len(vals) == 1:
        return vals[0]
    # multiple distinct → join
    return " & ".join(vals)

def filter_by_project(df, projno):
    # Split 'iProjNo' by ' & ' and check if projno is in the list
    mask = df['iProjNo_group'].apply(lambda x: str(projno) in [p.strip() for p in x.split('&')])
    return df[mask].reset_index(drop=True)

