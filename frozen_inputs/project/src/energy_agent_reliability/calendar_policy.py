"""Fixed UTC calendar rules for claim-bearing data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

EXCLUDED_DATES = ("2024-02-29",)


def source_index(year: int, minutes: int) -> pd.DatetimeIndex:
    """Return the complete raw UTC source-year grid before exclusions."""
    return pd.date_range(
        pd.Timestamp(year, 1, 1, tz="UTC"),
        pd.Timestamp(year + 1, 1, 1, tz="UTC"),
        inclusive="left",
        freq=f"{minutes}min",
    )


def claim_index(year: int, minutes: int, excluded_dates: Iterable[str] = EXCLUDED_DATES) -> pd.DatetimeIndex:
    """Return the canonical UTC grid after the frozen calendar exclusion."""
    expected = source_index(year, minutes)
    excluded = pd.DatetimeIndex(pd.to_datetime(list(excluded_dates), utc=True)).normalize()
    return expected[~expected.normalize().isin(excluded)]


def drop_excluded_dates(frame: pd.DataFrame, excluded_dates: Iterable[str] = EXCLUDED_DATES) -> pd.DataFrame:
    """Drop frozen excluded UTC dates from a timestamp-indexed frame."""
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Calendar normalization requires a DatetimeIndex.")
    index = pd.DatetimeIndex(frame.index)
    if index.tz is None:
        raise ValueError("Claim-bearing timestamps must be timezone-aware UTC.")
    utc_index = index.tz_convert("UTC")
    excluded = pd.DatetimeIndex(pd.to_datetime(list(excluded_dates), utc=True)).normalize()
    return frame.loc[~utc_index.normalize().isin(excluded)].copy()


def has_excluded_date(index: pd.DatetimeIndex, excluded_dates: Iterable[str] = EXCLUDED_DATES) -> bool:
    """Report whether a UTC index retains an excluded date."""
    if index.tz is None:
        return True
    excluded = pd.DatetimeIndex(pd.to_datetime(list(excluded_dates), utc=True)).normalize()
    return bool(index.tz_convert("UTC").normalize().isin(excluded).any())
