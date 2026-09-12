"""Causal, deterministic PV-availability forecasts for controlled events."""

from __future__ import annotations

import numpy as np
import pandas as pd

FORECAST_METHOD = "previous_day_clearsky_index_persistence"
REVISION_METHOD = "latest_observation_clearsky_index_persistence"
_CLEAR_SKY_EPSILON = 1e-6
_CLEAR_SKY_INDEX_MAX = 1.25


def day_ahead_pv_forecast(
    index: pd.DatetimeIndex,
    realized_pv_kw: pd.Series,
    clear_sky_ghi: pd.Series,
    ac_capacity_kw: float,
) -> pd.DataFrame:
    """Create a day-ahead PV forecast without reading values after its issue time."""
    if len(index) != len(realized_pv_kw) or len(index) != len(clear_sky_ghi):
        raise ValueError("Forecast index and source series must have equal lengths.")
    if index.tz is None or not index.is_monotonic_increasing:
        raise ValueError("Forecast timestamps must be monotonic and timezone-aware.")
    interval = _interval_steps(index, pd.Timedelta(days=1))
    realized = realized_pv_kw.to_numpy(dtype=float)
    clear = clear_sky_ghi.to_numpy(dtype=float)
    forecast = np.zeros(len(index), dtype=float)
    for position in range(interval, len(index)):
        prior = position - interval
        if clear[prior] > _CLEAR_SKY_EPSILON:
            clear_sky_index = np.clip(realized[prior] / ac_capacity_kw * 1000.0 / clear[prior], 0.0, _CLEAR_SKY_INDEX_MAX)
            forecast[position] = min(ac_capacity_kw, ac_capacity_kw * clear[position] / 1000.0 * clear_sky_index)
    issue_time = pd.DatetimeIndex(
        [index[position - interval] if position >= interval else index[position] - pd.Timedelta(days=1) for position in range(len(index))]
    )
    return pd.DataFrame(
        {
            "forecast_pv_kw": forecast,
            "forecast_issue_time": issue_time,
            "forecast_method": FORECAST_METHOD,
        },
        index=index,
    )


def revised_pv_forecast(
    frame: pd.DataFrame, activation_step: int, blend_weight: float, ac_capacity_kw: float
) -> pd.Series:
    """Update pending PV availability from the last observation before activation."""
    required = {"forecast_pv_kw", "pv_ac_kw", "clear_sky_ghi"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Forecast revision lacks columns: {sorted(missing)}")
    if not 1 <= activation_step < len(frame):
        raise ValueError("Forecast revision requires an observation before activation.")
    if not 0.0 <= blend_weight <= 1.0:
        raise ValueError("Forecast revision blend weight must be in [0, 1].")
    revised = frame["forecast_pv_kw"].astype(float).copy()
    observed_position = activation_step - 1
    observed_pv = float(frame["pv_ac_kw"].iloc[observed_position])
    observed_clear = float(frame["clear_sky_ghi"].iloc[observed_position])
    clear_sky_index = 0.0 if observed_clear <= _CLEAR_SKY_EPSILON else np.clip(
        observed_pv / ac_capacity_kw * 1000.0 / observed_clear, 0.0, _CLEAR_SKY_INDEX_MAX
    )
    clear = frame["clear_sky_ghi"].to_numpy(dtype=float)
    updated = np.minimum(ac_capacity_kw, ac_capacity_kw * clear / 1000.0 * clear_sky_index)
    revised_tail = pd.Series(
        (1.0 - blend_weight) * revised.iloc[activation_step:].to_numpy(dtype=float)
        + blend_weight * updated[activation_step:],
        index=revised.index[activation_step:],
    )
    return pd.concat([revised.iloc[:activation_step], revised_tail]).clip(lower=0.0, upper=ac_capacity_kw)


def forecast_is_causal(index: pd.DatetimeIndex, issue_time: pd.Series) -> bool:
    """Check the observable temporal boundary recorded in a processed table."""
    issues = pd.DatetimeIndex(pd.to_datetime(issue_time, utc=True))
    timestamps = pd.DatetimeIndex(index).tz_convert("UTC")
    return bool(len(issues) == len(timestamps) and (issues < timestamps).all())


def _interval_steps(index: pd.DatetimeIndex, duration: pd.Timedelta) -> int:
    if len(index) < 2:
        raise ValueError("At least two timestamps are required for a forecast.")
    step = index[1] - index[0]
    if step <= pd.Timedelta(0) or duration % step != pd.Timedelta(0):
        raise ValueError("Forecast resolution must evenly divide one day.")
    return int(duration // step)
