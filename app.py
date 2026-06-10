import math

import numpy as np
import pandas as pd
import streamlit as st

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    STL = None


st.set_page_config(
    page_title="Time Series Decomposition Lab",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


def make_series(
    periods: int,
    trend_strength: float,
    seasonal_strength: float,
    seasonal_period: int,
    noise_strength: float,
    shock_size: float,
    seed: int,
) -> pd.DataFrame:
    """Create a synthetic additive time series and its known components."""
    rng = np.random.default_rng(seed)
    t = np.arange(periods)
    dates = pd.date_range("2021-01-01", periods=periods, freq="W")

    level = np.full(periods, 100.0)
    trend = trend_strength * (t / max(periods - 1, 1)) * 80
    seasonality = seasonal_strength * 18 * np.sin(2 * math.pi * t / seasonal_period)
    seasonality += seasonal_strength * 6 * np.cos(4 * math.pi * t / seasonal_period)
    residual = rng.normal(0, noise_strength * 8, periods)

    shock = np.zeros(periods)
    if shock_size:
        center = int(periods * 0.68)
        width = max(2, periods // 28)
        shock = shock_size * 25 * np.exp(-0.5 * ((t - center) / width) ** 2)

    observed = level + trend + seasonality + shock + residual

    return pd.DataFrame(
        {
            "date": dates,
            "observed": observed,
            "level": level,
            "trend": trend,
            "seasonality": seasonality,
            "event": shock,
            "residual": residual,
        }
    )


def make_multiplicative_series(
    periods: int,
    trend_strength: float,
    seasonal_strength: float,
    seasonal_period: int,
    noise_strength: float,
    shock_size: float,
    seed: int,
) -> pd.DataFrame:
    """Create a strictly positive series from multiplicative factors."""
    rng = np.random.default_rng(seed + 1000)
    t = np.arange(periods)
    dates = pd.date_range("2021-01-01", periods=periods, freq="W")

    level = np.full(periods, 100.0)
    trend_factor = 1.0 + trend_strength * 0.45 * (t / max(periods - 1, 1))
    seasonal_factor = 1.0 + seasonal_strength * (
        0.12 * np.sin(2 * math.pi * t / seasonal_period)
        + 0.04 * np.cos(4 * math.pi * t / seasonal_period)
    )

    event_factor = np.ones(periods)
    if shock_size:
        center = int(periods * 0.68)
        width = max(2, periods // 28)
        event_factor = 1.0 + shock_size * 0.18 * np.exp(
            -0.5 * ((t - center) / width) ** 2
        )

    residual_factor = np.exp(rng.normal(0, noise_strength * 0.04, periods))
    observed = level * trend_factor * seasonal_factor * event_factor * residual_factor

    return pd.DataFrame(
        {
            "date": dates,
            "observed": observed,
            "level": level,
            "trend_factor": trend_factor,
            "seasonal_factor": seasonal_factor,
            "event_factor": event_factor,
            "residual_factor": residual_factor,
        }
    )


def normalize_component(series: pd.Series) -> pd.Series:
    spread = series.max() - series.min()
    if spread == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / spread


def reconstruction_error(df: pd.DataFrame) -> float:
    reconstructed = (
        df["level"] + df["trend"] + df["seasonality"] + df["event"] + df["residual"]
    )
    return float(np.abs(df["observed"] - reconstructed).max())


def make_trend_window(window: int) -> int:
    if window < 3:
        return 3
    return window + 1 if window % 2 == 0 else window


def make_odd_window(window: int, minimum: int = 3) -> int:
    """Return an odd smoothing window that satisfies STL-style constraints."""
    adjusted = max(int(window), minimum)
    return adjusted + 1 if adjusted % 2 == 0 else adjusted


def estimate_trend_methods(
    data: pd.DataFrame,
    smoothing_window: int,
    ewm_span: int,
    regression_history: int,
) -> pd.DataFrame:
    """Compare common trend-cycle estimates against the known synthetic trend."""
    work = data[["date", "observed", "level", "trend"]].copy()
    work["known_trend_cycle"] = work["level"] + work["trend"]

    smoothing_window = min(make_trend_window(smoothing_window), len(work))
    if smoothing_window % 2 == 0:
        smoothing_window -= 1
    smoothing_window = max(3, smoothing_window)
    work["centered_moving_average"] = (
        work["observed"]
        .rolling(
            window=smoothing_window,
            center=True,
            min_periods=max(2, smoothing_window // 3),
        )
        .mean()
        .interpolate(limit_direction="both")
    )

    ewm_span = max(2, min(int(ewm_span), len(work)))
    work["exponential_smoothing"] = work["observed"].ewm(
        span=ewm_span,
        adjust=False,
    ).mean()

    regression_history = min(max(2, int(regression_history)), len(work))
    x_all = np.arange(len(work))
    fit_start = len(work) - regression_history
    slope, intercept = np.polyfit(
        x_all[fit_start:],
        work["observed"].iloc[fit_start:].to_numpy(),
        1,
    )
    work["linear_regression"] = intercept + slope * x_all
    return work


def estimate_seasonality_methods(
    data: pd.DataFrame,
    period: int,
    trend_window: int,
    shrinkage: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare common seasonal estimates for a known-period additive series."""
    work = data[["date", "observed", "seasonality"]].copy()
    period = min(max(2, int(period)), len(work))
    phase = pd.Series(np.arange(len(work)) % period, index=work.index, name="phase")

    trend_window = min(make_trend_window(trend_window), len(work))
    if trend_window % 2 == 0:
        trend_window -= 1
    trend_window = max(3, trend_window)
    trend_cycle = (
        work["observed"]
        .rolling(
            window=trend_window,
            center=True,
            min_periods=max(2, trend_window // 3),
        )
        .mean()
        .interpolate(limit_direction="both")
    )

    detrended = work["observed"] - trend_cycle
    phase_average = detrended.groupby(phase).mean().reindex(range(period), fill_value=0.0)
    phase_average = (phase_average - phase_average.mean()) * shrinkage

    raw_average = (work["observed"] - work["observed"].mean()).groupby(phase).mean()
    raw_average = raw_average.reindex(range(period), fill_value=0.0)
    raw_average = (raw_average - raw_average.mean()) * shrinkage

    latest_cycle = pd.DataFrame(
        {
            "phase": phase.tail(period),
            "value": detrended.tail(period),
        }
    )
    latest_template = latest_cycle.groupby("phase")["value"].mean()
    latest_template = latest_template.reindex(range(period), fill_value=0.0)
    latest_template = (latest_template - latest_template.mean()) * shrinkage

    known_pattern = work["seasonality"].groupby(phase).mean()
    known_pattern = known_pattern.reindex(range(period), fill_value=0.0)
    known_pattern = known_pattern - known_pattern.mean()

    work["phase"] = phase
    work["known_seasonality"] = work["seasonality"]
    work["detrended_phase_average"] = phase.map(phase_average).astype(float)
    work["raw_phase_average"] = phase.map(raw_average).astype(float)
    work["latest_cycle_template"] = phase.map(latest_template).astype(float)

    pattern = pd.DataFrame(
        {
            "phase": range(period),
            "known_pattern": known_pattern.to_numpy(),
            "detrended_phase_average": phase_average.to_numpy(),
            "raw_phase_average": raw_average.to_numpy(),
            "latest_cycle_template": latest_template.to_numpy(),
        }
    )
    return work, pattern


def component_strength(signal: pd.Series, residual: pd.Series) -> float:
    """Measure how much variance a component explains after residual removal."""
    total_variance = float((signal + residual).var())
    residual_variance = float(residual.var())
    if total_variance <= 0:
        return 0.0
    return max(0.0, min(1.0, 1.0 - residual_variance / total_variance))


def residual_outlier_count(residual: pd.Series, threshold: float = 3.0) -> int:
    residual_std = float(residual.std())
    if residual_std == 0 or np.isnan(residual_std):
        return 0
    return int((residual.abs() > threshold * residual_std).sum())


def fit_stl_components(
    data: pd.DataFrame,
    period: int,
    seasonal_window: int,
    trend_window: int,
    low_pass_window: int,
    robust: bool,
) -> pd.DataFrame:
    """Fit STL and return observed, trend-cycle, seasonality, and residuals."""
    work = data[["date", "value"]].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"]).sort_values("date").reset_index(drop=True)

    result = STL(
        work["value"],
        period=period,
        seasonal=seasonal_window,
        trend=trend_window,
        low_pass=low_pass_window,
        robust=robust,
    ).fit()

    work["trend_cycle"] = result.trend
    work["seasonality"] = result.seasonal
    work["residual"] = result.resid
    work["fitted"] = work["trend_cycle"] + work["seasonality"]
    return work


def estimate_additive_components(
    data: pd.DataFrame,
    seasonal_period: int,
    trend_window: int,
    level_reference: str,
    seasonal_shrinkage: float,
) -> tuple[pd.DataFrame, pd.Series, float]:
    """Estimate components from observed values using simple additive steps."""
    work = data[["date", "value"]].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"]).reset_index(drop=True)

    trend_window = min(make_trend_window(trend_window), len(work))
    if trend_window % 2 == 0:
        trend_window -= 1
    trend_window = max(3, trend_window)

    trend_level = (
        work["value"]
        .rolling(window=trend_window, center=True, min_periods=max(2, trend_window // 3))
        .mean()
        .interpolate(limit_direction="both")
    )

    if level_reference == "First cycle":
        level_slice = trend_level.head(seasonal_period)
    elif level_reference == "Last cycle":
        level_slice = trend_level.tail(seasonal_period)
    else:
        level_slice = trend_level

    level_value = float(level_slice.mean())
    level = pd.Series(np.full(len(work), level_value), index=work.index)
    trend = trend_level - level

    detrended = work["value"] - level - trend
    phase = pd.Series(np.arange(len(work)) % seasonal_period, name="phase")
    seasonal_pattern = detrended.groupby(phase).mean()
    seasonal_pattern = seasonal_pattern.reindex(range(seasonal_period), fill_value=0.0)
    seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()
    seasonal_pattern = seasonal_pattern * seasonal_shrinkage

    seasonality = phase.map(seasonal_pattern).astype(float)
    residual = work["value"] - level - trend - seasonality

    work["level"] = level
    work["trend"] = trend
    work["trend_level"] = trend_level
    work["seasonality"] = seasonality
    work["residual"] = residual
    work["fitted"] = work["level"] + work["trend"] + work["seasonality"]
    return work, seasonal_pattern, level_value


def make_forecast(
    components: pd.DataFrame,
    seasonal_pattern: pd.Series,
    horizon: int,
    trend_fit_points: int,
    trend_damping: float,
    seasonal_scale: float,
    residual_adjustment: float,
) -> pd.DataFrame:
    """Forecast by extending trend, repeating seasonality, and applying residual bias."""
    history_len = len(components)
    fit_points = min(max(2, trend_fit_points), history_len)
    x_fit = np.arange(fit_points)
    y_fit = components["trend_level"].tail(fit_points).to_numpy()
    slope, _intercept = np.polyfit(x_fit, y_fit, 1)
    slope = slope * trend_damping

    last_date = components["date"].iloc[-1]
    inferred_step = components["date"].diff().dropna().median()
    if pd.isna(inferred_step):
        inferred_step = pd.Timedelta(days=7)

    future_steps = np.arange(1, horizon + 1)
    future_dates = [last_date + inferred_step * int(step) for step in future_steps]
    future_trend_level = components["trend_level"].iloc[-1] + slope * future_steps
    future_level = np.full(horizon, components["level"].iloc[-1])
    future_trend = future_trend_level - future_level
    future_phase = (np.arange(history_len, history_len + horizon) % len(seasonal_pattern))
    future_seasonality = seasonal_pattern.iloc[future_phase].to_numpy() * seasonal_scale
    recent_residual = components["residual"].tail(fit_points).mean() * residual_adjustment
    forecast = future_level + future_trend + future_seasonality + recent_residual

    residual_std = components["residual"].std()
    return pd.DataFrame(
        {
            "date": future_dates,
            "level": future_level,
            "trend": future_trend,
            "trend_level": future_trend_level,
            "seasonality": future_seasonality,
            "residual_adjustment": np.full(horizon, recent_residual),
            "forecast": forecast,
            "lower_band": forecast - 1.96 * residual_std,
            "upper_band": forecast + 1.96 * residual_std,
        }
    )


def estimate_regression_components(
    data: pd.DataFrame,
    fit_history: int,
    residual_centering: str,
) -> tuple[pd.DataFrame, float, float, int]:
    """Fit level plus linear trend with ordinary least squares."""
    work = data[["date", "value"]].copy()
    work["value"] = pd.to_numeric(work["value"], errors="coerce")
    work = work.dropna(subset=["value"]).reset_index(drop=True)

    history_len = len(work)
    fit_points = min(max(2, fit_history), history_len)
    fit_start = history_len - fit_points
    x = np.arange(history_len) - fit_start
    fit_mask = np.arange(history_len) >= fit_start

    slope, intercept = np.polyfit(x[fit_mask], work.loc[fit_mask, "value"], 1)
    regression_line = intercept + slope * x

    level = pd.Series(np.full(history_len, intercept), index=work.index)
    trend = pd.Series(slope * x, index=work.index)
    residual = work["value"] - level - trend
    if residual_centering == "Fit window":
        residual = residual - residual[fit_mask].mean()
    elif residual_centering == "Full history":
        residual = residual - residual.mean()

    work["x"] = x
    work["level"] = level
    work["trend"] = trend
    work["regression_line"] = regression_line
    work["residual"] = residual
    work["fitted"] = work["level"] + work["trend"]
    work["used_for_fit"] = fit_mask
    return work, float(slope), float(intercept), fit_start


def make_regression_forecast(
    components: pd.DataFrame,
    slope: float,
    intercept: float,
    fit_start: int,
    horizon: int,
    trend_damping: float,
    residual_adjustment: float,
    residual_history: int,
) -> pd.DataFrame:
    """Forecast by extending a fitted regression line."""
    history_len = len(components)
    last_date = components["date"].iloc[-1]
    inferred_step = components["date"].diff().dropna().median()
    if pd.isna(inferred_step):
        inferred_step = pd.Timedelta(days=7)

    future_index = np.arange(history_len, history_len + horizon)
    future_x = future_index - fit_start
    future_steps = np.arange(1, horizon + 1)
    future_dates = [
        last_date + inferred_step * int(step)
        for step in future_steps
    ]
    damped_slope = slope * trend_damping
    last_x = history_len - 1 - fit_start
    last_regression_line = intercept + slope * last_x
    future_regression_line = last_regression_line + damped_slope * future_steps
    future_level = np.full(horizon, intercept)
    future_trend = future_regression_line - future_level
    residual_points = min(max(1, residual_history), history_len)
    recent_residual = (
        components["residual"].tail(residual_points).mean() * residual_adjustment
    )
    forecast = future_level + future_trend + recent_residual

    residual_std = components["residual"].std()
    return pd.DataFrame(
        {
            "date": future_dates,
            "x": future_x,
            "level": future_level,
            "trend": future_trend,
            "residual_adjustment": np.full(horizon, recent_residual),
            "forecast": forecast,
            "lower_band": forecast - 1.96 * residual_std,
            "upper_band": forecast + 1.96 * residual_std,
        }
    )


def dataframe_to_csv(data: pd.DataFrame) -> bytes:
    """Encode a dataframe for Streamlit download buttons."""
    return data.to_csv(index=False).encode("utf-8")


def clean_uploaded_time_series(
    uploaded: pd.DataFrame,
    date_choice: str,
    value_choice: str,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return a sorted date/value series plus quality notes for the UI."""
    original_rows = len(uploaded)
    values = pd.to_numeric(uploaded[value_choice], errors="coerce")
    invalid_value_rows = int(values.isna().sum())

    using_row_number_dates = date_choice == "Use row number"
    if using_row_number_dates:
        dates = pd.date_range("2021-01-01", periods=original_rows, freq="D")
        invalid_date_rows = 0
    else:
        dates = pd.to_datetime(uploaded[date_choice], errors="coerce")
        invalid_date_rows = int(dates.isna().sum())

    work = pd.DataFrame({"date": dates, "value": values})
    cleaned = work.dropna(subset=["date", "value"]).copy()
    dropped_rows = original_rows - len(cleaned)

    duplicate_date_rows = int(cleaned.duplicated("date").sum())
    if duplicate_date_rows:
        cleaned = cleaned.groupby("date", as_index=False)["value"].mean()

    cleaned = cleaned.sort_values("date").reset_index(drop=True)
    inferred_step = None
    irregular_step_count = 0
    if len(cleaned) > 1:
        steps = cleaned["date"].diff().dropna()
        inferred_step = steps.median()
        irregular_step_count = int((steps != inferred_step).sum())

    return cleaned, {
        "original_rows": original_rows,
        "cleaned_rows": len(cleaned),
        "dropped_rows": dropped_rows,
        "invalid_value_rows": invalid_value_rows,
        "invalid_date_rows": invalid_date_rows,
        "duplicate_date_rows": duplicate_date_rows,
        "using_row_number_dates": using_row_number_dates,
        "inferred_step": inferred_step,
        "irregular_step_count": irregular_step_count,
    }


def render_upload_quality_notes(summary: dict[str, object]) -> None:
    """Show compact validation feedback after a CSV upload is parsed."""
    st.caption(
        f"Using {summary['cleaned_rows']} cleaned rows from "
        f"{summary['original_rows']} uploaded rows."
    )
    if summary["dropped_rows"]:
        st.warning(
            "Dropped "
            f"{summary['dropped_rows']} rows with missing or invalid dates/values."
        )
    if summary["invalid_value_rows"]:
        st.caption(f"Invalid numeric values found: {summary['invalid_value_rows']}")
    if summary["invalid_date_rows"]:
        st.caption(f"Unparseable dates found: {summary['invalid_date_rows']}")
    if summary["duplicate_date_rows"]:
        st.info(
            "Duplicate dates were aggregated by averaging their numeric values."
        )
    if summary["irregular_step_count"]:
        st.warning(
            "The cleaned dates are not evenly spaced. Forecast horizons use the "
            "median observed step, so check the result carefully."
        )
    elif summary["inferred_step"] is not None:
        st.caption(f"Inferred time step: {summary['inferred_step']}")


def render_time_series_uploader(
    uploader_label: str,
    empty_message: str,
    key_prefix: str,
    date_label: str,
    value_label: str,
) -> pd.DataFrame | None:
    """Render a reusable CSV uploader and return cleaned date/value data."""
    uploaded_file = st.file_uploader(
        uploader_label,
        type="csv",
        key=f"{key_prefix}_upload",
    )
    if uploaded_file is None:
        st.info(empty_message)
        return None

    try:
        uploaded = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return None

    if uploaded.empty or not len(uploaded.columns):
        st.error("The uploaded CSV does not contain any rows or columns.")
        return None

    st.dataframe(uploaded.head(8), width="stretch")

    column_names = list(uploaded.columns)
    date_options = ["Use row number"] + column_names
    date_choice = st.selectbox(
        date_label,
        date_options,
        key=f"{key_prefix}_date_column",
    )
    value_choice = st.selectbox(
        value_label,
        column_names,
        key=f"{key_prefix}_value_column",
    )

    cleaned, summary = clean_uploaded_time_series(
        uploaded,
        date_choice=date_choice,
        value_choice=value_choice,
    )
    render_upload_quality_notes(summary)

    if cleaned.empty:
        st.error("No usable rows remain after cleaning the selected columns.")
        return None

    st.download_button(
        "Download cleaned upload CSV",
        dataframe_to_csv(cleaned),
        file_name=f"{key_prefix}_cleaned_time_series.csv",
        mime="text/csv",
        key=f"{key_prefix}_cleaned_download",
    )

    return cleaned


with st.sidebar:
    st.header("Series Controls")
    st.caption(
        "These controls create the synthetic series used in the first learning "
        "tabs. Each rerun rebuilds the known components, then adds them together."
    )
    periods = st.slider(
        "Number of weekly observations",
        52,
        260,
        156,
        step=13,
        help="Controls the length of the generated weekly time series. 52 points is about one year; 260 points is about five years.",
    )
    trend_strength = st.slider(
        "Trend strength",
        -1.0,
        1.0,
        0.45,
        step=0.05,
        help="Controls the long-run direction. Positive values slope upward, negative values slope downward, and zero removes the trend.",
    )
    seasonal_strength = st.slider(
        "Seasonality strength",
        0.0,
        2.0,
        1.0,
        step=0.05,
        help="Scales the repeating seasonal wave. Zero removes seasonality; larger values make the cycle more visible.",
    )
    seasonal_period = st.select_slider(
        "Seasonal cycle length",
        options=[4, 8, 13, 26, 52],
        value=52,
        help="Sets how many observations make one full seasonal cycle. For weekly data, 52 means one yearly cycle.",
    )
    noise_strength = st.slider(
        "Residual noise",
        0.0,
        2.0,
        0.55,
        step=0.05,
        help="Controls random unexplained variation around the structured signal.",
    )
    shock_size = st.slider(
        "One-time event size",
        -2.0,
        2.0,
        0.65,
        step=0.05,
        help="Adds a single temporary event around two-thirds through the series. Positive values create a spike; negative values create a dip.",
    )
    seed = st.number_input(
        "Random seed",
        min_value=0,
        max_value=9999,
        value=7,
        step=1,
        help="Keeps the random residual noise reproducible. Change it to generate a different noise pattern with the same settings.",
    )

    with st.expander("How the initial signal is created"):
        st.markdown(
            """
            The app creates weekly dates starting on 2021-01-01, then builds each
            component separately:

            `observed = level + trend + seasonality + event + residual`

            **Level** is fixed at `100`.

            **Trend** is a straight line scaled by trend strength.

            **Seasonality** combines sine and cosine waves that repeat every
            selected seasonal period.

            **Event** is a temporary spike or dip centered late in the series.

            **Residual** is random noise generated from the selected seed.
            """
        )


df = make_series(
    periods=periods,
    trend_strength=trend_strength,
    seasonal_strength=seasonal_strength,
    seasonal_period=seasonal_period,
    noise_strength=noise_strength,
    shock_size=shock_size,
    seed=seed,
)

st.title("Time Series Decomposition Lab")
st.caption(
    "Build a signal, split it into components, and see how each part explains the final time series."
)

(
    overview,
    components,
    trend_estimation,
    seasonality_estimation,
    rebuild,
    additive_decomposition,
    multiplicative_decomposition,
    intermediate,
    stl_tab,
    regression,
    practice,
) = st.tabs(
    [
        "Signal",
        "Components",
        "Trend estimation",
        "Seasonality estimation",
        "Rebuild",
        "Additive decomposition",
        "Multiplicative decomposition",
        "Estimate & Forecast",
        "STL Decomposition",
        "Linear Regression Forecast",
        "Practice",
    ]
)

with overview:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Observed Series")
        st.line_chart(df.set_index("date")[["observed"]], height=390)
        st.markdown(
            f"""
            The displayed signal is generated from **{periods} weekly points**.
            It starts from a fixed level of `100`, adds a trend scaled by
            `{trend_strength:.2f}`, a repeating seasonal cycle of
            `{seasonal_period}` observations scaled by `{seasonal_strength:.2f}`,
            one temporary event scaled by `{shock_size:.2f}`, and residual noise
            scaled by `{noise_strength:.2f}`.
            """
        )

    with right:
        st.subheader("Additive Model")
        st.markdown(
            """
            A common decomposition writes a time series as:

            `observed = level + trend + seasonality + residual`

            This app also includes a one-time event component so you can see how
            unusual shocks differ from repeating seasonal movement.
            """
        )
        st.metric("Maximum reconstruction error", f"{reconstruction_error(df):.6f}")
        st.metric("Average observed value", f"{df['observed'].mean():.1f}")
        st.metric("Observed range", f"{df['observed'].max() - df['observed'].min():.1f}")
        st.download_button(
            "Download synthetic data CSV",
            dataframe_to_csv(df),
            file_name="synthetic_time_series_components.csv",
            mime="text/csv",
            key="download_synthetic_data",
        )

    st.subheader("What To Look For")
    lesson_cols = st.columns(4)
    lesson_cols[0].write("**Level** is the baseline around which the series moves.")
    lesson_cols[1].write("**Trend** is the long-run upward or downward direction.")
    lesson_cols[2].write("**Seasonality** is a pattern that repeats at a known interval.")
    lesson_cols[3].write("**Residuals** are what remains after known structure is removed.")

with components:
    st.subheader("Component Views")
    visible = st.multiselect(
        "Choose components to compare",
        ["level", "trend", "seasonality", "event", "residual"],
        default=["trend", "seasonality", "event", "residual"],
    )

    chart_df = df.set_index("date")[visible] if visible else df.set_index("date")[["observed"]]
    st.line_chart(chart_df, height=360)

    st.subheader("Component Strength")
    component_summary = pd.DataFrame(
        {
            "component": ["trend", "seasonality", "event", "residual"],
            "range": [
                df["trend"].max() - df["trend"].min(),
                df["seasonality"].max() - df["seasonality"].min(),
                df["event"].max() - df["event"].min(),
                df["residual"].max() - df["residual"].min(),
            ],
            "standard_deviation": [
                df["trend"].std(),
                df["seasonality"].std(),
                df["event"].std(),
                df["residual"].std(),
            ],
        }
    )
    st.dataframe(component_summary, hide_index=True, width="stretch")

    st.subheader("Shape Comparison")
    normalized = pd.DataFrame(
        {
            "trend": normalize_component(df["trend"]),
            "seasonality": normalize_component(df["seasonality"]),
            "event": normalize_component(df["event"]),
            "residual": normalize_component(df["residual"]),
        },
        index=df["date"],
    )
    st.line_chart(normalized, height=290)

with trend_estimation:
    st.subheader("Trend Estimation")
    st.write(
        "Trend estimators try to keep slow, persistent movement while suppressing "
        "seasonal cycles, events, and residual noise."
    )
    st.latex(r"y_t = \text{trend-cycle}_t + \text{short-term movement}_t")

    max_trend_window = min(105, len(df))
    if max_trend_window % 2 == 0:
        max_trend_window -= 1
    max_trend_window = max(3, max_trend_window)
    default_trend_window = min(max_trend_window, make_trend_window(seasonal_period))

    t1, t2, t3 = st.columns(3)
    trend_smoothing_window = t1.slider(
        "Centered moving-average window",
        min_value=3,
        max_value=max_trend_window,
        value=default_trend_window,
        step=2,
        help="A wider window removes more short-term movement but reacts more slowly to turning points.",
        key="trend_estimation_smoothing_window",
    )
    trend_ewm_span = t2.slider(
        "Exponential smoothing span",
        min_value=2,
        max_value=len(df),
        value=min(max(8, seasonal_period // 2), len(df)),
        step=1,
        help="A larger span gives older observations more influence and makes the estimate smoother.",
        key="trend_estimation_ewm_span",
    )
    trend_regression_history = t3.slider(
        "Regression history",
        min_value=2,
        max_value=len(df),
        value=min(max(52, seasonal_period), len(df)),
        step=1,
        help="Only this many recent observations are used to fit the straight-line trend.",
        key="trend_estimation_regression_history",
    )

    trend_methods = estimate_trend_methods(
        df,
        smoothing_window=trend_smoothing_window,
        ewm_span=trend_ewm_span,
        regression_history=trend_regression_history,
    )
    trend_chart = trend_methods.set_index("date")[
        [
            "observed",
            "known_trend_cycle",
            "centered_moving_average",
            "exponential_smoothing",
            "linear_regression",
        ]
    ]
    st.line_chart(trend_chart, height=430)

    trend_summary_rows = []
    for label, column in [
        ("Centered moving average", "centered_moving_average"),
        ("Exponential smoothing", "exponential_smoothing"),
        ("Linear regression", "linear_regression"),
    ]:
        estimate = trend_methods[column]
        known = trend_methods["known_trend_cycle"]
        trend_summary_rows.append(
            {
                "method": label,
                "mean_absolute_gap_to_known_trend_cycle": float(
                    np.mean(np.abs(estimate - known))
                ),
                "smoothness_diff_std": float(estimate.diff().std()),
                "latest_gap": float(estimate.iloc[-1] - known.iloc[-1]),
            }
        )

    s1, s2 = st.columns([1, 1], gap="large")
    with s1:
        st.subheader("Fit Comparison")
        st.dataframe(pd.DataFrame(trend_summary_rows), hide_index=True, width="stretch")

    with s2:
        st.subheader("When Each Method Helps")
        st.dataframe(
            pd.DataFrame(
                {
                    "method": [
                        "Centered moving average",
                        "Exponential smoothing",
                        "Linear regression",
                    ],
                    "best_for": [
                        "Removing cycles when the window covers a full seasonal period.",
                        "Online or recent-weighted smoothing where the newest data matters more.",
                        "A simple long-run slope when the trend is approximately straight.",
                    ],
                    "watch_out_for": [
                        "Lag near sharp turns and edge interpolation near the start and end.",
                        "Can follow noise if the span is too short.",
                        "Misses curved trends and leaves seasonality in the residuals.",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

with seasonality_estimation:
    st.subheader("Seasonality Estimation")
    st.write(
        "Seasonality estimators group observations by their position inside a "
        "repeating cycle, then estimate the typical deviation for each phase."
    )
    st.latex(r"S_j = \operatorname{average}(y_t - \hat{C}_t)\quad\text{where }t\bmod m=j")

    max_seasonal_period = max(2, min(104, len(df)))
    default_seasonal_period = min(seasonal_period, max_seasonal_period)
    c1, c2, c3 = st.columns(3)
    seasonality_period = c1.slider(
        "Seasonal period `m`",
        min_value=2,
        max_value=max_seasonal_period,
        value=default_seasonal_period,
        help="Number of observations in one full repeating cycle.",
        key="seasonality_estimation_period",
    )

    max_seasonality_window = min(105, len(df))
    if max_seasonality_window % 2 == 0:
        max_seasonality_window -= 1
    max_seasonality_window = max(3, max_seasonality_window)
    default_seasonality_window = min(
        max_seasonality_window,
        make_trend_window(seasonality_period),
    )
    seasonality_trend_window = c2.slider(
        "Detrending window",
        min_value=3,
        max_value=max_seasonality_window,
        value=default_seasonality_window,
        step=2,
        help="Used to estimate the trend-cycle before calculating seasonal deviations.",
        key="seasonality_estimation_trend_window",
    )
    seasonality_shrinkage = c3.slider(
        "Seasonal shrinkage",
        min_value=0.0,
        max_value=1.5,
        value=1.0,
        step=0.05,
        help="Scales seasonal estimates toward or away from zero.",
        key="seasonality_estimation_shrinkage",
    )

    seasonal_methods, seasonal_pattern = estimate_seasonality_methods(
        df,
        period=seasonality_period,
        trend_window=seasonality_trend_window,
        shrinkage=seasonality_shrinkage,
    )

    seasonal_chart = seasonal_methods.set_index("date")[
        [
            "known_seasonality",
            "detrended_phase_average",
            "raw_phase_average",
            "latest_cycle_template",
        ]
    ]
    st.line_chart(seasonal_chart, height=390)

    st.subheader("Estimated Seasonal Pattern By Phase")
    st.bar_chart(seasonal_pattern.set_index("phase"), height=320)

    seasonal_summary_rows = []
    for label, column in [
        ("Detrended phase average", "detrended_phase_average"),
        ("Raw phase average", "raw_phase_average"),
        ("Latest-cycle template", "latest_cycle_template"),
    ]:
        estimate = seasonal_methods[column]
        known = seasonal_methods["known_seasonality"]
        seasonal_summary_rows.append(
            {
                "method": label,
                "mean_absolute_gap_to_known_seasonality": float(
                    np.mean(np.abs(estimate - known))
                ),
                "estimated_amplitude": float(estimate.max() - estimate.min()),
                "mean_bias": float((estimate - known).mean()),
            }
        )

    se1, se2 = st.columns([1, 1], gap="large")
    with se1:
        st.subheader("Fit Comparison")
        st.dataframe(pd.DataFrame(seasonal_summary_rows), hide_index=True, width="stretch")

    with se2:
        st.subheader("Method Intuition")
        st.dataframe(
            pd.DataFrame(
                {
                    "method": [
                        "Detrended phase average",
                        "Raw phase average",
                        "Latest-cycle template",
                    ],
                    "idea": [
                        "Remove the trend-cycle first, then average each seasonal phase.",
                        "Average phases without removing trend first.",
                        "Reuse the most recent complete seasonal cycle as the pattern.",
                    ],
                    "typical_risk": [
                        "Needs a reasonable trend estimate.",
                        "Trend can leak into the seasonal pattern.",
                        "Can overreact to recent noise or one-time events.",
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )

with rebuild:
    st.subheader("Rebuild The Observed Series")
    st.write("Turn components on and off to see what each part contributes.")

    c1, c2, c3, c4, c5 = st.columns(5)
    use_level = c1.toggle("Level", value=True)
    use_trend = c2.toggle("Trend", value=True)
    use_seasonality = c3.toggle("Seasonality", value=True)
    use_event = c4.toggle("Event", value=True)
    use_residual = c5.toggle("Residual", value=False)

    partial = np.zeros(len(df))
    selected_parts = []
    for enabled, name in [
        (use_level, "level"),
        (use_trend, "trend"),
        (use_seasonality, "seasonality"),
        (use_event, "event"),
        (use_residual, "residual"),
    ]:
        if enabled:
            partial = partial + df[name].to_numpy()
            selected_parts.append(name)

    rebuild_df = pd.DataFrame(
        {
            "date": df["date"],
            "observed": df["observed"],
            "selected_components": partial,
        }
    ).set_index("date")

    st.line_chart(rebuild_df, height=390)

    mean_abs_error = float(np.mean(np.abs(df["observed"].to_numpy() - partial)))
    st.metric("Mean absolute gap from observed series", f"{mean_abs_error:.2f}")
    if "residual" not in selected_parts:
        st.info(
            "The selected components explain the structure. The remaining gap is mostly residual noise."
        )
    else:
        st.success("With every component included, the rebuilt line matches the observed series.")

with additive_decomposition:
    st.subheader("Additive Decomposition")
    st.write(
        "Additive decomposition treats each component as an amount measured in "
        "the same units as the observed series."
    )
    st.latex(r"y_t = L_t + T_t + S_t + E_t + e_t")

    additive_build = pd.DataFrame(
        {
            "date": df["date"],
            "observed": df["observed"],
            "level": df["level"],
            "level + trend": df["level"] + df["trend"],
            "level + trend + seasonality": df["level"]
            + df["trend"]
            + df["seasonality"],
            "full additive reconstruction": df["level"]
            + df["trend"]
            + df["seasonality"]
            + df["event"]
            + df["residual"],
        }
    ).set_index("date")
    st.line_chart(additive_build, height=430)

    additive_component_view = df.set_index("date")[
        ["level", "trend", "seasonality", "event", "residual"]
    ]
    st.subheader("Components In Original Units")
    st.line_chart(additive_component_view, height=330)

    additive_gap = float(
        np.abs(additive_build["observed"] - additive_build["full additive reconstruction"]).max()
    )
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("Reconstruction gap", f"{additive_gap:.6f}")
    a2.metric("Trend contribution", f"{df['trend'].iloc[-1] - df['trend'].iloc[0]:.1f}")
    a3.metric("Seasonal amplitude", f"{df['seasonality'].max() - df['seasonality'].min():.1f}")
    a4.metric("Residual std. dev.", f"{df['residual'].std():.1f}")

    ad1, ad2 = st.columns([1, 1], gap="large")
    with ad1:
        st.subheader("How To Read It")
        st.markdown(
            """
            **Level** sets the baseline.

            **Trend** adds slow movement above or below the baseline.

            **Seasonality** adds a repeating amount for each phase of the cycle.

            **Events and residuals** add one-time shocks and leftover noise.
            """
        )
    with ad2:
        st.subheader("When Additive Fits")
        st.markdown(
            """
            Additive decomposition is usually appropriate when seasonal swings
            stay about the same size as the series rises or falls.

            If a series grows from `100` to `200`, an additive seasonal effect
            might still be roughly `+10` in high season and `-10` in low season.
            """
        )

    st.subheader("Recent Additive Breakdown")
    st.dataframe(
        df[
            [
                "date",
                "observed",
                "level",
                "trend",
                "seasonality",
                "event",
                "residual",
            ]
        ].tail(12),
        hide_index=True,
        width="stretch",
    )

with multiplicative_decomposition:
    st.subheader("Multiplicative Decomposition")
    st.write(
        "Multiplicative decomposition treats trend, seasonality, events, and "
        "residuals as factors that scale a positive baseline."
    )
    st.latex(r"y_t = L_t \times T_t \times S_t \times E_t \times R_t")
    st.info(
        "Multiplicative decomposition requires strictly positive values. This "
        "tab builds a positive demo series from the same sidebar settings."
    )

    multiplicative_df = make_multiplicative_series(
        periods=periods,
        trend_strength=trend_strength,
        seasonal_strength=seasonal_strength,
        seasonal_period=seasonal_period,
        noise_strength=noise_strength,
        shock_size=shock_size,
        seed=seed,
    )
    multiplicative_df["level x trend"] = (
        multiplicative_df["level"] * multiplicative_df["trend_factor"]
    )
    multiplicative_df["level x trend x seasonality"] = (
        multiplicative_df["level x trend"] * multiplicative_df["seasonal_factor"]
    )
    multiplicative_df["full multiplicative reconstruction"] = (
        multiplicative_df["level"]
        * multiplicative_df["trend_factor"]
        * multiplicative_df["seasonal_factor"]
        * multiplicative_df["event_factor"]
        * multiplicative_df["residual_factor"]
    )

    multiplicative_build = multiplicative_df.set_index("date")[
        [
            "observed",
            "level x trend",
            "level x trend x seasonality",
            "full multiplicative reconstruction",
        ]
    ]
    st.line_chart(multiplicative_build, height=430)

    st.subheader("Multiplicative Factors")
    factor_chart = multiplicative_df.set_index("date")[
        ["trend_factor", "seasonal_factor", "event_factor", "residual_factor"]
    ]
    st.line_chart(factor_chart, height=330)

    multiplicative_gap = float(
        np.abs(
            multiplicative_df["observed"]
            - multiplicative_df["full multiplicative reconstruction"]
        ).max()
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Reconstruction gap", f"{multiplicative_gap:.6f}")
    m2.metric("Final trend factor", f"{multiplicative_df['trend_factor'].iloc[-1]:.2f}x")
    m3.metric(
        "Seasonal factor range",
        f"{multiplicative_df['seasonal_factor'].min():.2f}x to "
        f"{multiplicative_df['seasonal_factor'].max():.2f}x",
    )
    m4.metric("Residual factor std. dev.", f"{multiplicative_df['residual_factor'].std():.3f}")

    md1, md2 = st.columns([1, 1], gap="large")
    with md1:
        st.subheader("How To Read It")
        st.markdown(
            """
            A factor of `1.00` means no change.

            A seasonal factor of `1.15` means the season lifts the trend level by 15%.

            A residual factor below `1.00` means the observed value landed below
            what the structured factors predicted.
            """
        )
    with md2:
        st.subheader("When Multiplicative Fits")
        st.markdown(
            """
            Multiplicative decomposition is usually appropriate when seasonal
            swings grow or shrink with the level of the series.

            If a series doubles, a multiplicative seasonal effect of `1.10`
            changes from `+10` units at level `100` to `+20` units at level `200`.
            """
        )

    st.subheader("Recent Multiplicative Breakdown")
    st.dataframe(
        multiplicative_df[
            [
                "date",
                "observed",
                "level",
                "trend_factor",
                "seasonal_factor",
                "event_factor",
                "residual_factor",
            ]
        ].tail(12),
        hide_index=True,
        width="stretch",
    )

with intermediate:
    st.subheader("Estimate Components From Existing Data")
    st.write(
        "In real datasets the components are not given to us. This section fits them from the observed series with a simple additive model:"
    )
    st.latex(r"y_t = L + T_t + S_t + e_t")
    st.caption("Here `y_t` is the observed value, `L` is the fitted level, `T_t` is trend, `S_t` is seasonality, and `e_t` is the residual.")

    source = st.radio(
        "Dataset source",
        ["Use current synthetic series", "Upload CSV"],
        horizontal=True,
    )

    observed_data = df[["date", "observed"]].rename(columns={"observed": "value"})
    ready_to_fit = True

    if source == "Upload CSV":
        uploaded_data = render_time_series_uploader(
            "Upload a CSV with a date column and numeric value column",
            "Upload a CSV to estimate components from your own dataset.",
            key_prefix="additive",
            date_label="Date column",
            value_label="Value column",
        )
        ready_to_fit = uploaded_data is not None
        if uploaded_data is not None:
            observed_data = uploaded_data

    if not ready_to_fit:
        pass
    elif len(observed_data) < 12:
        st.warning("Use at least 12 observations so the trend and seasonal estimates have enough data.")
    else:
        controls, explanation = st.columns([1, 1], gap="large")

        with controls:
            st.subheader("Fitting Parameters")
            max_period = max(2, min(104, len(observed_data) // 2))
            default_period = min(seasonal_period, max_period)
            estimated_period = st.slider(
                "Seasonal period `m`",
                min_value=2,
                max_value=max_period,
                value=default_period,
                help="Number of observations in one full repeating cycle. Weekly yearly data usually uses 52.",
            )
            max_window = max(3, min(105, len(observed_data)))
            if max_window % 2 == 0:
                max_window -= 1
            default_window = min(max(make_trend_window(estimated_period), 5), max_window)
            trend_window = st.slider(
                "Trend-cycle smoothing window `w`",
                min_value=3,
                max_value=max_window,
                value=default_window,
                step=2,
                help="Centered moving-average window used to remove short-term wiggles before estimating trend.",
            )
            level_reference = st.selectbox(
                "Level reference",
                ["Full history", "First cycle", "Last cycle"],
                help="The part of the smoothed trend-cycle used to define the baseline level.",
            )
            seasonal_shrinkage = st.slider(
                "Seasonal shrinkage `gamma`",
                0.0,
                1.5,
                1.0,
                step=0.05,
                help="Scales the learned seasonal pattern. Values below 1 make seasonality more conservative.",
            )

        with explanation:
            st.subheader("Fitting Steps")
            st.markdown(
                """
                **1. Smooth the observed data**

                `C_t = moving_average_w(y_t)`

                `C_t` is the trend-cycle curve. The window `w` controls how smooth it is.

                **2. Fit the level**

                `L = average(C_t over the selected reference range)`

                The level is the baseline. Changing the reference range changes what the app treats as normal.

                **3. Fit the trend**

                `T_t = C_t - L`

                Trend is the smoothed movement above or below the fitted level.

                **4. Fit seasonality**

                `S_j = gamma * average(y_t - L - T_t)` for observations where `t mod m = j`

                The seasonal period `m` defines the repeating cycle. `gamma` scales the strength of the fitted seasonal pattern.

                **5. Fit residuals**

                `e_t = y_t - L - T_t - S_t`

                Residuals are the leftover errors after level, trend, and seasonality are removed.
                """
            )

        estimated, seasonal_pattern, level_value = estimate_additive_components(
            observed_data,
            seasonal_period=estimated_period,
            trend_window=trend_window,
            level_reference=level_reference,
            seasonal_shrinkage=seasonal_shrinkage,
        )

        st.subheader("Estimated Components")
        component_chart = estimated.set_index("date")[
            ["value", "level", "trend", "seasonality", "residual"]
        ]
        st.line_chart(component_chart, height=390)

        fit_error = float(np.mean(np.abs(estimated["value"] - estimated["fitted"])))
        residual_std = float(estimated["residual"].std())
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fitted level", f"{level_value:.2f}")
        m2.metric("Mean absolute fit error", f"{fit_error:.2f}")
        m3.metric("Residual standard deviation", f"{residual_std:.2f}")
        m4.metric("Seasonal amplitude", f"{seasonal_pattern.max() - seasonal_pattern.min():.2f}")

        st.subheader("How The Fit Reconstructs The Data")
        st.dataframe(
            estimated[
                [
                    "date",
                    "value",
                    "level",
                    "trend",
                    "seasonality",
                    "residual",
                    "fitted",
                ]
            ].tail(12),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download estimated components CSV",
            dataframe_to_csv(estimated),
            file_name="estimated_components.csv",
            mime="text/csv",
            key="download_additive_components",
        )

        st.subheader("Forecast Parameters")
        st.write(
            "The forecast extends the fitted trend-cycle, repeats the learned seasonal pattern, and optionally carries forward part of the recent residual bias."
        )
        st.markdown(
            """
            **Trend damping** scales the future slope before it is projected forward.

            `future_slope = fitted_slope * damping`

            A value of `1.0` keeps the recent trend unchanged. A value below `1.0`
            makes the future trend flatter. A value above `1.0` makes the future
            trend steeper. A value of `0.0` freezes the trend at its latest level.
            """
        )
        f1, f2, f3, f4 = st.columns(4)
        horizon = f1.slider("Forecast horizon", 4, 104, 26, step=1)
        trend_fit_points = f2.slider(
            "Recent points for future trend",
            2,
            min(104, len(estimated)),
            min(26, len(estimated)),
            step=1,
            help="Number of recent fitted trend-cycle points used to estimate the future trend slope.",
        )
        trend_damping = f3.slider(
            "Future trend damping",
            0.0,
            1.5,
            1.0,
            step=0.05,
            help="Scales the fitted future trend slope. Use lower values when you expect growth or decline to slow down.",
        )
        seasonal_scale = f4.slider(
            "Future seasonal strength",
            0.0,
            1.5,
            1.0,
            step=0.05,
            help="Scales the seasonal pattern used in the forecast without changing the historical fit.",
        )
        residual_adjustment = st.slider(
            "Future residual adjustment",
            -1.0,
            1.0,
            0.0,
            step=0.05,
            help="Adds a fraction of recent average residuals to the forecast when the model has been under- or over-predicting recently.",
        )

        forecast = make_forecast(
            estimated,
            seasonal_pattern,
            horizon=horizon,
            trend_fit_points=trend_fit_points,
            trend_damping=trend_damping,
            seasonal_scale=seasonal_scale,
            residual_adjustment=residual_adjustment,
        )

        history_for_chart = estimated[["date", "value", "fitted"]].rename(
            columns={"value": "observed", "fitted": "estimated_fit"}
        )
        forecast_for_chart = forecast[["date", "forecast", "lower_band", "upper_band"]]
        combined = pd.merge(
            history_for_chart,
            forecast_for_chart,
            how="outer",
            on="date",
        ).set_index("date")

        st.subheader("Forecast From Estimated Components")
        st.line_chart(combined, height=420)
        st.info(
            "The lower and upper bands are approximate uncertainty bands: "
            "`lower = forecast - 1.96 * residual_std` and "
            "`upper = forecast + 1.96 * residual_std`. The residual standard "
            "deviation is calculated from the historical errors left after fitting "
            "level, trend, and seasonality."
        )

        st.subheader("Forecast Breakdown")
        st.dataframe(
            forecast[
                [
                    "date",
                    "level",
                    "trend",
                    "seasonality",
                    "residual_adjustment",
                    "forecast",
                    "lower_band",
                    "upper_band",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download component forecast CSV",
            dataframe_to_csv(forecast),
            file_name="component_forecast.csv",
            mime="text/csv",
            key="download_additive_forecast",
        )

with stl_tab:
    st.subheader("STL Decomposition")
    st.write(
        "STL separates a series into trend-cycle, seasonality, and residuals with LOESS smoothers."
    )
    st.latex(r"y_t = T_t + S_t + R_t")

    if STL is None:
        st.error(
            "STL requires the `statsmodels` package. Install dependencies with `pip install -r requirements.txt`."
        )
    else:
        stl_source = st.radio(
            "STL dataset source",
            ["Use current synthetic series", "Upload CSV"],
            horizontal=True,
            key="stl_source",
        )

        stl_data = df[["date", "observed"]].rename(columns={"observed": "value"})
        using_synthetic_stl = stl_source == "Use current synthetic series"
        ready_to_fit_stl = True

        if stl_source == "Upload CSV":
            uploaded_stl_data = render_time_series_uploader(
                "Upload a CSV for STL decomposition",
                "Upload a CSV to run STL on your own dataset.",
                key_prefix="stl",
                date_label="STL date column",
                value_label="STL value column",
            )
            ready_to_fit_stl = uploaded_stl_data is not None
            if uploaded_stl_data is not None:
                stl_data = uploaded_stl_data
                using_synthetic_stl = False

        if not ready_to_fit_stl:
            pass
        elif len(stl_data) < 14:
            st.warning("Use at least 14 observations so STL has enough data to estimate smoothers.")
        else:
            controls, guidance = st.columns([1, 1], gap="large")

            with controls:
                st.subheader("STL Parameters")
                max_period = max(2, min(104, len(stl_data) // 2))
                default_stl_period = min(seasonal_period, max_period)
                stl_period = st.slider(
                    "Seasonal period",
                    min_value=2,
                    max_value=max_period,
                    value=default_stl_period,
                    help="Number of observations in one repeating cycle. Weekly annual data usually uses 52.",
                    key="stl_period",
                )

                max_seasonal_window = make_odd_window(min(105, len(stl_data)), minimum=7)
                if max_seasonal_window > len(stl_data) and len(stl_data) % 2 == 0:
                    max_seasonal_window = len(stl_data) - 1
                max_seasonal_window = max(7, max_seasonal_window)
                default_seasonal_window = min(
                    max_seasonal_window,
                    make_odd_window(max(7, stl_period // 2 + 1), minimum=7),
                )
                stl_seasonal_window = st.slider(
                    "Seasonal smoother",
                    min_value=7,
                    max_value=max_seasonal_window,
                    value=default_seasonal_window,
                    step=2,
                    help="Smaller values let the seasonal pattern change more over time; larger values make it more stable.",
                    key="stl_seasonal_window",
                )

                min_trend_window = make_odd_window(stl_period + 1)
                max_trend_window = make_odd_window(min(157, max(len(stl_data), min_trend_window)))
                if max_trend_window < min_trend_window:
                    max_trend_window = min_trend_window
                default_trend_window = min(
                    max_trend_window,
                    make_odd_window(max(min_trend_window, int(1.5 * stl_period))),
                )
                stl_trend_window = st.slider(
                    "Trend smoother",
                    min_value=min_trend_window,
                    max_value=max_trend_window,
                    value=default_trend_window,
                    step=2,
                    help="Larger values make the trend-cycle smoother and push more movement into seasonality or residuals.",
                    key="stl_trend_window",
                )

                min_low_pass_window = make_odd_window(stl_period + 1)
                max_low_pass_window = make_odd_window(min(157, max(len(stl_data), min_low_pass_window)))
                default_low_pass_window = min(
                    max_low_pass_window,
                    make_odd_window(max(min_low_pass_window, stl_period + 2)),
                )
                stl_low_pass_window = st.slider(
                    "Low-pass smoother",
                    min_value=min_low_pass_window,
                    max_value=max_low_pass_window,
                    value=default_low_pass_window,
                    step=2,
                    help="Controls smoothing used while separating seasonal movement from the trend-cycle.",
                    key="stl_low_pass_window",
                )
                stl_robust = st.toggle(
                    "Robust to outliers",
                    value=True,
                    help="Downweights unusual points so spikes are more likely to remain in residuals.",
                    key="stl_robust",
                )

            with guidance:
                st.subheader("How To Read STL")
                st.markdown(
                    """
                    **Period** tells STL where repeating seasonal phases line up.

                    **Seasonal smoother** controls how much the seasonal pattern can evolve across cycles.

                    **Trend smoother** controls how slowly the trend-cycle is allowed to move.

                    **Robust mode** is useful when one-time shocks should not distort the trend or seasonal pattern.
                    """
                )
                st.info(
                    "If the residuals still show a repeating wave, the period or seasonal smoother is probably wrong."
                )

            try:
                stl_estimated = fit_stl_components(
                    stl_data,
                    period=stl_period,
                    seasonal_window=stl_seasonal_window,
                    trend_window=stl_trend_window,
                    low_pass_window=stl_low_pass_window,
                    robust=stl_robust,
                )
            except ValueError as exc:
                st.error(f"STL could not fit with the selected settings: {exc}")
                stl_estimated = None

            if stl_estimated is not None:
                st.subheader("STL Components")
                stl_chart = stl_estimated.set_index("date")[
                    ["value", "trend_cycle", "seasonality", "residual"]
                ]
                st.line_chart(stl_chart, height=410)

                stl_fit_error = float(np.mean(np.abs(stl_estimated["value"] - stl_estimated["fitted"])))
                stl_reconstruction_error = float(
                    np.abs(stl_estimated["value"] - stl_estimated["fitted"] - stl_estimated["residual"]).max()
                )
                stl_residual_std = float(stl_estimated["residual"].std())
                stl_seasonal_strength = component_strength(
                    stl_estimated["seasonality"], stl_estimated["residual"]
                )
                stl_trend_strength = component_strength(
                    stl_estimated["trend_cycle"], stl_estimated["residual"]
                )
                stl_outliers = residual_outlier_count(stl_estimated["residual"])

                sm1, sm2, sm3, sm4, sm5 = st.columns(5)
                sm1.metric("Mean absolute residual", f"{stl_fit_error:.2f}")
                sm2.metric("Residual std. dev.", f"{stl_residual_std:.2f}")
                sm3.metric("Seasonal strength", f"{stl_seasonal_strength:.2f}")
                sm4.metric("Trend strength", f"{stl_trend_strength:.2f}")
                sm5.metric("3-sigma residuals", f"{stl_outliers}")

                st.caption(
                    f"Maximum reconstruction error after adding trend-cycle, seasonality, and residuals: {stl_reconstruction_error:.6f}"
                )
                st.download_button(
                    "Download STL components CSV",
                    dataframe_to_csv(stl_estimated),
                    file_name="stl_components.csv",
                    mime="text/csv",
                    key="download_stl_components",
                )

                if using_synthetic_stl:
                    comparison = pd.DataFrame(
                        {
                            "target": ["known seasonality", "known trend plus level"],
                            "mean_absolute_gap": [
                                float(np.mean(np.abs(stl_estimated["seasonality"] - df["seasonality"]))),
                                float(
                                    np.mean(
                                        np.abs(
                                            stl_estimated["trend_cycle"]
                                            - (df["level"] + df["trend"])
                                        )
                                    )
                                ),
                            ],
                        }
                    )
                    st.subheader("Synthetic-Series Check")
                    st.dataframe(comparison, hide_index=True, width="stretch")
                    st.caption(
                        "The event spike is not part of the true level or trend. Robust STL should tend to leave more of that spike in residuals."
                    )

                st.subheader("Residual Diagnostics")
                diagnostic_df = stl_estimated.copy()
                residual_std_for_flags = diagnostic_df["residual"].std()
                if residual_std_for_flags and not np.isnan(residual_std_for_flags):
                    diagnostic_df["residual_z"] = diagnostic_df["residual"] / residual_std_for_flags
                else:
                    diagnostic_df["residual_z"] = 0.0
                st.line_chart(
                    diagnostic_df.set_index("date")[["residual", "seasonality"]],
                    height=300,
                )
                st.dataframe(
                    diagnostic_df.reindex(
                        diagnostic_df["residual"].abs().sort_values(ascending=False).index
                    )[["date", "value", "trend_cycle", "seasonality", "residual", "residual_z"]].head(10),
                    hide_index=True,
                    width="stretch",
                )

                st.subheader("Period Sensitivity")
                candidate_periods = sorted(
                    {
                        max(2, stl_period // 2),
                        stl_period,
                        min(max_period, stl_period * 2),
                    }
                )
                sensitivity_rows = []
                for candidate in candidate_periods:
                    candidate_trend = make_odd_window(max(candidate + 1, stl_trend_window))
                    candidate_low_pass = make_odd_window(max(candidate + 1, stl_low_pass_window))
                    try:
                        candidate_fit = fit_stl_components(
                            stl_data,
                            period=candidate,
                            seasonal_window=stl_seasonal_window,
                            trend_window=candidate_trend,
                            low_pass_window=candidate_low_pass,
                            robust=stl_robust,
                        )
                    except ValueError:
                        continue
                    sensitivity_rows.append(
                        {
                            "period": candidate,
                            "mean_absolute_residual": float(candidate_fit["residual"].abs().mean()),
                            "residual_std": float(candidate_fit["residual"].std()),
                            "seasonal_strength": component_strength(
                                candidate_fit["seasonality"],
                                candidate_fit["residual"],
                            ),
                            "trend_strength": component_strength(
                                candidate_fit["trend_cycle"],
                                candidate_fit["residual"],
                            ),
                            "3_sigma_residuals": residual_outlier_count(candidate_fit["residual"]),
                        }
                    )

                sensitivity_df = pd.DataFrame(sensitivity_rows)
                st.dataframe(sensitivity_df, hide_index=True, width="stretch")
                st.download_button(
                    "Download period sensitivity CSV",
                    dataframe_to_csv(sensitivity_df),
                    file_name="stl_period_sensitivity.csv",
                    mime="text/csv",
                    key="download_stl_sensitivity",
                )
                st.caption(
                    "Lower residuals are useful, but the chosen period should also make domain sense. A tiny residual from the wrong period can overfit."
                )

with regression:
    st.subheader("Linear Regression Forecast")
    st.write(
        "This version fits only a straight-line baseline. It is useful for learning how level and linear trend can be estimated with regression before adding more advanced components."
    )
    st.warning(
        "Seasonality is intentionally not included in this example. Any repeating seasonal pattern will remain in the residuals."
    )
    st.latex(r"\hat{y}_t = b + a x_t")
    st.caption(
        "`b` is the fitted level at the start of the selected training window, `a` is the slope per observation, and `x_t` counts steps from that start point."
    )

    regression_source = st.radio(
        "Regression dataset source",
        ["Use current synthetic series", "Upload CSV"],
        horizontal=True,
        key="regression_source",
    )

    regression_data = df[["date", "observed"]].rename(columns={"observed": "value"})
    ready_to_fit_regression = True

    if regression_source == "Upload CSV":
        uploaded_regression_data = render_time_series_uploader(
            "Upload a CSV for regression forecasting",
            "Upload a CSV to fit the regression model to your own dataset.",
            key_prefix="regression",
            date_label="Regression date column",
            value_label="Regression value column",
        )
        ready_to_fit_regression = uploaded_regression_data is not None
        if uploaded_regression_data is not None:
            regression_data = uploaded_regression_data

    if not ready_to_fit_regression:
        pass
    elif len(regression_data) < 12:
        st.warning("Use at least 12 observations so the regression and seasonal estimates have enough data.")
    else:
        controls, explanation = st.columns([1, 1], gap="large")

        with controls:
            st.subheader("Regression Fitting Parameters")
            max_fit_history = len(regression_data)
            default_fit_history = min(max(52, seasonal_period), max_fit_history)
            fit_history = st.slider(
                "History used to fit `a` and `b`",
                min_value=2,
                max_value=max_fit_history,
                value=default_fit_history,
                step=1,
                help="Only the most recent selected observations are used to fit the regression line.",
            )
            residual_centering = st.selectbox(
                "Residual centering",
                ["None", "Fit window", "Full history"],
                help="Optionally remove the average residual so the residual series is centered around zero.",
            )

        with explanation:
            st.subheader("Regression Fitting Steps")
            st.markdown(
                """
                **1. Choose the training window**

                Use the last `h` observations. This is the history window that controls how much old data affects the fitted line.

                **2. Create a time index**

                `x_t = 0, 1, 2, ...` from the start of the selected training window.

                **3. Fit the line**

                `line_t = b + a x_t`

                The app chooses `a` and `b` by least squares, meaning it minimizes:

                `sum((y_t - (b + a x_t))^2)` over the selected training window.

                **4. Split line into level and trend**

                `L = b`

                `T_t = a x_t`

                **5. Residuals are the remaining errors**

                `e_t = y_t - L - T_t`

                This page does not estimate `S_t`. If the data has weekly, monthly,
                or yearly repeating patterns, those patterns stay visible in the
                residuals and in the forecast errors.
                """
            )

        regression_estimated, slope, intercept, fit_start = estimate_regression_components(
            regression_data,
            fit_history=fit_history,
            residual_centering=residual_centering,
        )

        st.subheader("Fitted Regression Components")
        regression_chart = regression_estimated.set_index("date")[
            ["value", "level", "trend", "residual"]
        ]
        st.line_chart(regression_chart, height=390)

        regression_fit_error = float(
            np.mean(np.abs(regression_estimated["value"] - regression_estimated["fitted"]))
        )
        regression_residual_std = float(regression_estimated["residual"].std())
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Slope `a`", f"{slope:.3f}")
        r2.metric("Level `b`", f"{intercept:.2f}")
        r3.metric("Mean absolute fit error", f"{regression_fit_error:.2f}")
        r4.metric("Residual standard deviation", f"{regression_residual_std:.2f}")

        st.subheader("Training Window And Reconstruction")
        st.write(
            "Rows marked `True` are the observations used to estimate the regression line."
        )
        st.dataframe(
            regression_estimated[
                [
                    "date",
                    "x",
                    "value",
                    "level",
                    "trend",
                    "residual",
                    "fitted",
                    "used_for_fit",
                ]
            ].tail(max(12, min(fit_history, 30))),
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download regression fit CSV",
            dataframe_to_csv(regression_estimated),
            file_name="regression_fit.csv",
            mime="text/csv",
            key="download_regression_fit",
        )

        st.subheader("Regression Forecast Parameters")
        st.write(
            "The forecast extends only the line `b + a x` and optionally carries forward recent residual bias."
        )
        st.markdown(
            """
            **Trend damping** scales the regression slope used after the last observed point.

            `future_slope = a * damping`

            A value of `1.0` extends the fitted regression slope exactly. A value
            below `1.0` assumes the trend slows down. A value above `1.0` assumes
            the trend accelerates. A value of `0.0` keeps the future regression
            line flat from the last fitted value.
            """
        )
        lf1, lf2 = st.columns(2)
        regression_horizon = lf1.slider(
            "Regression forecast horizon",
            4,
            104,
            26,
            step=1,
        )
        regression_trend_damping = lf2.slider(
            "Regression future trend damping",
            0.0,
            1.5,
            1.0,
            step=0.05,
            help="Scales the fitted slope `a` when forecasting.",
        )
        regression_residual_adjustment = st.slider(
            "Regression future residual adjustment",
            -1.0,
            1.0,
            0.0,
            step=0.05,
            help="Adds a fraction of recent average residuals to the forecast.",
        )
        regression_residual_history = st.slider(
            "Recent residuals used for bias",
            1,
            min(104, len(regression_estimated)),
            min(26, len(regression_estimated)),
            step=1,
        )

        regression_forecast = make_regression_forecast(
            regression_estimated,
            slope=slope,
            intercept=intercept,
            fit_start=fit_start,
            horizon=regression_horizon,
            trend_damping=regression_trend_damping,
            residual_adjustment=regression_residual_adjustment,
            residual_history=regression_residual_history,
        )

        regression_history_for_chart = regression_estimated[
            ["date", "value", "fitted", "regression_line"]
        ].rename(
            columns={
                "value": "observed",
                "fitted": "linear_regression_fit",
            }
        )
        regression_forecast_for_chart = regression_forecast[
            ["date", "forecast", "lower_band", "upper_band"]
        ]
        regression_combined = pd.merge(
            regression_history_for_chart,
            regression_forecast_for_chart,
            how="outer",
            on="date",
        ).set_index("date")

        st.subheader("Forecast From Linear Regression")
        st.line_chart(regression_combined, height=420)
        st.info(
            "The lower and upper bands are approximate uncertainty bands: "
            "`lower = forecast - 1.96 * residual_std` and "
            "`upper = forecast + 1.96 * residual_std`. Here `residual_std` is "
            "the historical standard deviation of `y_t - (b + a x_t)`. Because "
            "seasonality is not included here, seasonal movement can make these "
            "residuals and bands wider."
        )

        st.subheader("Regression Forecast Breakdown")
        st.dataframe(
            regression_forecast[
                [
                    "date",
                    "x",
                    "level",
                    "trend",
                    "residual_adjustment",
                    "forecast",
                    "lower_band",
                    "upper_band",
                ]
            ],
            hide_index=True,
            width="stretch",
        )
        st.download_button(
            "Download regression forecast CSV",
            dataframe_to_csv(regression_forecast),
            file_name="regression_forecast.csv",
            mime="text/csv",
            key="download_regression_forecast",
        )

with practice:
    st.subheader("Check Your Understanding")

    question_one = st.radio(
        "A pattern repeats every 52 weeks. Which component usually captures it?",
        ["Trend", "Seasonality", "Residual", "Level"],
        index=None,
    )
    if question_one:
        if question_one == "Seasonality":
            st.success("Correct. Seasonality captures repeating behavior at a known interval.")
        else:
            st.error("Try again. Repeating behavior belongs to seasonality.")

    question_two = st.radio(
        "A sudden promotion causes one unusual spike. Which component best describes it here?",
        ["Level", "Trend", "Event", "Seasonality"],
        index=None,
    )
    if question_two:
        if question_two == "Event":
            st.success("Correct. A one-time spike is better treated as an event or anomaly.")
        else:
            st.error("Try again. A one-time spike is not a long-run or repeating pattern.")

    question_three = st.radio(
        "After removing level, trend, seasonality, and events, what remains?",
        ["Observed values", "Residuals", "Calendar periods", "Baseline"],
        index=None,
    )
    if question_three:
        if question_three == "Residuals":
            st.success("Correct. Residuals contain leftover noise and unexplained movement.")
        else:
            st.error("Try again. The leftover part is the residual component.")

    st.subheader("Mini Experiment")
    st.write(
        "Set residual noise to zero, then rebuild the series without residuals. The selected components should explain the observed line almost perfectly."
    )
