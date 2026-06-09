import math

import numpy as np
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Time Series Decomposition Lab",
    page_icon="",
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


with st.sidebar:
    st.header("Series Controls")
    periods = st.slider("Number of weekly observations", 52, 260, 156, step=13)
    trend_strength = st.slider("Trend strength", -1.0, 1.0, 0.45, step=0.05)
    seasonal_strength = st.slider("Seasonality strength", 0.0, 2.0, 1.0, step=0.05)
    seasonal_period = st.select_slider(
        "Seasonal cycle length",
        options=[4, 8, 13, 26, 52],
        value=52,
    )
    noise_strength = st.slider("Residual noise", 0.0, 2.0, 0.55, step=0.05)
    shock_size = st.slider("One-time event size", -2.0, 2.0, 0.65, step=0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=7, step=1)


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

overview, components, rebuild, intermediate, regression, practice = st.tabs(
    [
        "Signal",
        "Components",
        "Rebuild",
        "Estimate & Forecast",
        "Linear Regression Forecast",
        "Practice",
    ]
)

with overview:
    left, right = st.columns([2, 1], gap="large")

    with left:
        st.subheader("Observed Series")
        st.line_chart(df.set_index("date")[["observed"]], height=390)

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

    if source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload a CSV with a date column and numeric value column", type="csv")
        if uploaded_file is not None:
            uploaded = pd.read_csv(uploaded_file)
            st.dataframe(uploaded.head(8), width="stretch")

            column_names = list(uploaded.columns)
            date_options = ["Use row number"] + column_names
            date_choice = st.selectbox("Date column", date_options)
            value_choice = st.selectbox("Value column", column_names)

            values = pd.to_numeric(uploaded[value_choice], errors="coerce")
            if date_choice == "Use row number":
                dates = pd.date_range("2021-01-01", periods=len(uploaded), freq="D")
            else:
                dates = pd.to_datetime(uploaded[date_choice], errors="coerce")
                if dates.isna().any():
                    st.warning("Some dates could not be parsed, so row-number dates are being used.")
                    dates = pd.date_range("2021-01-01", periods=len(uploaded), freq="D")

            observed_data = (
                pd.DataFrame({"date": dates, "value": values})
                .dropna(subset=["value"])
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            st.info("Upload a CSV to estimate components from your own dataset.")

    if len(observed_data) < 12:
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

    if regression_source == "Upload CSV":
        regression_upload = st.file_uploader(
            "Upload a CSV for regression forecasting",
            type="csv",
            key="regression_upload",
        )
        if regression_upload is not None:
            uploaded = pd.read_csv(regression_upload)
            st.dataframe(uploaded.head(8), width="stretch")

            column_names = list(uploaded.columns)
            date_options = ["Use row number"] + column_names
            date_choice = st.selectbox(
                "Regression date column",
                date_options,
                key="regression_date_column",
            )
            value_choice = st.selectbox(
                "Regression value column",
                column_names,
                key="regression_value_column",
            )

            values = pd.to_numeric(uploaded[value_choice], errors="coerce")
            if date_choice == "Use row number":
                dates = pd.date_range("2021-01-01", periods=len(uploaded), freq="D")
            else:
                dates = pd.to_datetime(uploaded[date_choice], errors="coerce")
                if dates.isna().any():
                    st.warning("Some dates could not be parsed, so row-number dates are being used.")
                    dates = pd.date_range("2021-01-01", periods=len(uploaded), freq="D")

            regression_data = (
                pd.DataFrame({"date": dates, "value": values})
                .dropna(subset=["value"])
                .sort_values("date")
                .reset_index(drop=True)
            )
        else:
            st.info("Upload a CSV to fit the regression model to your own dataset.")

    if len(regression_data) < 12:
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
