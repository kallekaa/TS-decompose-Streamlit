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

overview, components, rebuild, practice = st.tabs(
    ["Signal", "Components", "Rebuild", "Practice"]
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
    st.dataframe(component_summary, hide_index=True, use_container_width=True)

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

