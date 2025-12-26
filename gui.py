import numpy as np
import pandas as pd
import streamlit as st

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# -----------------------
# Page config + styling
# -----------------------
st.set_page_config(
    page_title="Unloading Hours Predictor",
    page_icon="⏱️",
    layout="wide",
)

st.title("Unloading Hours Predictor")
st.caption(
    "Predict unloading time (hours) from gross truck weight, leg distance, and load of leg "
    "using a LightGBM regression model."
)

DATA_URL = "https://raw.githubusercontent.com/cakirogl/unloading_hours/refs/heads/main/inliers0.15.csv"
FEATURES = ["gross_weight", "leg_distance", "load_of_leg"]  # will auto-fallback if CSV has different names


# -----------------------
# Data + model (cached)
# -----------------------
@st.cache_data(show_spinner=False)
def load_data(url: str) -> pd.DataFrame:
    df_ = pd.read_csv(url)
    return df_


def infer_feature_target_columns(df_: pd.DataFrame):
    """
    If your CSV columns are not exactly the names you want in the UI,
    we’ll use: all-but-last as features and last as target.
    """
    x_cols = df_.columns[:-1].tolist()
    y_col = df_.columns[-1]
    return x_cols, y_col


@st.cache_resource(show_spinner=False)
def train_model(df_: pd.DataFrame) -> tuple[Pipeline, list[str], str, dict]:
    x_cols, y_col = infer_feature_target_columns(df_)

    X = df_[x_cols].values
    y = df_[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14
    )

    # Note: scaling is not required for LightGBM, but we keep it
    # to match your current approach and to be consistent in inference.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("lgbm", LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                random_state=14,
            )),
        ]
    )
    model.fit(X_train, y_train)

    # quick health check info (optional to display)
    r2_test = model.score(X_test, y_test)
    meta = {
        "n_samples": int(df_.shape[0]),
        "n_features": int(len(x_cols)),
        "test_r2": float(r2_test),
    }
    return model, x_cols, y_col, meta


# -----------------------
# Load
# -----------------------
with st.spinner("Loading data and preparing the model..."):
    df = load_data(DATA_URL)
    model, x_cols, y_col, meta = train_model(df)

# Use the dataset’s actual min/max as bounds for the UI
mins = df[x_cols].min()
maxs = df[x_cols].max()


# -----------------------
# Sidebar (info)
# -----------------------
with st.sidebar:
    st.header("Model & Dataset")
    st.write(f"**Samples:** {meta['n_samples']:,}")
    st.write(f"**Features:** {meta['n_features']}")
    st.write(f"**Target:** `{y_col}`")
    st.write(f"**Hold-out test R²:** {meta['test_r2']:.5f}")

    with st.expander("Show feature ranges"):
        ranges_df = pd.DataFrame({"min": mins, "max": maxs})
        st.dataframe(ranges_df, use_container_width=True)

    with st.expander("Preview data"):
        st.dataframe(df.head(10), use_container_width=True)


# -----------------------
# Main UI
# -----------------------
st.subheader("Enter shipment details")

# Nice UX: only predict when user clicks the button
with st.form("prediction_form", border=True):
    c1, c2, c3 = st.columns(3)

    with c1:
        gross_weight = st.number_input(
            "Truck gross weight (kg)",
            min_value=float(mins.iloc[0]),
            max_value=float(maxs.iloc[0]),
            value=float(np.clip(5000.0, mins.iloc[0], maxs.iloc[0])),
            step=100.0,
            help="Total truck gross weight within the data-supported range.",
        )

    with c2:
        leg_distance = st.number_input(
            "Leg distance (km)",
            min_value=float(mins.iloc[1]),
            max_value=float(maxs.iloc[1]),
            value=float(np.clip(50.0, mins.iloc[1], maxs.iloc[1])),
            step=1.0,
            help="Trip distance for this leg (kilometers).",
        )

    with c3:
        load_of_leg = st.number_input(
            "Load of leg (kg)",
            min_value=float(mins.iloc[2]),
            max_value=float(maxs.iloc[2]),
            value=float(np.clip(4000.0, mins.iloc[2], maxs.iloc[2])),
            step=100.0,
            help="Cargo weight for this leg within the data-supported range.",
        )

    submitted = st.form_submit_button("Predict unloading hours", type="primary")

if submitted:
    # Build the input in the same column order as training
    new_sample = np.array([[gross_weight, leg_distance, load_of_leg]], dtype=float)

    pred_hours = float(model.predict(new_sample)[0])

    st.divider()
    left, right = st.columns([1, 2], vertical_alignment="center")

    with left:
        st.metric("Predicted unloading time", f"{pred_hours:.2f} h")

    with right:
        st.info(
            "Prediction is produced by a LightGBM regression model trained on the provided dataset. "
            #"Use within the displayed feature ranges for most reliable results."
        )

    with st.expander("Show the input used for prediction"):
        st.write(pd.DataFrame([new_sample[0]], columns=x_cols))
