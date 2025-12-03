import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

st.set_page_config(page_title="Forecasting App", layout="wide")
st.title("📈 Streamlit Forecasting — Robust loader for various .pkl structures")

MODEL_PATH = "best_model_AALI.pkl"

@st.cache_resource
def load_pickle(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj

def find_estimator(obj):
    """
    Try to extract an estimator (object with .predict) from loaded pickle.
    Returns a tuple (predict_callable, meta) where:
      - predict_callable(X_df) -> predictions (numpy array / list)
      - meta: dict with debug info (found keys, types)
    """
    meta = {"type": type(obj).__name__}
    # direct estimator
    if hasattr(obj, "predict"):
        meta["found"] = "direct"
        return (lambda X: obj.predict(X)), meta

    # sklearn GridSearchCV or similar
    if hasattr(obj, "best_estimator_"):
        est = obj.best_estimator_
        if hasattr(est, "predict"):
            meta["found"] = "best_estimator_ attribute"
            meta["type"] = type(est).__name__
            return (lambda X: est.predict(X)), meta

    # dict-like structures
    if isinstance(obj, dict):
        meta["keys"] = list(obj.keys())
        # Common key names to try (order matters)
        key_candidates = ["model", "estimator", "best_estimator", "pipeline", "clf", "regressor"]
        for k in key_candidates:
            if k in obj:
                candidate = obj[k]
                if hasattr(candidate, "predict"):
                    meta["found"] = f"dict['{k}']"
                    return (lambda X, c=candidate: c.predict(X)), meta

        # If has preprocessor + model
        if "model" in obj and any(k in obj for k in ["preprocessor", "scaler", "transformer"]):
            model = obj["model"]
            pre = None
            for k in ["preprocessor", "scaler", "transformer"]:
                if k in obj:
                    pre = obj[k]
                    break
            if hasattr(model, "predict") and hasattr(pre, "transform"):
                meta["found"] = "dict['preprocessor'] + dict['model']"
                def predict_fn(X):
                    # expect X is DataFrame; pre.transform should accept DataFrame or numpy
                    Xt = pre.transform(X)
                    return model.predict(Xt)
                return predict_fn, meta

        # Sometimes entire pipeline stored under 'pipeline' but pipeline may be a list/dict
        if "pipeline" in obj:
            p = obj["pipeline"]
            if hasattr(p, "predict"):
                meta["found"] = "dict['pipeline'] (has predict)"
                return (lambda X: p.predict(X)), meta

        # last-resort: search dict values for any object with predict
        for k, v in obj.items():
            if hasattr(v, "predict"):
                meta["found"] = f"dict['{k}'] (fallthrough)"
                return (lambda X, c=v: c.predict(X)), meta

    # cannot find
    meta["found"] = None
    return (None, meta)

def prepare_features_from_df(df, expected_cols=None):
    """
    Basic safety: if model expects specific columns, try to reorder/select them.
    If expected_cols provided, we pick those; otherwise we pass df as-is.
    """
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Input CSV missing columns required by model: {missing}")
        return df[expected_cols]
    return df

# Load model on startup if exists
loaded_obj = None
try:
    loaded_obj = load_pickle(MODEL_PATH)
    predictor, meta = find_estimator(loaded_obj)
except FileNotFoundError:
    predictor, meta = (None, {"error": f"File {MODEL_PATH} not found."})
except Exception as e:
    predictor, meta = (None, {"error": str(e)})

st.sidebar.header("Model status")
if meta is None:
    st.sidebar.write("No model loaded.")
else:
    st.sidebar.write(meta)

st.write("**Debug note:** jika prediksi gagal, lihat `Model status` di sidebar untuk info struktur `.pkl`.")

st.markdown("---")
st.header("1) Predict from manual inputs (single-row)")
with st.form("manual_form"):
    st.write("Isi fitur sesuai modelmu. Jika tidak tahu fitur, gunakan CSV upload di bawah.")
    # example generic fields; user can add/remove as needed
    c1 = st.number_input("feature_1 (contoh)", value=0.0, step=1.0)
    c2 = st.number_input("feature_2 (contoh)", value=0.0, step=1.0)
    c3 = st.number_input("feature_3 (contoh)", value=0.0, step=1.0)
    submitted = st.form_submit_button("Predict single row")
if submitted:
    if predictor is None:
        st.error("Model belum ter-load atau tidak ditemukan fungsi predict. Periksa sidebar untuk detail.")
    else:
        X_manual = pd.DataFrame([{"feature_1": c1, "feature_2": c2, "feature_3": c3}])
        try:
            preds = predictor(X_manual)
            st.success("Prediksi:")
            st.write(preds)
        except Exception as e:
            st.exception(e)

st.markdown("---")
st.header("2) Upload CSV for batch prediction")
st.write("CSV harus memiliki header. Jika model memerlukan kolom khusus, pastikan kolom tersebut ada.")
uploaded = st.file_uploader("Upload CSV", type=["csv"])
expected_cols_input = st.text_input("Optional: sebutkan kolom yang model harapkan (comma-separated)", value="")
expected_cols = [c.strip() for c in expected_cols_input.split(",")] if expected_cols_input.strip() else None

if uploaded is not None:
    try:
        df = pd.read_csv(uploaded)
        st.write("Preview data (5 baris):")
        st.dataframe(df.head())
        if predictor is None:
            st.error("Model belum ter-load atau tidak ditemukan fungsi predict. Periksa sidebar untuk detail.")
        else:
            try:
                X_prepared = prepare_features_from_df(df, expected_cols)
                preds = predictor(X_prepared)
                # attach predictions
                out = df.copy()
                out["prediction"] = preds
                st.write("Hasil prediksi (preview):")
                st.dataframe(out.head())
                # allow download
                csv_bytes = out.to_csv(index=False).encode("utf-8")
                st.download_button("Download predictions CSV", data=csv_bytes, file_name="predictions.csv", mime="text/csv")
            except Exception as e:
                st.exception(e)
    except Exception as e:
        st.exception(e)

st.markdown("---")
st.header("Troubleshooting tips")
st.write("""
- Jika sidebar menunjukkan `type='dict'` dengan keys, perhatikan nama key yang menyimpan model (`model`, `estimator`, `pipeline`, dll).  
- Jika dict menyimpan `preprocessor` + `model`, app akan mencoba memanggil `preprocessor.transform(X)` lalu `model.predict(transformed_X)`.  
- Jika model membutuhkan urutan kolom tertentu, isi `Optional: sebutkan kolom ...` atau unggah CSV yang kolomnya cocok.  
- Jika masih error, kirimkan pesan `Model status` dari sidebar atau kirim snapshot struktur pickled object (kamu bisa copy `meta` atau keys) supaya aku bantu sesuaikan loader.
""")
