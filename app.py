import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix

# ------------------------------
# Load artifacts saved by notebook
# ------------------------------
ARTIFACTS_DIR = "artifacts"

best_models = pickle.load(open(f"{ARTIFACTS_DIR}/best_models.pkl", "rb"))
best_params = pickle.load(open(f"{ARTIFACTS_DIR}/best_params.pkl", "rb"))
tfidf = pickle.load(open(f"{ARTIFACTS_DIR}/tfidf.pkl", "rb"))
X_test = pickle.load(open(f"{ARTIFACTS_DIR}/X_test.pkl", "rb"))
y_test = pickle.load(open(f"{ARTIFACTS_DIR}/y_test.pkl", "rb"))

MODEL_DEFAULT = "LinearSVM" if "LinearSVM" in best_models else list(best_models.keys())[0]

def get_scores(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        return model.decision_function(X)
    else:
        return model.predict(X)

# ------------------------------
# Page config + Dark Neon styling
# ------------------------------
st.set_page_config(page_title="SMS Spam Detector", page_icon="💀", layout="wide")
st.markdown("""
<style>
body { background-color:#0D0D0D; color:#fff; }
textarea { background-color:#1C1C1C !important; color:#fff !important; }
.stButton > button {
  background: linear-gradient(90deg,#ff0084,#ff8c00);
  color:white; padding:10px 22px; border:none; border-radius:10px; font-weight:700; font-size:16px;
}
.stButton > button:hover { background: linear-gradient(90deg,#ff8c00,#ff0084); }
.metric-card{ background:#121212; border:1px solid #222; padding:16px; border-radius:12px; }
</style>
""", unsafe_allow_html=True)

st.title("📩 SMS Spam Detection — Tuned & Synced Dashboard")

tab1, tab2, tab3 = st.tabs(["💬 Test Message", "📊 Model Performance", "⚙️ Tuned Hyperparameters"])

# ------------------------------
# Tab 1 — Test message with chosen model
# ------------------------------
with tab1:
    colA, colB = st.columns([2,1])
    with colA:
        msg = st.text_area("Type a message to classify:", height=140, placeholder="e.g., Congratulations! You won a prize. Click here...")
    with colB:
        model_name = st.selectbox("Choose model", list(best_models.keys()), index=list(best_models.keys()).index(MODEL_DEFAULT))

    if st.button("Analyze"):
        if not msg.strip():
            st.warning("Please enter a message.")
        else:
            vec = tfidf.transform([msg])
            mdl = best_models[model_name]
            pred = mdl.predict(vec)[0]
            score = get_scores(mdl, vec)[0]
            conf = float(score if pred==1 else (1 - score))

            if pred == 1:
                st.error(f"🚫 SPAM — Confidence: {conf*100:.2f}%")
            else:
                st.success(f"✅ HAM (Safe) — Confidence: {conf*100:.2f}%")

# ------------------------------
# Tab 2 — Performance dashboard (uses X_test/y_test from notebook)
# ------------------------------
with tab2:
    st.subheader("ROC Curve Comparison")
    fig_roc = plt.figure(figsize=(8,6))
    for name, mdl in best_models.items():
        scores = get_scores(mdl, X_test)
        fpr, tpr, _ = roc_curve(y_test, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (Tuned Models)")
    plt.legend()
    st.pyplot(fig_roc)

    st.markdown("---")
    st.subheader("Confusion Matrices")
    for name, mdl in best_models.items():
        preds = mdl.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        fig_cm = plt.figure(figsize=(4,3))
        sns.heatmap(cm, annot=True, cmap="Blues", fmt="d")
        plt.title(f"{name} — Confusion Matrix")
        plt.xlabel("Predicted"); plt.ylabel("True")
        plt.tight_layout()
        st.pyplot(fig_cm)

# ------------------------------
# Tab 3 — Show tuned hyperparameters
# ------------------------------
with tab3:
    st.subheader("Best Hyperparameters (from GridSearchCV)")
    for name, params in best_params.items():
        st.markdown(f"**{name}**")
        st.json(params)
