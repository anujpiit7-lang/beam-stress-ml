import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon, Circle
import streamlit as st

st.set_page_config(page_title="Beam Predictor", page_icon="🏗️", layout="wide")
st.title("🏗️ Beam Stress & Deflection Predictor")
st.caption("Log-linear ML model trained on Euler–Bernoulli beam theory")

@st.cache_resource
def load_models():
    return (joblib.load("data/stress_model.pkl"),
            joblib.load("data/deflection_model.pkl"))
stress_model, defl_model = load_models()

materials = {"Steel": (200, 250), "Aluminium": (70, 270), "Timber": (10, 40)}  # (E GPa, yield MPa)

# ---------------- Sidebar ----------------
st.sidebar.header("Beam inputs")
L   = st.sidebar.slider("Length L (m)", 1.0, 5.0, 3.0)
P   = st.sidebar.slider("Load P (kN)", 1.0, 50.0, 20.0)
b   = st.sidebar.slider("Width b (m)", 0.05, 0.30, 0.10)
h   = st.sidebar.slider("Height h (m)", 0.10, 0.50, 0.20)
mat = st.sidebar.selectbox("Material", list(materials))
sup = st.sidebar.radio("Support", ["Simply supported", "Cantilever"])
E, yield_MPa = materials[mat]
cant = int(sup == "Cantilever")

# ---------------- ML prediction ----------------
def ml_predict(P_vals):
    P_vals = np.atleast_1d(P_vals).astype(float)
    n = len(P_vals)
    X = pd.DataFrame({
        "log_length_m": np.full(n, np.log(L)),
        "log_load_kN":  np.log(P_vals),
        "log_width_m":  np.full(n, np.log(b)),
        "log_height_m": np.full(n, np.log(h)),
        "log_E_GPa":    np.full(n, np.log(E)),
        "is_cantilever": np.full(n, cant),
    })
    return np.exp(stress_model.predict(X)), np.exp(defl_model.predict(X))

ml_s, ml_d = ml_predict(P)
ml_stress, ml_defl = ml_s[0], ml_d[0]

# ---------------- Theory along the beam ----------------
I  = b * h**3 / 12
Pn, En = P * 1e3, E * 1e9
x = np.linspace(0, L, 400)

if cant:   # fixed at x = 0, load at free end
    V  = np.full_like(x, P)
    Mx = -P * (L - x)
    y  = Pn * x**2 * (3*L - x) / (6 * En * I) * 1e3
    M_max, load_x = P * L, L
else:      # pinned-roller, load at centre
    V  = np.where(x < L/2, P/2, -P/2)
    Mx = np.where(x <= L/2, P*x/2, P*(L - x)/2)
    xm = np.minimum(x, L - x)
    y  = Pn * xm * (3*L**2 - 4*xm**2) / (48 * En * I) * 1e3
    M_max, load_x = P * L / 4, L / 2

th_stress = M_max * 1e3 * (h/2) / I / 1e6
th_defl   = y.max()

# ---------------- Metrics ----------------
c1, c2, c3 = st.columns(3)
c1.metric("Max Stress (ML)", f"{ml_stress:.2f} MPa",
          f"{(ml_stress - th_stress)/th_stress*100:+.2f}% vs theory", delta_color="off")
c2.metric("Max Deflection (ML)", f"{ml_defl:.3f} mm",
          f"{(ml_defl - th_defl)/th_defl*100:+.2f}% vs theory", delta_color="off")
fos = yield_MPa / ml_stress
c3.metric("Factor of Safety", f"{fos:.2f}")

if fos >= 1.5:   st.success(f"Safe ✅ — {mat} yield strength is {yield_MPa} MPa")
elif fos >= 1:   st.warning("Marginal ⚠️ — close to yield")
else:            st.error(f"Fails ❌ — stress exceeds {mat} strength ({yield_MPa} MPa)")

# ---------------- Beam diagram ----------------
def beam_diagram():
    fig, ax = plt.subplots(figsize=(10, 3.2))
    t = 0.06 * L
    ax.add_patch(Rectangle((0, 0), L, t, fc="#9ecae1", ec="k", lw=1.2))

    shape = y / max(y.max(), 1e-12) * 0.15 * L          # exaggerated
    ax.plot(x, t/2 - shape, "--", color="#d62728", lw=1.5,
            label="Deflected shape (exaggerated)")

    s = 0.05 * L
    if cant:
        ax.add_patch(Rectangle((-0.04*L, -0.2*L), 0.04*L, 0.4*L + t,
                               fc="lightgray", ec="k", hatch="///"))
    else:
        ax.add_patch(Polygon([(0, 0), (-s/2, -s), (s/2, -s)], fc="gray", ec="k"))
        ax.add_patch(Polygon([(L, 0), (L - s/2, -s), (L + s/2, -s)], fc="gray", ec="k"))
        for dx in (-s/3, 0, s/3):
            ax.add_patch(Circle((L + dx, -s - s/6), s/6, fc="white", ec="k"))

    ax.annotate("", xy=(load_x, t), xytext=(load_x, t + 0.22*L),
                arrowprops=dict(arrowstyle="-|>", color="red", lw=2.5))
    ax.text(load_x, t + 0.24*L, f"P = {P:.1f} kN", ha="center", color="red",
            fontsize=11, fontweight="bold")

    ax.annotate("", xy=(0, -0.27*L), xytext=(L, -0.27*L),
                arrowprops=dict(arrowstyle="<->", color="k"))
    ax.text(L/2, -0.33*L, f"L = {L:.2f} m", ha="center")

    ax.set_xlim(-0.12*L, 1.12*L); ax.set_ylim(-0.38*L, 0.36*L)
    ax.set_aspect("equal"); ax.axis("off")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    return fig

def cross_section():
    fig, ax = plt.subplots(figsize=(3, 3.2))
    ax.add_patch(Rectangle((-b/2, -h/2), b, h, fc="#9ecae1", ec="k", lw=1.5))
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.text(0, -h/2 - 0.04, f"b = {b*100:.0f} cm", ha="center")
    ax.text(b/2 + 0.02, 0, f"h = {h*100:.0f} cm", va="center")
    ax.set_xlim(-0.3, 0.35); ax.set_ylim(-0.33, 0.3)
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_title("Cross-section", fontsize=10)
    return fig

st.subheader("Beam diagram")
d1, d2 = st.columns([4, 1])
d1.pyplot(beam_diagram()); d2.pyplot(cross_section())

# ---------------- Graphs ----------------
def line_plot(xv, yv, ylabel, color, title):
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(xv, yv, color=color, lw=2)
    ax.fill_between(xv, yv, alpha=0.2, color=color)
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("Position along beam x (m)"); ax.set_ylabel(ylabel)
    ax.set_title(title); ax.grid(alpha=0.3)
    return fig, ax

st.subheader("Analysis")
tab1, tab2, tab3, tab4 = st.tabs(["Shear Force", "Bending Moment",
                                  "Deflection", "Stress vs Load"])

with tab1:
    fig, _ = line_plot(x, V, "Shear force V (kN)", "#2ca02c", "Shear Force Diagram")
    st.pyplot(fig)

with tab2:
    fig, _ = line_plot(x, Mx, "Bending moment M (kN·m)", "#1f77b4", "Bending Moment Diagram")
    st.pyplot(fig)

with tab3:
    fig, ax = line_plot(x, -y, "Deflection (mm)", "#d62728", "Deflection Curve")
    ax.plot(x[np.argmax(y)], -ml_defl, "ko", ms=8, label=f"ML max = {ml_defl:.3f} mm")
    ax.legend(); st.pyplot(fig)

with tab4:
    P_range = np.linspace(1, 50, 100)
    ml_range, _ = ml_predict(P_range)
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.plot(P_range, th_stress / P * P_range, "k-", lw=2, label="Theory")
    ax.plot(P_range[::5], ml_range[::5], "o", color="#ff7f0e", label="ML prediction")
    ax.axhline(yield_MPa, color="red", ls="--", label=f"{mat} yield ({yield_MPa} MPa)")
    ax.plot(P, ml_stress, "*", color="purple", ms=16, label="Current load")
    ax.set_xlabel("Load P (kN)"); ax.set_ylabel("Max stress (MPa)")
    ax.set_title("Stress vs Load"); ax.grid(alpha=0.3); ax.legend()
    st.pyplot(fig)