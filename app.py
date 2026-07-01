import re
import numpy as np
import streamlit as st
import joblib
import altair as alt
import pandas as pd

st.set_page_config(page_title="Smish · SMS Spam Inspector", page_icon="🛰️",
                   layout="centered", initial_sidebar_state="collapsed")

# ------------------------------------------------------------------ styling
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

:root{
  --ink:#14181f; --muted:#6b7280; --line:#e6e8ec;
  --safe:#1f9d6b; --safe-bg:#e7f6ef;
  --spam:#e5484d; --spam-bg:#fdecec;
  --accent:#6c5ce7; --surface:#ffffff; --bg:#f4f5f7;
}
.stApp{ background:var(--bg); }
.block-container{ max-width:760px; padding-top:2.2rem; }
html,body,[class*="css"]{ font-family:'Inter',sans-serif; color:var(--ink); }

.brandbar{ display:flex; align-items:center; gap:.7rem; margin-bottom:.2rem; }
.brand-dot{ width:34px; height:34px; border-radius:9px;
  background:linear-gradient(135deg,var(--accent),#8b7bff);
  display:grid; place-items:center; font-size:1.1rem; }
.brand-name{ font-family:'Space Grotesk',sans-serif; font-weight:700;
  font-size:1.5rem; letter-spacing:-.02em; }
.brand-sub{ color:var(--muted); font-size:.92rem; margin:.1rem 0 1.4rem; }

.card{ background:var(--surface); border:1px solid var(--line); border-radius:16px;
  padding:20px 22px; box-shadow:0 1px 3px rgba(20,24,31,.04); }
.eyebrow{ font-family:'Space Grotesk',sans-serif; font-size:.72rem; font-weight:600;
  letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin-bottom:.5rem; }

/* verdict */
.verdict{ border-radius:16px; padding:22px 24px; margin-bottom:14px; border:1px solid; }
.verdict.spam{ background:var(--spam-bg); border-color:#f5c2c4; }
.verdict.ham{  background:var(--safe-bg); border-color:#bfe6d4; }
.verdict-label{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:2rem;
  letter-spacing:-.02em; line-height:1; }
.verdict.spam .verdict-label{ color:var(--spam); }
.verdict.ham  .verdict-label{ color:var(--safe); }
.verdict-note{ color:var(--ink); opacity:.75; font-size:.95rem; margin-top:.5rem; }

/* meter */
.meter-track{ height:12px; border-radius:20px; background:#eceef1; overflow:hidden; margin-top:6px; }
.meter-fill{ height:100%; border-radius:20px;
  background:linear-gradient(90deg,#1f9d6b 0%,#e6b800 55%,#e5484d 100%); }
.meter-cap{ display:flex; justify-content:space-between; font-size:.78rem;
  color:var(--muted); margin-top:5px; font-family:'JetBrains Mono',monospace; }

/* message render */
.msg{ font-family:'JetBrains Mono',monospace; font-size:.96rem; line-height:1.7;
  background:#fbfbfc; border:1px dashed var(--line); border-radius:12px;
  padding:16px 18px; white-space:pre-wrap; word-break:break-word; }
.trigger{ background:#fde2e2; color:#b21f24; border-radius:5px; padding:1px 4px;
  font-weight:500; box-shadow:inset 0 -2px 0 #f3a9ab; }

/* chips */
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:4px; }
.chip{ font-family:'JetBrains Mono',monospace; font-size:.82rem; padding:5px 11px;
  border-radius:20px; background:#fdecec; color:var(--spam); border:1px solid #f5c2c4; }
.chip.none{ background:#eef2f7; color:var(--muted); border-color:var(--line); }

.stat{ font-family:'Space Grotesk',sans-serif; font-weight:700; font-size:1.5rem; }
.stat-l{ color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.08em; }
.footnote{ color:var(--muted); font-size:.82rem; margin-top:1.5rem; text-align:center; }
div[data-testid="stTextArea"] textarea{ font-family:'JetBrains Mono',monospace; font-size:.95rem;
  border-radius:12px; }
.stButton>button{ border-radius:11px; font-weight:600; font-family:'Space Grotesk',sans-serif;
  border:1px solid var(--line); }
.stButton>button[kind="primary"]{ background:var(--accent); border-color:var(--accent); }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------ model
@st.cache_resource
def load_model():
    b = joblib.load("sms_spam_model.joblib")
    return b["model"], b.get("metrics", {})

try:
    model, metrics = load_model()
except FileNotFoundError:
    st.error("**sms_spam_model.joblib** tidak ditemukan. Jalankan notebook Colab dulu, "
             "lalu taruh model di folder yang sama dengan app ini.")
    st.stop()

vec = model.named_steps["tfidf"]
clf = model.named_steps["clf"]
feat = np.array(vec.get_feature_names_out())
coef = clf.coef_[0]

def analyze(text):
    x = vec.transform([text])
    proba = float(model.predict_proba([text])[0][1])
    contrib = {}
    for j in x.nonzero()[1]:
        contrib[feat[j]] = x[0, j] * coef[j]
    spam_words = sorted([(w, c) for w, c in contrib.items() if c > 0],
                        key=lambda t: -t[1])
    return proba, spam_words

def highlight(text, spam_words):
    unis = {w for w, _ in spam_words if " " not in w}
    def rep(m):
        return f'<span class="trigger">{m.group(0)}</span>' if m.group(0).lower() in unis else m.group(0)
    return re.sub(r"[A-Za-z0-9']+", rep, text)

EXAMPLES = {
    "🎁 Contoh spam": "WINNER!! You've been selected for a FREE £1000 prize. Txt CLAIM to 80086 now to win. Reply STOP to cancel.",
    "💬 Contoh biasa": "Hey, are we still on for lunch tomorrow? Let me know what time works for you.",
    "🔗 Smishing": "URGENT: your bank account is locked. Verify your details now at http://secure-verify.co to avoid suspension.",
}

# ------------------------------------------------------------------ header
st.markdown("""
<div class="brandbar">
  <div class="brand-dot">🛰️</div>
  <div class="brand-name">Smish</div>
</div>
<div class="brand-sub">Tempel sebuah SMS — model akan memeriksa apakah itu <b>spam</b> dan menunjukkan <b>kenapa</b>.</div>
""", unsafe_allow_html=True)

if "msg" not in st.session_state:
    st.session_state.msg = ""

st.markdown('<div class="eyebrow">Pesan masuk</div>', unsafe_allow_html=True)
cols = st.columns(3)
for (label, txt), col in zip(EXAMPLES.items(), cols):
    if col.button(label, use_container_width=True):
        st.session_state.msg = txt

msg = st.text_area("SMS", key="msg", height=120, label_visibility="collapsed",
                   placeholder="Ketik atau tempel isi SMS di sini…")

go = st.button("🔍  Periksa pesan", type="primary", use_container_width=True)

# ------------------------------------------------------------------ result
if go and msg.strip():
    proba, spam_words = analyze(msg)
    is_spam = proba >= 0.5
    conf = proba if is_spam else 1 - proba

    kind = "spam" if is_spam else "ham"
    label = "SPAM" if is_spam else "AMAN"
    note = ("Pesan ini menunjukkan pola khas spam/penipuan. Jangan klik tautan atau balas nomor tak dikenal."
            if is_spam else
            "Pesan ini tampak seperti percakapan biasa. Tidak ada pola spam yang kuat terdeteksi.")

    st.markdown(f"""
    <div class="verdict {kind}">
      <div class="verdict-label">{'⚠️ ' if is_spam else '✅ '}{label}</div>
      <div class="verdict-note">{note}</div>
      <div class="meter-track"><div class="meter-fill" style="width:{proba*100:.0f}%"></div></div>
      <div class="meter-cap"><span>aman</span><span>skor spam: {proba*100:.1f}%</span><span>spam</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="eyebrow">Pesan yang diperiksa</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="msg">{highlight(msg, spam_words)}</div>', unsafe_allow_html=True)

    st.markdown('<div class="eyebrow" style="margin-top:1.1rem">Sinyal yang terdeteksi</div>',
                unsafe_allow_html=True)
    if spam_words:
        chips = "".join(f'<span class="chip">{w}</span>' for w, _ in spam_words[:10])
    else:
        chips = '<span class="chip none">tidak ada kata pemicu spam yang menonjol</span>'
    st.markdown(f'<div class="chips">{chips}</div>', unsafe_allow_html=True)

    if spam_words:
        df = pd.DataFrame(spam_words[:8], columns=["kata", "kontribusi"])
        chart = alt.Chart(df).mark_bar(color="#e5484d", cornerRadiusEnd=4).encode(
            x=alt.X("kontribusi:Q", title="dorongan ke arah spam"),
            y=alt.Y("kata:N", sort="-x", title=None),
            tooltip=["kata", alt.Tooltip("kontribusi:Q", format=".3f")]
        ).properties(height=min(30*len(df)+30, 300))
        st.markdown('<div class="eyebrow" style="margin-top:1.1rem">Bobot tiap kata</div>',
                    unsafe_allow_html=True)
        st.altair_chart(chart, use_container_width=True)

elif go:
    st.warning("Tulis atau tempel sebuah pesan dulu.")

# ------------------------------------------------------------------ stats
st.divider()
c1, c2, c3 = st.columns(3)
c1.markdown(f'<div class="stat">{metrics.get("spam_f1",0)*100:.1f}%</div>'
            f'<div class="stat-l">Spam F1</div>', unsafe_allow_html=True)
c2.markdown(f'<div class="stat">{metrics.get("accuracy",0)*100:.1f}%</div>'
            f'<div class="stat-l">Akurasi</div>', unsafe_allow_html=True)
c3.markdown(f'<div class="stat">{metrics.get("roc_auc",0):.3f}</div>'
            f'<div class="stat-l">ROC-AUC</div>', unsafe_allow_html=True)

st.markdown('<div class="footnote">Logistic Regression · TF-IDF · dilatih pada SMS Spam Collection (UCI, 5.572 pesan). '
            'Prediksi bersifat probabilistik dan bisa keliru.</div>', unsafe_allow_html=True)
