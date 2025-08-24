import time, streamlit as st

def _fmt(datestr):
    return datestr or "—"

def sidebar_branding(email="contact@skillnestedu.com", board="IB", expiry_str=None):
    st.sidebar.image("assets/skillnestlogo.png", use_container_width=True)
    st.sidebar.markdown(f"""
<div style="padding:10px;border-radius:12px;background:#F2F2F2">
  <div style="font-weight:700;color:#333;font-size:18px">SkillNestEdu</div>
  <div style="color:#333;font-size:13px;margin-top:6px;line-height:1.3">
    Licensed to <b>{email}</b><br/>Board: <b>{board}</b><br/>Valid till <b>{_fmt(expiry_str)}</b>
  </div>
</div>""", unsafe_allow_html=True)

def page_watermark(email="contact@skillnestedu.com", expiry_str=None):
    ex = _fmt(expiry_str)
    st.markdown(f"""
<style>
.block-container::before {{
  content: "© SkillNestEdu • {email} • Valid till {ex}";
  position: fixed; top:28%; left:-12%; transform: rotate(-30deg);
  color: rgba(0,0,0,0.07); font-size:4.2rem; font-weight:700;
  z-index:0; pointer-events:none;
}}
</style>""", unsafe_allow_html=True)
