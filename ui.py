from pathlib import Path

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CSS_DIR = BASE_DIR / "css"


def load_css(*filenames: str) -> None:
    """css/ 폴더의 파일들을 순서대로 읽어 하나의 <style> 태그로 적용한다.

    파일이 없어도 앱이 죽지 않도록 조용히 건너뛴다 (로컬/Colab 어디서든 동작).
    """
    css_chunks = []
    for filename in filenames:
        try:
            css_chunks.append((CSS_DIR / filename).read_text(encoding="utf-8"))
        except FileNotFoundError:
            continue

    if css_chunks:
        st.markdown(f"<style>{''.join(css_chunks)}</style>", unsafe_allow_html=True)
