import json
from pathlib import Path

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정 (브라우저 탭 제목과 아이콘)
# ------------------------------------------------------------------
APP_TITLE = "뇌졸중 예측 실습실"
st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/gim589005/data/refs/heads/main/"
    "stroke%20(1).csv"
)
RECORD_DIR = Path(__file__).parent / "내_기록"
RECORD_FILE = RECORD_DIR / "기록.json"


# ------------------------------------------------------------------
# 데이터 불러오기
# ------------------------------------------------------------------
@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


def describe_values(s: pd.Series) -> str:
    """열의 '값의 종류'를 짧은 글로 설명한다."""
    values = s.dropna().unique()
    n = len(values)
    if n <= 6:
        return ", ".join(str(v) for v in sorted(values, key=str))
    if pd.api.types.is_numeric_dtype(s):
        return f"숫자 (최소 {s.min():g} ~ 최대 {s.max():g}, 서로 다른 값 {n}개)"
    return f"문자 (서로 다른 값 {n}개)"


# ------------------------------------------------------------------
# 내_기록 폴더의 json 저장 / 불러오기
# ------------------------------------------------------------------
def load_record() -> dict:
    record = {"meanings": {}, "source": ""}
    try:
        if RECORD_FILE.exists():
            data = json.loads(RECORD_FILE.read_text(encoding="utf-8"))
            record["meanings"] = dict(data.get("meanings", {}))
            record["source"] = str(data.get("source", ""))
    except (json.JSONDecodeError, OSError):
        pass  # 파일이 깨졌으면 빈 기록으로 시작
    return record


def save_record(record: dict) -> bool:
    try:
        RECORD_DIR.mkdir(parents=True, exist_ok=True)
        RECORD_FILE.write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return True
    except OSError:
        return False


def big_card(label: str, value: str) -> str:
    return f"""
    <div style="border:1px solid rgba(128,128,128,0.35); border-radius:14px;
                padding:22px 10px; text-align:center;">
        <div style="font-size:1rem; opacity:0.75; margin-bottom:6px;">{label}</div>
        <div style="font-size:2.8rem; font-weight:700; line-height:1.2;">{value}</div>
    </div>
    """


# ------------------------------------------------------------------
# 화면 시작
# ------------------------------------------------------------------
st.title(f"🧠 {APP_TITLE}")
st.caption("뇌졸중 환자 데이터가 어떤 모습인지 먼저 살펴봐요.")

try:
    df = load_data(DATA_URL)
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# 처음 실행(또는 새로고침) 때 저장된 기록을 한 번만 불러온다
if "saved" not in st.session_state:
    loaded = load_record()
    st.session_state.saved = loaded
    st.session_state.initial_meanings = dict(loaded["meanings"])
    st.session_state.initial_source = loaded["source"]

# ---- 1. 큰 숫자 카드 네 개 -------------------------------------------
total = len(df)
n_cols = df.shape[1]
n_stroke = int((df["stroke"] == 1).sum())
ratio = n_stroke / total * 100 if total else 0

c1, c2, c3, c4 = st.columns(4)
c1.markdown(big_card("전체 사람 수", f"{total:,}명"), unsafe_allow_html=True)
c2.markdown(big_card("열 개수", f"{n_cols}개"), unsafe_allow_html=True)
c3.markdown(big_card("stroke = 1인 사람 수", f"{n_stroke:,}명"), unsafe_allow_html=True)
c4.markdown(big_card("stroke = 1 비율", f"{ratio:.2f}%"), unsafe_allow_html=True)

# ---- 2. 열 설명표 ----------------------------------------------------
st.subheader("📋 열 설명표")
st.caption("‘우리말 뜻’ 칸을 더블클릭해서 교재를 보고 직접 채워 보세요.")

table = pd.DataFrame(
    {
        "열 이름": df.columns,
        "우리말 뜻": [
            st.session_state.initial_meanings.get(c, "") for c in df.columns
        ],
        "값의 종류": [describe_values(df[c]) for c in df.columns],
        "빈 값 개수": [int(df[c].isna().sum()) for c in df.columns],
    }
)

edited = st.data_editor(
    table,
    key="meaning_table",
    hide_index=True,
    use_container_width=True,
    disabled=["열 이름", "값의 종류", "빈 값 개수"],
    column_config={
        "우리말 뜻": st.column_config.TextColumn("우리말 뜻", width="medium"),
        "값의 종류": st.column_config.TextColumn("값의 종류", width="large"),
    },
)

# ---- 3. 처음 다섯 줄 -------------------------------------------------
st.subheader("👀 데이터 처음 다섯 줄")
st.dataframe(df.head(5), use_container_width=True, hide_index=True)

# ---- 4. 데이터 출처 --------------------------------------------------
st.subheader("📚 데이터 출처")
source_text = st.text_area(
    "교재에 나온 출처를 적어 주세요.",
    value=st.session_state.initial_source,
    key="source_text",
    height=140,
    placeholder="여기에 출처를 적고, 바깥을 클릭하면 저장돼요.",
)

# ---- 저장: 내용이 바뀌었을 때만 json 파일에 쓴다 ---------------------
new_record = {
    "meanings": {
        str(name): ("" if pd.isna(meaning) else str(meaning))
        for name, meaning in zip(edited["열 이름"], edited["우리말 뜻"])
    },
    "source": source_text,
}

if new_record != st.session_state.saved:
    if save_record(new_record):
        st.session_state.saved = new_record
        st.caption("💾 저장했어요. (내_기록/기록.json)")
    else:
        st.warning("저장하지 못했어요. 폴더 쓰기 권한을 확인해 주세요.")
else:
    st.caption("💾 적은 내용은 내_기록/기록.json 에 자동으로 저장돼요.")
