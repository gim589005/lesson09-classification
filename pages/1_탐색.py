import pandas as pd
import plotly.express as px
import streamlit as st

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
st.set_page_config(page_title="탐색 - 뇌졸중 예측 실습실", page_icon="🔍", layout="wide")

DATA_URL = (
    "https://raw.githubusercontent.com/gim589005/data/refs/heads/main/"
    "stroke%20(1).csv"
)


@st.cache_data(show_spinner="데이터를 불러오는 중...")
def load_data(url: str) -> pd.DataFrame:
    return pd.read_csv(url, encoding="utf-8")


st.title("🔍 탐색")
st.caption("데이터를 그림과 표로 살펴봐요.")

try:
    df = load_data(DATA_URL)
except Exception as e:  # 네트워크 오류 등
    st.error(f"데이터를 불러오지 못했어요: {e}")
    st.stop()

# 뇌졸중 여부를 알아보기 쉬운 글자로 바꾼 열 (그림과 표에 사용)
GROUP_COL = "뇌졸중 경험"
GROUP_ORDER = ["겪지 않음", "겪음"]
df[GROUP_COL] = df["stroke"].map({0: GROUP_ORDER[0], 1: GROUP_ORDER[1]})

# ---- 1. 나이와 평균 혈당의 분포 --------------------------------------
st.subheader("📊 나이와 평균 혈당의 분포")

col_a, col_b = st.columns(2)

fig_age = px.histogram(
    df,
    x="age",
    nbins=30,
    title="나이 분포",
    labels={"age": "나이"},
)
fig_age.update_yaxes(title_text="사람 수")
col_a.plotly_chart(fig_age, use_container_width=True)

fig_glu = px.histogram(
    df,
    x="avg_glucose_level",
    nbins=40,
    title="평균 혈당 분포",
    labels={"avg_glucose_level": "평균 혈당"},
)
fig_glu.update_yaxes(title_text="사람 수")
col_b.plotly_chart(fig_glu, use_container_width=True)

# ---- 2. 뇌졸중 여부에 따른 비교 --------------------------------------
st.subheader("📦 뇌졸중을 겪은 사람과 겪지 않은 사람 비교")

col_c, col_d = st.columns(2)

fig_box_age = px.box(
    df,
    x=GROUP_COL,
    y="age",
    color=GROUP_COL,
    category_orders={GROUP_COL: GROUP_ORDER},
    title="나이 비교",
    labels={"age": "나이"},
)
fig_box_age.update_layout(showlegend=False)
col_c.plotly_chart(fig_box_age, use_container_width=True)

fig_box_glu = px.box(
    df,
    x=GROUP_COL,
    y="avg_glucose_level",
    color=GROUP_COL,
    category_orders={GROUP_COL: GROUP_ORDER},
    title="평균 혈당 비교",
    labels={"avg_glucose_level": "평균 혈당"},
)
fig_box_glu.update_layout(showlegend=False)
col_d.plotly_chart(fig_box_glu, use_container_width=True)

st.markdown("**두 그룹의 평균값**")
mean_table = (
    df.groupby(GROUP_COL)
    .agg(
        사람_수=("stroke", "size"),
        평균_나이=("age", "mean"),
        평균_혈당=("avg_glucose_level", "mean"),
    )
    .reindex(GROUP_ORDER)
    .reset_index()
    .rename(
        columns={
            GROUP_COL: "그룹",
            "사람_수": "사람 수",
            "평균_나이": "평균 나이",
            "평균_혈당": "평균 혈당",
        }
    )
)
mean_table["평균 나이"] = mean_table["평균 나이"].round(2)
mean_table["평균 혈당"] = mean_table["평균 혈당"].round(2)
st.dataframe(mean_table, use_container_width=True, hide_index=True)

# ---- 3. 고혈압·심장병에 따른 뇌졸중 비율 ------------------------------
st.subheader("❤️ 고혈압·심장병 여부에 따른 뇌졸중 비율")


def stroke_rate_bar(column: str, title: str):
    """column 값(0/1)별 뇌졸중 비율(%)을 막대그래프로 만든다."""
    rate = (
        df.groupby(column)["stroke"]
        .mean()
        .mul(100)
        .reset_index(name="뇌졸중 비율(%)")
    )
    rate["구분"] = rate[column].map({0: "없음", 1: "있음"})
    fig = px.bar(
        rate,
        x="구분",
        y="뇌졸중 비율(%)",
        color="구분",
        text_auto=".2f",
        category_orders={"구분": ["없음", "있음"]},
        title=title,
    )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(title_text=title.split(" ")[0] + " 여부")
    fig.update_yaxes(title_text="뇌졸중 비율 (%)")
    return fig


col_e, col_f = st.columns(2)
col_e.plotly_chart(
    stroke_rate_bar("hypertension", "고혈압 여부에 따른 뇌졸중 비율"),
    use_container_width=True,
)
col_f.plotly_chart(
    stroke_rate_bar("heart_disease", "심장병 여부에 따른 뇌졸중 비율"),
    use_container_width=True,
)

# ---- 4. bmi가 빈 사람들 ----------------------------------------------
st.subheader("🕳️ bmi가 비어 있는 사람들")

missing = df[df["bmi"].isna()]
n_missing = len(missing)
n_missing_stroke = int((missing["stroke"] == 1).sum())
n_total = len(df)
n_total_stroke = int((df["stroke"] == 1).sum())

bmi_table = pd.DataFrame(
    {
        "구분": ["bmi가 빈 사람", "전체"],
        "사람 수": [n_missing, n_total],
        "뇌졸중을 겪은 사람 수": [n_missing_stroke, n_total_stroke],
        "뇌졸중 비율(%)": [
            round(n_missing_stroke / n_missing * 100, 2) if n_missing else None,
            round(n_total_stroke / n_total * 100, 2) if n_total else None,
        ],
    }
)
st.dataframe(bmi_table, use_container_width=True, hide_index=True)

# ---- 5. 흡연 상태별 사람 수 ------------------------------------------
st.subheader("🚬 흡연 상태별 사람 수")

smoking_table = (
    df["smoking_status"]
    .value_counts(dropna=False)
    .rename_axis("흡연 상태")
    .reset_index(name="사람 수")
)
st.dataframe(smoking_table, use_container_width=True, hide_index=True)
