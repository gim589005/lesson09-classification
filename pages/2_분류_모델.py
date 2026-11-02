from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------------
ALL_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KO = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
BINARY = {"hypertension", "heart_disease"}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

COLOR_NEG = "#2a6fdb"   # 아님
COLOR_POS = "#d62728"   # 뇌졸중
LEAF_NEG = "#BBD6F5"
LEAF_POS = "#F5B7B1"

st.title("분류 모델")
st.caption("뇌졸중(stroke=1)을 양성으로 두고, 선택한 속성으로 뇌졸중 여부를 맞히는 두 모델을 비교합니다.")


# ------------------------------------------------------------------
# 데이터 읽기
# ------------------------------------------------------------------
REQUIRED = ["id", "stroke"] + ALL_FEATURES


def read_csv_any(source) -> pd.DataFrame:
    """인코딩이 달라도 읽히도록 여러 방식으로 시도한다."""
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp949"):
        try:
            if hasattr(source, "seek"):
                source.seek(0)
            return pd.read_csv(source, encoding=enc)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise last_err


def normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """열 이름의 공백·대소문자를 정리하고, 필요한 열이 모두 있으면 돌려준다."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not set(REQUIRED).issubset(df.columns):
        return None
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df


@st.cache_data
def find_data() -> tuple[pd.DataFrame | None, list[str]]:
    here = Path(__file__).resolve().parent
    roots = []
    for r in (here.parent, Path.cwd()):
        if r not in roots:
            roots.append(r)

    tried: list[str] = []
    seen = set()
    for root in roots:
        for path in sorted(root.rglob("*.csv")):
            if path in seen or any(p in {".venv", "venv", "site-packages", "node_modules"} for p in path.parts):
                continue
            seen.add(path)
            try:
                df = normalize(read_csv_any(path))
            except Exception as e:  # noqa: BLE001
                tried.append(f"{path} → 읽기 실패 ({type(e).__name__})")
                continue
            if df is not None:
                return df, tried
            tried.append(f"{path} → 필요한 열이 없음")
    return None, tried


df, tried = find_data()
if df is None:
    st.warning(
        "프로젝트 안에서 뇌졸중 데이터 CSV를 찾지 못했습니다. "
        "필요한 열: id, stroke, age, avg_glucose_level, bmi, hypertension, heart_disease"
    )
    with st.expander("살펴본 CSV 파일"):
        if tried:
            for line in tried:
                st.text(line)
        else:
            st.text("CSV 파일이 하나도 없었습니다.")
    uploaded = st.file_uploader("뇌졸중 데이터 CSV 올리기", type="csv")
    if uploaded is None:
        st.stop()
    df = normalize(read_csv_any(uploaded))
    if df is None:
        st.error("올린 파일에 필요한 열(id, stroke, age, avg_glucose_level, bmi, hypertension, heart_disease)이 없습니다.")
        st.stop()

# ------------------------------------------------------------------
# 입력 속성 고르기
# ------------------------------------------------------------------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KO[c],
)
if len(chosen) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개보다 적으면 모델과 그림을 만들 수 없습니다.")
    st.stop()

features = [c for c in ALL_FEATURES if c in chosen]

# ------------------------------------------------------------------
# 데이터 나누기: 번호 순 정렬 -> 열 명씩 묶음 -> 각 묶음의 앞 세 명이 테스트용
# ------------------------------------------------------------------
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3

train_df = data.loc[~is_test].copy()
test_df = data.loc[is_test].copy()

X_train = train_df[features].copy()
X_test = test_df[features].copy()
y_train = train_df["stroke"].astype(int).to_numpy()
y_test = test_df["stroke"].astype(int).to_numpy()

bmi_fill = None
if "bmi" in features:
    bmi_fill = X_train["bmi"].median()  # 훈련용의 중앙값
    X_train["bmi"] = X_train["bmi"].fillna(bmi_fill)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_fill)

st.write(
    f"전체 {len(data):,}명 중 훈련용 {len(train_df):,}명, 테스트용 {len(test_df):,}명입니다. "
    f"(뇌졸중은 훈련용 {int(y_train.sum()):,}명, 테스트용 {int(y_test.sum()):,}명)"
)
if bmi_fill is not None:
    st.write(f"체질량지수의 빈 칸은 훈련용의 중앙값 {bmi_fill:.1f}로 채웠습니다.")

# ------------------------------------------------------------------
# 모델 만들기
# ------------------------------------------------------------------
# 크기 맞추기는 훈련용으로만 학습한다. 로지스틱 회귀에만 쓰고,
# 의사결정트리는 크기 맞추기에 영향을 받지 않으므로 질문의 기준값을 원래 단위로
# 읽을 수 있도록 원래 값으로 학습한다(답은 동일하다).
scaler = StandardScaler().fit(X_train)
Z_train = scaler.transform(X_train)
Z_test = scaler.transform(X_test)

logit = LogisticRegression(max_iter=1000).fit(Z_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

acc = {
    "logit": (logit.score(Z_train, y_train), logit.score(Z_test, y_test)),
    "tree": (tree.score(X_train, y_train), tree.score(X_test, y_test)),
    "base": (base.score(X_train, y_train), base.score(X_test, y_test)),
}

# ------------------------------------------------------------------
# 정확도 카드
# ------------------------------------------------------------------
st.subheader("정확도")

CARD_CSS = """
<style>
.acc-card {border: 1px solid rgba(128,128,128,0.4); border-radius: 12px; padding: 16px 18px; height: 100%;}
.acc-name {font-size: 0.9rem; opacity: 0.85; min-height: 3.2em;}
.acc-big {font-size: 2.8rem; font-weight: 700; line-height: 1.2;}
.acc-big-label {font-size: 0.8rem; opacity: 0.7;}
.acc-small-row {display: flex; gap: 24px; margin-top: 10px; font-size: 0.85rem; opacity: 0.85;}
</style>
"""
st.markdown(CARD_CSS, unsafe_allow_html=True)


def card_html(name: str, train_acc: float, test_acc: float) -> str:
    return (
        '<div class="acc-card">'
        f'<div class="acc-name">{name}</div>'
        '<div class="acc-big-label">테스트 데이터의 정확도</div>'
        f'<div class="acc-big">{test_acc * 100:.1f}%</div>'
        '<div class="acc-small-row">'
        f"<div>훈련 {train_acc * 100:.1f}%</div>"
        f"<div>테스트 {test_acc * 100:.1f}%</div>"
        "</div></div>"
    )


c1, c2, c3 = st.columns(3)
c1.markdown(card_html(NAME_LOGIT, *acc["logit"]), unsafe_allow_html=True)
c2.markdown(card_html(NAME_TREE, *acc["tree"]), unsafe_allow_html=True)
c3.markdown(card_html(NAME_BASE, *acc["base"]), unsafe_allow_html=True)

# ------------------------------------------------------------------
# 산점도 + 로지스틱 회귀 0.5 경계선 + 의사결정트리 칸 칠하기
# ------------------------------------------------------------------
st.subheader("테스트 데이터 그림")

col_x, col_y = st.columns(2)
x_feat = col_x.selectbox("가로축", features, index=0, format_func=lambda c: KO[c])
y_options = [f for f in features if f != x_feat]
y_feat = col_y.selectbox("세로축", y_options, index=0, format_func=lambda c: KO[c])

# 두 축이 아닌 속성은 테스트 데이터의 중앙값에 세운다.
others = [f for f in features if f not in (x_feat, y_feat)]
medians = {f: float(X_test[f].median()) for f in others}


def value_range(col: str) -> tuple[float, float]:
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    span = hi - lo if hi > lo else 1.0
    return lo - 0.08 * span, hi + 0.08 * span


x_lo, x_hi = value_range(x_feat)
y_lo, y_hi = value_range(y_feat)
xs = np.linspace(x_lo, x_hi, 200)
ys = np.linspace(y_lo, y_hi, 200)
XX, YY = np.meshgrid(xs, ys)

grid = pd.DataFrame({x_feat: XX.ravel(), y_feat: YY.ravel()})
for f, m in medians.items():
    grid[f] = m
grid = grid[features]

logit_score = logit.decision_function(scaler.transform(grid)).reshape(XX.shape)  # 0이면 확률 0.5
tree_pred = tree.predict(grid).reshape(XX.shape)

fig = go.Figure()

# 의사결정트리가 나눈 칸(옅은 색)
fig.add_trace(
    go.Heatmap(
        x=xs, y=ys, z=tree_pred,
        zmin=0, zmax=1,
        colorscale=[[0, "#9ecae1"], [1, "#fcae91"]],
        opacity=0.35, showscale=False, hoverinfo="skip",
    )
)

# 로지스틱 회귀가 0.5로 가르는 선
fig.add_trace(
    go.Contour(
        x=xs, y=ys, z=logit_score,
        contours=dict(start=0, end=0, size=1, coloring="none"),
        line=dict(color="black", width=3),
        showscale=False, hoverinfo="skip",
    )
)

# 점 찍기(이진 속성은 겹치지 않게 그림에서만 살짝 흩뿌림)
rng = np.random.default_rng(0)


def jittered(col: str) -> np.ndarray:
    v = X_test[col].to_numpy(dtype=float)
    if col in BINARY:
        v = v + rng.uniform(-0.04, 0.04, size=len(v))
    return v


px, py = jittered(x_feat), jittered(y_feat)
for label, value, color in [("아님", 0, COLOR_NEG), ("뇌졸중", 1, COLOR_POS)]:
    m = y_test == value
    fig.add_trace(
        go.Scatter(
            x=px[m], y=py[m], mode="markers", name=f"실제 {label}",
            marker=dict(color=color, size=6, opacity=0.8, line=dict(width=0.5, color="white")),
        )
    )

# 범례용 표시(선, 칸 색)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode="lines", name="로지스틱 회귀 0.5 경계선",
               line=dict(color="black", width=3))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode="markers", name="의사결정트리 칸: 아님",
               marker=dict(symbol="square", size=12, color="#9ecae1"))
)
fig.add_trace(
    go.Scatter(x=[None], y=[None], mode="markers", name="의사결정트리 칸: 뇌졸중",
               marker=dict(symbol="square", size=12, color="#fcae91"))
)

fig.update_layout(
    xaxis=dict(title=KO[x_feat], range=[x_lo, x_hi]),
    yaxis=dict(title=KO[y_feat], range=[y_lo, y_hi]),
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    height=560,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig)

if others:
    fixed_text = ", ".join(f"{KO[f]} {medians[f]:g}" for f in others)
    st.write(f"두 축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: {fixed_text}")
else:
    st.write("고른 속성이 두 개뿐이라 따로 세워 둔 속성은 없습니다.")

if logit_score.min() > 0:
    st.write("로지스틱 회귀가 0.5로 가르는 선은 그림 밖에 있습니다. 그림 안은 모두 뇌졸중 확률이 0.5보다 큽니다.")
elif logit_score.max() < 0:
    st.write("로지스틱 회귀가 0.5로 가르는 선은 그림 밖에 있습니다. 그림 안은 모두 뇌졸중 확률이 0.5보다 작습니다.")

if BINARY & {x_feat, y_feat}:
    st.caption("고혈압·심장병은 0 또는 1이라 점이 겹치지 않도록 그림에서만 위치를 아주 조금 흩뿌렸습니다.")

# ------------------------------------------------------------------
# 의사결정트리 가지 그림
# ------------------------------------------------------------------
st.subheader("의사결정트리가 던진 질문")

t = tree.tree_
path = tree.decision_path(X_train)
node_total = np.asarray(path.sum(axis=0)).ravel().astype(int)
node_pos = np.asarray(path[y_train == 1].sum(axis=0)).ravel().astype(int)


def fmt_num(v: float) -> str:
    return f"{v:,.1f}"


def stat_text(i: int) -> str:
    n, p = int(node_total[i]), int(node_pos[i])
    rate = (p / n * 100) if n else 0.0
    return f"훈련 {n:,}명\\n뇌졸중 {p:,}명 ({rate:.1f}%)"


dot = [
    "digraph T {",
    'graph [bgcolor="white"];',
    'node [shape=box, style="rounded,filled", fontname="sans-serif", fontcolor="black", color="#555555"];',
    'edge [color="#444444", fontcolor="#222222", fontname="sans-serif"];',
]

leaf_count = 0
leaf_neg_count = 0
for i in range(t.node_count):
    left, right = t.children_left[i], t.children_right[i]
    if left == -1:  # 답을 내는 마디
        leaf_count += 1
        answer = int(tree.classes_[np.argmax(t.value[i][0])])
        if answer == 0:
            leaf_neg_count += 1
        name = "아님" if answer == 0 else "뇌졸중"
        color = LEAF_NEG if answer == 0 else LEAF_POS
        dot.append(f'n{i} [label="답: {name}\\n{stat_text(i)}", fillcolor="{color}"];')
    else:
        feat = features[t.feature[i]]
        thr = float(t.threshold[i])
        if feat in BINARY:
            question = f"{KO[feat]}이(가) 있나요?"
            left_label, right_label = "아니요", "예"  # 왼쪽은 값이 0(없음)
        else:
            question = f"{KO[feat]} ≤ {fmt_num(thr)} ?"
            left_label, right_label = "예", "아니요"
        dot.append(f'n{i} [label="{question}\\n{stat_text(i)}", fillcolor="#F2F2F2"];')
        dot.append(f'n{i} -> n{left} [label="{left_label}"];')
        dot.append(f'n{i} -> n{right} [label="{right_label}"];')
dot.append("}")

st.graphviz_chart("\n".join(dot))

used = sorted({int(f) for f in t.feature if f >= 0})
used_names = ", ".join(KO[features[i]] for i in used) if used else "없음(질문을 하나도 던지지 않았습니다)"

st.write(f"답을 내는 마디는 모두 {leaf_count}칸이고, 그중 {leaf_neg_count}칸이 '아님'이라고 답합니다.")
st.write(f"고른 속성 가운데 이 나무가 실제로 물은 것: {used_names}")
