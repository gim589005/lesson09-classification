from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정 ('분류 모델'·'채점표' 페이지와 같다)
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

NA_TEXT = "계산할 수 없음"
COLOR_NEG = "#2a6fdb"     # 실제 아님
COLOR_POS = "#d62728"     # 실제 뇌졸중
COLOR_BEFORE = "#111111"  # 가중치를 맞추기 전 경계
COLOR_AFTER = "#2ca02c"   # 가중치를 맞춘 뒤 경계

st.title("성능 높이기")

# ------------------------------------------------------------------
# 입력 속성 고르기 (화면 맨 위, '분류 모델' 페이지와 같다)
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

st.caption(
    "뇌졸중(stroke=1)을 양성으로 두고, 뇌졸중과 아님의 가중치를 맞추기 전과 맞춘 뒤를 비교합니다. "
    "가중치를 맞추면 드문 쪽(뇌졸중) 한 명의 무게가 인원수에 반비례해 더 커집니다."
)


# ------------------------------------------------------------------
# 데이터 읽기 ('채점표' 페이지와 같다)
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
# 데이터 나누기: 번호 순 정렬 -> 열 명씩 묶음 -> 각 묶음의 앞 세 명이 테스트용
# ------------------------------------------------------------------
data = df.sort_values("id").reset_index(drop=True)
is_test = (np.arange(len(data)) % 10) < 3

train_df = data.loc[~is_test].copy()
test_df = data.loc[is_test].copy().reset_index(drop=True)

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
# 모델 만들기: 가중치를 맞추기 전(기본) / 맞춘 뒤(class_weight="balanced")
# 크기 맞추기는 훈련용으로만 학습해 로지스틱 회귀에만 쓴다.
# 의사결정트리는 원래 값으로 학습한다(크기 맞추기에 영향을 받지 않는다).
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
Z_train = scaler.transform(X_train)
Z_test = scaler.transform(X_test)


def make_logit(weight):
    return LogisticRegression(max_iter=1000, class_weight=weight).fit(Z_train, y_train)


def make_tree(weight):
    return DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=5, random_state=42, class_weight=weight
    ).fit(X_train, y_train)


logit_before, logit_after = make_logit(None), make_logit("balanced")
tree_before, tree_after = make_tree(None), make_tree("balanced")

pred = {
    "logit": (logit_before.predict(Z_test).astype(int), logit_after.predict(Z_test).astype(int)),
    "tree": (tree_before.predict(X_test).astype(int), tree_after.predict(X_test).astype(int)),
}


# ------------------------------------------------------------------
# 채점 도구 ('채점표' 페이지와 같다)
# ------------------------------------------------------------------
def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    return {
        "TP": int(((y_true == 1) & (y_pred == 1)).sum()),
        "FN": int(((y_true == 1) & (y_pred == 0)).sum()),
        "FP": int(((y_true == 0) & (y_pred == 1)).sum()),
        "TN": int(((y_true == 0) & (y_pred == 0)).sum()),
    }


def pct(v: float) -> str:
    return f"{v * 100:.1f}%"


def metrics(c: dict[str, int]) -> dict[str, str]:
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    actual_pos = tp + fn      # 재현율의 분모: 실제 뇌졸중인 사람 수
    predicted_pos = tp + fp   # 정밀도의 분모: 뇌졸중이라 예측한 사람 수

    accuracy = (tp + tn) / total if total else None
    recall = tp / actual_pos if actual_pos else None
    precision = tp / predicted_pos if predicted_pos else None

    if precision is None or recall is None:
        f1 = None
    elif precision + recall == 0:
        f1 = 0.0  # 맞힌 양성이 하나도 없으면 F1은 0으로 둔다
    else:
        f1 = 2 * precision * recall / (precision + recall)

    def show(v):
        return NA_TEXT if v is None else pct(v)

    return {"정확도": show(accuracy), "재현율": show(recall), "정밀도": show(precision), "F1": show(f1)}


def before_after_table(c0: dict[str, int], c1: dict[str, int]) -> pd.DataFrame:
    m0, m1 = metrics(c0), metrics(c1)
    labels = {
        "TP": "TP (찾아낸 환자)",
        "FN": "FN (놓친 환자)",
        "FP": "FP (헛짚은 사람)",
        "TN": "TN (아님을 맞힌 사람)",
    }
    rows = {labels[k]: (f"{c0[k]:,}명", f"{c1[k]:,}명") for k in ("TP", "FN", "FP", "TN")}
    for k in ("정확도", "재현율", "정밀도", "F1"):
        rows[k] = (m0[k], m1[k])
    return pd.DataFrame(rows, index=["가중치 맞추기 전", "가중치 맞춘 뒤"]).T


# ------------------------------------------------------------------
# 전·후 비교표
# ------------------------------------------------------------------
st.subheader("가중치를 맞추기 전과 맞춘 뒤")
st.caption(
    "TP·FN·FP·TN은 테스트 데이터의 사람 수입니다. 재현율 = TP ÷ (TP+FN), 정밀도 = TP ÷ (TP+FP), "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). 뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없습니다."
)

left, right = st.columns(2)
for col, key, name in [(left, "logit", NAME_LOGIT), (right, "tree", NAME_TREE)]:
    c0, c1 = confusion(y_test, pred[key][0]), confusion(y_test, pred[key][1])
    with col:
        st.markdown(f"**{name}**")
        st.table(before_after_table(c0, c1))
        st.write(
            f"찾아낸 환자 {c0['TP']:,}명 → {c1['TP']:,}명, "
            f"놓친 환자 {c0['FN']:,}명 → {c1['FN']:,}명, "
            f"헛짚은 사람 {c0['FP']:,}명 → {c1['FP']:,}명"
        )


# ------------------------------------------------------------------
# 그림 도구
# ------------------------------------------------------------------
def value_range(col: str) -> tuple[float, float]:
    lo, hi = float(X_test[col].min()), float(X_test[col].max())
    span = hi - lo if hi > lo else 1.0
    return lo - 0.08 * span, hi + 0.08 * span


def jittered(col: str, rng: np.random.Generator) -> np.ndarray:
    v = X_test[col].to_numpy(dtype=float)
    if col in BINARY:  # 겹치지 않도록 그림에서만 살짝 흩뿌림
        v = v + rng.uniform(-0.04, 0.04, size=len(v))
    return v


def fixed_text(axes: list[str]) -> tuple[dict[str, float], str]:
    others = [f for f in features if f not in axes]
    medians = {f: float(X_test[f].median()) for f in others}
    if others:
        text = "축이 아닌 속성은 테스트 데이터의 중앙값에 세워 두고 계산했습니다: " + ", ".join(
            f"{KO[f]} {medians[f]:g}" for f in others
        )
    else:
        text = "고른 속성이 모두 축이라 따로 세워 둔 속성은 없습니다."
    return medians, text


def out_of_box_text(score: np.ndarray, label: str, shape: str) -> str | None:
    if score.min() > 0:
        return f"{label}의 0.5 {shape}은(는) 그림 밖에 있습니다. 그림 안은 모두 뇌졸중 확률이 0.5보다 큽니다."
    if score.max() < 0:
        return f"{label}의 0.5 {shape}은(는) 그림 밖에 있습니다. 그림 안은 모두 뇌졸중 확률이 0.5보다 작습니다."
    return None


# ------------------------------------------------------------------
# 2차원 산점도 + 가중치 전·후 0.5 경계선 (로지스틱 회귀)
# ------------------------------------------------------------------
st.subheader("2차원 그림: 0.5 경계선")
st.caption(f"경계선은 {NAME_LOGIT}이 뇌졸중 확률 0.5로 가르는 선입니다.")

col_x, col_y = st.columns(2)
x_feat = col_x.selectbox("가로축", features, index=0, format_func=lambda c: KO[c], key="x2")
y_feat = col_y.selectbox(
    "세로축", [f for f in features if f != x_feat], index=0, format_func=lambda c: KO[c], key="y2"
)

medians2, fixed2 = fixed_text([x_feat, y_feat])
x_lo, x_hi = value_range(x_feat)
y_lo, y_hi = value_range(y_feat)
xs = np.linspace(x_lo, x_hi, 200)
ys = np.linspace(y_lo, y_hi, 200)
XX, YY = np.meshgrid(xs, ys)

grid = pd.DataFrame({x_feat: XX.ravel(), y_feat: YY.ravel()})
for f, m in medians2.items():
    grid[f] = m
Zg = scaler.transform(grid[features])
score_before = logit_before.decision_function(Zg).reshape(XX.shape)  # 0이면 확률 0.5
score_after = logit_after.decision_function(Zg).reshape(XX.shape)

fig2 = go.Figure()
rng = np.random.default_rng(0)
px, py = jittered(x_feat, rng), jittered(y_feat, rng)
for label, value, color in [("아님", 0, COLOR_NEG), ("뇌졸중", 1, COLOR_POS)]:
    m = y_test == value
    fig2.add_trace(
        go.Scatter(
            x=px[m], y=py[m], mode="markers", name=f"실제 {label}",
            marker=dict(color=color, size=6, opacity=0.75, line=dict(width=0.5, color="white")),
        )
    )
for score, color, label in [
    (score_before, COLOR_BEFORE, "가중치 맞추기 전 0.5 경계선"),
    (score_after, COLOR_AFTER, "가중치 맞춘 뒤 0.5 경계선"),
]:
    fig2.add_trace(
        go.Contour(
            x=xs, y=ys, z=score,
            contours=dict(start=0, end=0, size=1, coloring="none"),
            line=dict(color=color, width=3),
            showscale=False, hoverinfo="skip",
        )
    )
    fig2.add_trace(go.Scatter(x=[None], y=[None], mode="lines", name=label, line=dict(color=color, width=3)))

fig2.update_layout(
    xaxis=dict(title=KO[x_feat], range=[x_lo, x_hi]),
    yaxis=dict(title=KO[y_feat], range=[y_lo, y_hi]),
    legend=dict(orientation="h", yanchor="top", y=-0.2, xanchor="left", x=0),
    height=560,
    margin=dict(l=10, r=10, t=10, b=10),
)
st.plotly_chart(fig2)

st.write(fixed2)
for score, label in [(score_before, "가중치를 맞추기 전"), (score_after, "가중치를 맞춘 뒤")]:
    msg = out_of_box_text(score, label, "경계선")
    if msg:
        st.write(msg)
if BINARY & {x_feat, y_feat}:
    st.caption("고혈압·심장병은 0 또는 1이라 점이 겹치지 않도록 그림에서만 위치를 아주 조금 흩뿌렸습니다.")

# ------------------------------------------------------------------
# 3차원 산점도 + 가중치 전·후 0.5 평면 (속성을 셋 이상 골랐을 때)
# ------------------------------------------------------------------
st.subheader("3차원 그림: 0.5 평면")
if len(features) < 3:
    st.info("입력 속성을 세 개 이상 고르면 마우스로 돌려 볼 수 있는 3차원 그림이 나옵니다.")
    st.stop()

st.caption(f"평면은 {NAME_LOGIT}이 뇌졸중 확률 0.5로 가르는 면입니다. 그림은 마우스로 끌어서 돌려 볼 수 있습니다.")
c_x, c_y, c_z = st.columns(3)
x3 = c_x.selectbox("가로축 (3차원)", features, index=0, format_func=lambda c: KO[c], key="x3")
y3 = c_y.selectbox(
    "세로축 (3차원)", [f for f in features if f != x3], index=0, format_func=lambda c: KO[c], key="y3"
)
z3 = c_z.selectbox(
    "깊이축 (3차원)", [f for f in features if f not in (x3, y3)], index=0, format_func=lambda c: KO[c], key="z3"
)

medians3, fixed3 = fixed_text([x3, y3, z3])
rx, ry, rz = value_range(x3), value_range(y3), value_range(z3)
N3 = 20
gx = np.linspace(rx[0], rx[1], N3)
gy = np.linspace(ry[0], ry[1], N3)
gz = np.linspace(rz[0], rz[1], N3)
GX, GY, GZ = np.meshgrid(gx, gy, gz, indexing="ij")

grid3 = pd.DataFrame({x3: GX.ravel(), y3: GY.ravel(), z3: GZ.ravel()})
for f, m in medians3.items():
    grid3[f] = m
Zg3 = scaler.transform(grid3[features])
s3_before = logit_before.decision_function(Zg3)
s3_after = logit_after.decision_function(Zg3)

fig3 = go.Figure()
rng3 = np.random.default_rng(1)
tx, ty, tz = jittered(x3, rng3), jittered(y3, rng3), jittered(z3, rng3)
for label, value, color in [("아님", 0, COLOR_NEG), ("뇌졸중", 1, COLOR_POS)]:
    m = y_test == value
    fig3.add_trace(
        go.Scatter3d(
            x=tx[m], y=ty[m], z=tz[m], mode="markers", name=f"실제 {label}",
            marker=dict(color=color, size=3, opacity=0.8),
        )
    )
for score, color, label in [
    (s3_before, COLOR_BEFORE, "가중치 맞추기 전 0.5 평면"),
    (s3_after, COLOR_AFTER, "가중치 맞춘 뒤 0.5 평면"),
]:
    fig3.add_trace(
        go.Isosurface(
            x=grid3[x3], y=grid3[y3], z=grid3[z3], value=score,
            isomin=0, isomax=0, surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=[[0, color], [1, color]],
            opacity=0.45, showscale=False, hoverinfo="skip",
        )
    )
    # 범례용 표시
    fig3.add_trace(
        go.Scatter3d(
            x=[None], y=[None], z=[None], mode="markers", name=label,
            marker=dict(symbol="square", size=6, color=color),
        )
    )

fig3.update_layout(
    scene=dict(
        xaxis=dict(title=KO[x3], range=list(rx)),
        yaxis=dict(title=KO[y3], range=list(ry)),
        zaxis=dict(title=KO[z3], range=list(rz)),
    ),
    legend=dict(orientation="h", yanchor="top", y=-0.02, xanchor="left", x=0),
    height=650,
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig3)

st.write(fixed3)
for score, label in [(s3_before, "가중치를 맞추기 전"), (s3_after, "가중치를 맞춘 뒤")]:
    msg = out_of_box_text(score, label, "평면")
    if msg:
        st.write(msg)
if BINARY & {x3, y3, z3}:
    st.caption("고혈압·심장병은 0 또는 1이라 점이 겹치지 않도록 그림에서만 위치를 아주 조금 흩뿌렸습니다.")
