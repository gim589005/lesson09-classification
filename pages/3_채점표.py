import numpy as np
import pandas as pd
import streamlit as st
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# ------------------------------------------------------------------
# 기본 설정 ('분류 모델' 페이지와 같다)
# ------------------------------------------------------------------
ALL_FEATURES = ["age", "avg_glucose_level", "bmi", "hypertension", "heart_disease"]
KO = {
    "age": "나이",
    "avg_glucose_level": "평균 혈당",
    "bmi": "체질량지수",
    "hypertension": "고혈압",
    "heart_disease": "심장병",
}
DEFAULT_FEATURES = ["age", "avg_glucose_level", "hypertension", "heart_disease"]

NAME_LOGIT = "로지스틱 회귀(확률로 답하는 모델)"
NAME_TREE = "의사결정트리(질문으로 답하는 모델)"
NAME_BASE = "다수 클래스 기준 모델(입력을 하나도 보지 않고 훈련용에서 많은 쪽으로만 답하는 모델)"

NA_TEXT = "계산할 수 없음"

st.title("채점표")
st.caption("뇌졸중(stroke=1)을 양성으로 두고, 테스트 데이터로 두 모델의 답을 채점합니다.")


# ------------------------------------------------------------------
# 데이터 읽기 ('분류 모델' 페이지와 같다)
# ------------------------------------------------------------------
DATA_URL = "https://raw.githubusercontent.com/gim589005/data/refs/heads/main/stroke.csv"
REQUIRED = ["id", "stroke"] + ALL_FEATURES


def normalize(df: pd.DataFrame) -> pd.DataFrame | None:
    """열 이름의 공백·대소문자를 정리하고, 필요한 열이 모두 있으면 돌려준다."""
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if not set(REQUIRED).issubset(df.columns):
        return None
    df["bmi"] = pd.to_numeric(df["bmi"], errors="coerce")
    return df


@st.cache_data
def load_data() -> pd.DataFrame:
    """메인 페이지와 같은 주소에서 UTF-8 csv를 읽는다. bmi 열의 빈 값은 그대로 남는다."""
    return pd.read_csv(DATA_URL, encoding="utf-8")


df = None
try:
    df = normalize(load_data())
except Exception as e:  # noqa: BLE001
    st.warning(f"데이터 주소에서 읽지 못했습니다 ({type(e).__name__}). 인터넷 연결을 확인하거나 CSV를 직접 올려 주세요.")
    uploaded = st.file_uploader("뇌졸중 데이터 CSV 올리기", type="csv")
    if uploaded is None:
        st.stop()
    df = normalize(pd.read_csv(uploaded, encoding="utf-8"))
if df is None:
    st.error("데이터에 필요한 열(id, stroke, age, avg_glucose_level, bmi, hypertension, heart_disease)이 없습니다.")
    st.stop()

# ------------------------------------------------------------------
# 입력 속성 고르기 ('분류 모델' 페이지와 같다)
# ------------------------------------------------------------------
chosen = st.multiselect(
    "입력으로 사용할 속성",
    options=ALL_FEATURES,
    default=DEFAULT_FEATURES,
    format_func=lambda c: KO[c],
)
if len(chosen) < 2:
    st.info("입력 속성을 두 개 이상 골라 주세요. 두 개보다 적으면 모델과 채점표를 만들 수 없습니다.")
    st.stop()

features = [c for c in ALL_FEATURES if c in chosen]

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

if "bmi" in features:
    bmi_fill = X_train["bmi"].median()  # 훈련용의 중앙값
    X_train["bmi"] = X_train["bmi"].fillna(bmi_fill)
    X_test["bmi"] = X_test["bmi"].fillna(bmi_fill)

# ------------------------------------------------------------------
# 모델 만들기 ('분류 모델' 페이지와 같다)
# ------------------------------------------------------------------
scaler = StandardScaler().fit(X_train)
Z_train = scaler.transform(X_train)
Z_test = scaler.transform(X_test)

logit = LogisticRegression(max_iter=1000).fit(Z_train, y_train)
tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=5, random_state=42).fit(X_train, y_train)
base = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)

pred = {
    "logit": logit.predict(Z_test).astype(int),
    "tree": tree.predict(X_test).astype(int),
    "base": base.predict(X_test).astype(int),
}
names = {"logit": NAME_LOGIT, "tree": NAME_TREE, "base": NAME_BASE}


# ------------------------------------------------------------------
# 채점 도구
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

    return {
        "정확도": show(accuracy),
        "재현율": show(recall),
        "정밀도": show(precision),
        "F1": show(f1),
    }


def cm_table(c: dict[str, int]) -> pd.DataFrame:
    tp, fn, fp, tn = c["TP"], c["FN"], c["FP"], c["TN"]
    total = tp + fn + fp + tn
    return pd.DataFrame(
        [
            [f"TP · {tp:,}명", f"FN · {fn:,}명", f"{tp + fn:,}명"],
            [f"FP · {fp:,}명", f"TN · {tn:,}명", f"{fp + tn:,}명"],
            [f"{tp + fp:,}명", f"{fn + tn:,}명", f"{total:,}명"],
        ],
        index=["실제 뇌졸중", "실제 아님", "합계"],
        columns=["예측 뇌졸중", "예측 아님", "합계"],
    )


cms = {k: confusion(y_test, p) for k, p in pred.items()}

# ------------------------------------------------------------------
# 혼동행렬
# ------------------------------------------------------------------
st.write(f"테스트용 {len(test_df):,}명 중 뇌졸중은 {int(y_test.sum()):,}명입니다.")

st.subheader("혼동행렬")
st.caption("행은 실제, 열은 예측입니다. TP·FN·FP·TN은 뇌졸중(양성)을 기준으로 붙였습니다.")
left, right = st.columns(2)
with left:
    st.markdown(f"**{NAME_LOGIT}**")
    st.table(cm_table(cms["logit"]))
with right:
    st.markdown(f"**{NAME_TREE}**")
    st.table(cm_table(cms["tree"]))

# ------------------------------------------------------------------
# 지표 비교표
# ------------------------------------------------------------------
st.subheader("네 가지 지표")
rows = {names[k]: metrics(cms[k]) for k in ("logit", "tree", "base")}
st.table(pd.DataFrame(rows).T)
st.caption(
    "정확도 = (TP+TN) ÷ 전체 · 재현율 = TP ÷ (TP+FN), 실제 뇌졸중인 사람 수로 나눕니다 · "
    "정밀도 = TP ÷ (TP+FP), 뇌졸중이라 예측한 사람 수로 나눕니다 · "
    "F1 = 2 × 정밀도 × 재현율 ÷ (정밀도 + 재현율). "
    "뇌졸중이라 예측한 사람이 0명이면 정밀도와 F1은 계산할 수 없습니다."
)

# ------------------------------------------------------------------
# 틀린 사람 목록
# ------------------------------------------------------------------
st.subheader("틀린 사람 목록")
model_key = st.radio(
    "어느 모델의 목록을 볼까요?",
    options=["logit", "tree"],
    format_func=lambda k: names[k],
    horizontal=True,
)
y_hat = pred[model_key]
LIMIT = 20


def person_list(mask: np.ndarray) -> tuple[pd.DataFrame, int]:
    people = test_df.loc[mask].sort_values(["age", "id"], ascending=[False, True])
    total = len(people)
    shown = people.head(LIMIT)
    table = pd.DataFrame(
        {
            "번호": shown["id"].astype(int).to_numpy(),
            "나이": [f"{a:g}" for a in shown["age"]],
            "평균 혈당": [f"{g:.1f}" for g in shown["avg_glucose_level"]],
            "고혈압 여부": ["예" if h == 1 else "아니요" for h in shown["hypertension"]],
            "심장병 여부": ["예" if h == 1 else "아니요" for h in shown["heart_disease"]],
        }
    )
    return table, total


missed, n_missed = person_list((y_test == 1) & (y_hat == 0))
wrong_guess, n_wrong = person_list((y_test == 0) & (y_hat == 1))


def list_title(label: str, total: int) -> str:
    title = f"{label} — 전체 {total:,}명"
    if total > LIMIT:
        title += f" (나이가 많은 순으로 앞 {LIMIT}명만 표시)"
    return title


left, right = st.columns(2)
with left:
    st.markdown(f"**{list_title('놓친 환자(실제 뇌졸중인데 아니라고 예측)', n_missed)}**")
    if n_missed:
        st.dataframe(missed, hide_index=True)
    else:
        st.write("해당하는 사람이 없습니다.")
with right:
    st.markdown(f"**{list_title('헛짚은 사람(아닌데 뇌졸중이라고 예측)', n_wrong)}**")
    if n_wrong:
        st.dataframe(wrong_guess, hide_index=True)
    else:
        st.write("해당하는 사람이 없습니다.")

c = cms[model_key]
st.write(
    f"왼쪽 목록의 전체 {n_missed:,}명은 혼동행렬의 FN 칸({c['FN']:,}명)과, "
    f"오른쪽 목록의 전체 {n_wrong:,}명은 FP 칸({c['FP']:,}명)과 같습니다."
)
