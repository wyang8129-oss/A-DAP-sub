import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import matplotlib
import platform
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ===============================================
# 한글 폰트 설정
# ===============================================
import matplotlib
import matplotlib.pyplot as plt
import platform
import subprocess
import os

# ----------------------------
# 1. 한글 폰트 설치 (Cloud 환경용)
# ----------------------------
if platform.system() == "Linux":
    # NanumGothic 설치
    subprocess.run(["apt-get", "update"])
    subprocess.run(["apt-get", "install", "-y", "fonts-nanum"])
    matplotlib.rc('font', family='NanumGothic')

elif platform.system() == "Windows":
    matplotlib.rc('font', family='Malgun Gothic')
elif platform.system() == "Darwin":
    matplotlib.rc('font', family='AppleGothic')

# 마이너스 기호 깨짐 방지
matplotlib.rc('axes', unicode_minus=False)

#if platform.system() == 'Windows':
#    matplotlib.rc('font', family='Malgun Gothic')
#elif platform.system() == 'Darwin':
#    matplotlib.rc('font', family='AppleGothic')
#else:
#    matplotlib.rc('font', family='NanumGothic')
#matplotlib.rc('axes', unicode_minus=False)

st.set_page_config(page_title="토마토 생육·수확 통합 분석", layout="wide")
st.title("🍅 토마토 생육 + 수확 데이터 통합 분석 대시보드")
#st.markdown("---")

# -------------------------
# CSS / UI 스타일 적용
# -------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700&display=swap');
    html, body, [class*="css"]  {
        font-family: 'Noto Sans KR', sans-serif;
    }
    .stApp {
        background-color: #fbfbfd;
    }
    .card {
        background: #ffffff;
        padding: 12px;
        border-radius: 10px;
        box-shadow: 0 1px 6px rgba(32,33,36,0.08);
        margin-bottom: 12px;
    }
    h1, h2, h3 { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# 🌱 생육 데이터 처리
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("🌱 생육 데이터 처리")
#st.markdown('</div>', unsafe_allow_html=True)

# ---- top controls layout ----
with st.container():
    c1, c2, c3 = st.columns([2, 2, 1])
    with c1:
        growth_file = st.file_uploader("📂 생육 데이터 업로드 (CSV)", type=["csv"], key="growth")
    with c2:
        fill_option = st.selectbox("결측치 처리 방법 선택", ["없음", "0", "평균값", "최빈값"], index=0)
    with c3:
        marker_size = st.slider("마커 크기", min_value=1, max_value=20, value=6)

if not growth_file:
    st.info("CSV 파일을 업로드하면 시각화/집계 기능을 사용할 수 있습니다.")
    st.stop()

# -----------------------------
# 데이터 불러오기 및 표시
# -----------------------------
growth_df = pd.read_csv(growth_file)
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 업로드된 생육 데이터 미리보기")
st.dataframe(growth_df.head())
st.markdown('</div>', unsafe_allow_html=True)

# -----------------------------
# 개체번호 필터링
# -----------------------------
if "개체번호" not in growth_df.columns:
    st.error("❌ 생육 데이터에 '개체번호' 컬럼이 필요합니다.")
    st.stop()

unique_ids = growth_df["개체번호"].unique().tolist()
selected_ids = st.multiselect("분석할 개체번호 선택", unique_ids, default=unique_ids)

if len(selected_ids) == 0:
    st.info("분석할 개체를 하나 이상 선택하세요.")
    st.stop()

growth_df = growth_df[growth_df["개체번호"].isin(selected_ids)]

# -----------------------------
# 결측치 처리
# -----------------------------
if fill_option != "없음":
    for col in growth_df.columns:
        if growth_df[col].isnull().sum() > 0:
            if fill_option == "0":
                growth_df[col] = growth_df[col].fillna(0)
            elif fill_option == "평균값" and pd.api.types.is_numeric_dtype(growth_df[col]):
                growth_df[col] = growth_df[col].fillna(growth_df[col].mean())
            elif fill_option == "최빈값":
                growth_df[col] = growth_df[col].fillna(growth_df[col].mode()[0])

# 날짜 변환
growth_df["조사일자"] = pd.to_datetime(growth_df["조사일자"], errors="coerce")
if growth_df["조사일자"].isna().all():
    st.error("조사일자 날짜 변환 실패. 날짜 형식을 확인하세요.")
    st.stop()

# ============================================================
# 시각화 옵션
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 시계열 시각화 옵션")

numeric_cols = [
    col for col in growth_df.columns
    if pd.api.types.is_numeric_dtype(growth_df[col]) and col not in ["개체번호"]
]

selected_features = st.multiselect("시계열로 볼 생육 지표 선택", numeric_cols)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------
# 🔍 이동평균 기반 이상치 최초 발생 탐지 함수
# ------------------------------------------------------------
def detect_first_outlier_rolling(df, feature, selected_ids):

    df = df.sort_values("조사일자").copy()

    mean_by_date = (
        df.groupby("조사일자")[feature]
        .mean()
        .reset_index()
        .sort_values("조사일자")
    )

    mean_by_date["rolling_mean"] = (
        mean_by_date[feature].rolling(window=3, min_periods=1).mean()
    )

    df = pd.merge(
        df,
        mean_by_date[["조사일자", "rolling_mean"]],
        on="조사일자",
        how="left"
    )

    results = {}

    for cid in selected_ids:
        sub = df[df["개체번호"] == cid].sort_values("조사일자").copy()
        sub["value"] = sub[feature].fillna(0)
        sub["rm"] = sub["rolling_mean"].fillna(0)

        # 첫 조사일자 0은 정상 처리
        if len(sub) > 0 and sub.iloc[0]["value"] == 0:
            first_zero_date = sub.iloc[0]["조사일자"]
        else:
            first_zero_date = None

        sub["is_outlier"] = False
        zero_mask = (sub["value"] == 0)

        if first_zero_date is not None:
            idx0 = sub.index[0]
            zero_mask.loc[idx0] = False

        inf_mask = (sub["rm"] == 0) & (sub["value"] > 0)
        normal_out_mask = sub["value"] > (sub["rm"] * 5)

        sub["is_outlier"] = zero_mask | inf_mask | normal_out_mask
        first_out = sub[sub["is_outlier"]]

        if len(first_out) > 0:
            first = first_out.iloc[0]
            results[cid] = {
                "개체번호": cid,
                "조사일자": first["조사일자"],
                "값": first["value"],
                "rolling_mean": first["rm"]
            }
        else:
            results[cid] = None

    return results


# ============================================================
# 📈 개체별 시계열 그래프 (2열 구성)
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 개체별 시계열 그래프 (2열 구성)")

if selected_features:

    graph_cols = st.columns(2)
    color_cycle = plt.cm.tab20.colors
    color_map = {cid: color_cycle[i % len(color_cycle)] for i, cid in enumerate(selected_ids)}

    for idx, feature in enumerate(selected_features):
        outlier_info = detect_first_outlier_rolling(growth_df, feature, selected_ids)
        col_widget = graph_cols[idx % 2]

        with col_widget:
            fig, ax = plt.subplots(figsize=(7, 4))

            for cid in selected_ids:
                sub_df = growth_df[growth_df["개체번호"] == cid].sort_values("조사일자")
                if sub_df.empty:
                    continue

                ax.plot(
                    sub_df["조사일자"],
                    sub_df[feature],
                    marker='o',
                    markersize=marker_size,
                    label=f"{cid}",
                    color=color_map[cid]
                )

                out = outlier_info[cid]
                if out is not None:
                    out_date = out["조사일자"]
                    out_value = out["값"]

                    ax.scatter(
                        out_date, out_value,
                        s=140, color="red",
                        edgecolors="black", zorder=5
                    )
                    ax.axvline(out_date, color="red", linestyle="--", alpha=0.5)

                    ax.text(
                        out_date, out_value,
                        f"\n이상치 시작\n{out_date.date()}\n개체번호: {cid}",
                        color="red", fontsize=8,
                        ha="left", va="bottom"
                    )

            ax.set_title(f"{feature} 변화 추이", fontsize=12)
            ax.set_xlabel("조사일자")
            ax.set_ylabel(feature)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

            ax.grid(alpha=0.25)
            ax.legend(title="개체번호", fontsize=8, ncol=1)

            plt.tight_layout()
            st.pyplot(fig)

else:
    st.info("시계열로 표시할 지표를 선택하세요.")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 📈 개체별 박스플롯 그래프
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📦 개체별 생육 지표 박스플롯")

if selected_features:

    box_cols = st.columns(2)

    for idx, feature in enumerate(selected_features):

        col_widget = box_cols[idx % 2]

        with col_widget:
            fig, ax = plt.subplots(figsize=(7, 4))

            box_data = []
            labels = []

            for cid in selected_ids:
                sub_df = growth_df[growth_df["개체번호"] == cid]
                values = sub_df[feature].dropna()

                if len(values) > 0:
                    box_data.append(values)
                    labels.append(str(cid))

            if len(box_data) > 0:
                bp = ax.boxplot(
                    box_data,
                    labels=labels,
                    patch_artist=True,
                    showmeans=True
                )

                # 색상 적용
                colors = plt.cm.tab20.colors
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)

                ax.set_title(f"{feature} 개체별 분포", fontsize=12)
                ax.set_xlabel("개체번호")
                ax.set_ylabel(feature)
                ax.grid(alpha=0.3)

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

            else:
                st.info(f"{feature} 데이터가 부족합니다.")

else:
    st.info("박스플롯을 표시할 지표를 선택하세요.")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 📉 ANOVA 개체 간 유의성 검정 (지표별 2열 세로 정렬)
# ============================================================

st.markdown('<div class="card">', unsafe_allow_html=True)
st.header("📉 ANOVA 개체 간 유의성 검정")

if selected_features:

    # 🔹 지표를 2열로 배치
    feature_cols = st.columns(2)

    for idx, feature in enumerate(selected_features):

        col = feature_cols[idx % 2]

        with col:

            st.subheader(f"📌 [{feature} 지표]")

            feature_df = growth_df[["개체번호", feature]].dropna()

            groups = []
            summary_data = []

            for cid in selected_ids:
                group_values = feature_df[
                    feature_df["개체번호"] == cid
                ][feature]

                if len(group_values) > 1:
                    groups.append(group_values)

                    summary_data.append({
                        "개체번호": cid,
                        "표본수(n)": len(group_values),
                        "평균": group_values.mean(),
                        "표준편차": group_values.std()
                    })

            if len(groups) >= 2 and len(summary_data) >= 2:

                f_stat, p_val = stats.f_oneway(*groups)

                summary_df = pd.DataFrame(summary_data)
                summary_df = summary_df.sort_values("평균", ascending=False)
                summary_df = summary_df.reset_index(drop=True)

                # ============================
                # 1️⃣ ANOVA 결과
                # ============================
                st.markdown("### 📊 ANOVA 결과")
                st.write(f"F-statistic: {round(f_stat,4)}")
                st.write(f"p-value: {round(p_val,6)}")

                if p_val < 0.05:
                    st.success("✅ 개체 간 평균 차이 유의함")
                else:
                    st.info("ℹ 개체 간 평균 차이 유의하지 않음")

                # ============================
                # 2️⃣ 그래프
                # ============================
                st.markdown("### 📈 평균 ± 표준편차 그래프")

                fig, ax = plt.subplots(figsize=(6,4))

                labels = summary_df["개체번호"].astype(str)
                means = summary_df["평균"]
                stds = summary_df["표준편차"]

                ax.bar(labels, means, yerr=stds, capsize=5)

                ax.set_title(f"{feature} 평균 ± 표준편차")
                ax.set_xlabel("개체번호")
                ax.set_ylabel(feature)
                ax.grid(alpha=0.3)

                ax.text(
                    0.5,
                    max(means)*1.05,
                    f"ANOVA p = {round(p_val,5)}",
                    ha="center"
                )

                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)

                # ============================
                # 3️⃣ 개체별 요약 통계
                # ============================
                st.markdown("### 📋 개체별 요약 통계")

                display_df = summary_df.copy()
                display_df["평균"] = display_df["평균"].round(3)
                display_df["표준편차"] = display_df["표준편차"].round(3)

                st.dataframe(display_df, use_container_width=True)

                # ============================
                # 4️⃣ 전체 통계
                # ============================
                st.markdown("### 📌 전체 통계")
                st.write(f"전체 평균: {round(feature_df[feature].mean(),3)}")
                st.write(f"전체 표준편차: {round(feature_df[feature].std(),3)}")
                st.write(f"전체 표본수: {len(feature_df)}")

                # ============================
                # 5️⃣ 자동 해석
                # ============================
                st.markdown("### 🧠 자동 해석")

                max_row = summary_df.loc[summary_df["평균"].idxmax()]
                min_row = summary_df.loc[summary_df["평균"].idxmin()]

                max_id = max_row["개체번호"]
                min_id = min_row["개체번호"]

                if p_val < 0.001:
                    sig_level = "매우 높은 통계적 유의성(p < 0.001)"
                elif p_val < 0.01:
                    sig_level = "높은 통계적 유의성(p < 0.01)"
                elif p_val < 0.05:
                    sig_level = "통계적으로 유의한 차이(p < 0.05)"
                else:
                    sig_level = "통계적으로 유의한 차이가 없음(p ≥ 0.05)"

                if p_val < 0.05:
                    interpretation = f"""
                    {feature}는 개체 간 {sig_level}이 확인되었다.
                    평균이 가장 높은 개체는 {max_id}번,
                    가장 낮은 개체는 {min_id}번으로 나타났다.
                    이는 생육 특성 차이가 존재함을 시사한다.
                    """
                else:
                    interpretation = f"""
                    {feature}는 개체 간 {sig_level}.
                    평균 차이는 존재하나 통계적으로 유의하지 않았다.
                    """

                st.info(interpretation)

            else:
                st.warning("ANOVA 수행에 충분한 데이터가 없습니다.")

else:
    st.info("분석할 생육 지표를 선택하세요.")

st.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# 🌱 생육 대표값 계산
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🌱 생육 대표값 계산")

# 숫자형 변환
for col in growth_df.columns:
    if col not in ["개체번호", "조사일자"]:
        growth_df[col] = pd.to_numeric(growth_df[col], errors="ignore")

agg_dict = {}

for col in growth_df.columns:
    if col in ["개체번호", "조사일자"]:
        continue

    # -------------------------------
    # ⬇ 엽수(leaf count) → 무조건 mean
    # -------------------------------
    if any(keyword in col for keyword in ["엽수", "잎수", "leaf", "Leaf", "leaf_count"]):
        agg_dict[col] = "mean"
        continue

    # -------------------------------
    # 기존 규칙
    # -------------------------------
    if pd.api.types.is_numeric_dtype(growth_df[col]):
        # "수" 또는 "개수" 포함 → 합계(sum)
        if "수" in col or "개수" in col:
            agg_dict[col] = "sum"
        else:
            agg_dict[col] = "mean"
    else:
        agg_dict[col] = "first"

growth_group = growth_df.groupby("조사일자").agg(agg_dict).reset_index()
st.dataframe(growth_group)
st.markdown('</div>', unsafe_allow_html=True)

# ------------------------- 총합 표시 -------------------------
total_set = int(growth_group.get("화방별착과수", 0).sum())
total_harvest = int(growth_group.get("화방별수확수", 0).sum())
total_yield_rate = (total_harvest / total_set * 100) if total_set > 0 else 0

st.markdown(f"### 🌼 화방별착과수(총합): **{total_set:,} 개**")
st.markdown(f"### 🍅 화방별수확수(총합): **{total_harvest:,} 개**")
st.markdown(f"### 📊 총생산량률: **{total_yield_rate:.2f}%**")

# ============================================================
# 📈 개체 통합 시계열 그래프 (총합 지표)
# ============================================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 개체 통합 시계열 그래프 (총합 지표, 최대 2개)")

sum_metrics = [col for col, m in agg_dict.items() if m == "sum"]
selected_sum_metrics = st.multiselect("총합 지표 선택 (최대 2개)", sum_metrics, max_selections=2)

if selected_sum_metrics:
    grouped = growth_df.groupby("조사일자")[selected_sum_metrics].sum().reset_index()

    col_t1, col_t2 = st.columns(2)
    graph_cols2 = [col_t1, col_t2]

    for idx, feature in enumerate(selected_sum_metrics):
        with graph_cols2[idx % 2]:
            fig, ax = plt.subplots(figsize=(7, 4))

            ax.plot(
                grouped["조사일자"],
                grouped[feature],
                marker='o',
                markersize=marker_size,
                linewidth=2
            )

            ax.set_title(f"{feature} - 전체 개체 합계")
            ax.set_xlabel("조사일자")
            ax.set_ylabel(feature)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)

            ax.grid(alpha=0.2)
            plt.tight_layout()
            st.pyplot(fig)

    st.markdown("### 📄 조사일자별 합계 데이터")
    st.dataframe(grouped)
else:
    st.info("총합 지표를 선택하면 통합 시계열 그래프를 표시합니다.")

# 다운로드 버튼
st.markdown('<div class="card">', unsafe_allow_html=True)
st.download_button(
    "📥 생육 대표값 다운로드",
    growth_group.to_csv(index=False).encode("utf-8-sig"),
    "생육대표값.csv",
    "text/csv"
)
st.success("✔ 생육 데이터 처리 완료")
st.markdown('</div>', unsafe_allow_html=True)


