
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.naive_bayes import GaussianNB
import shap
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.inspection import PartialDependenceDisplay
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib
import platform
import os

# ===============================================
# 한글 폰트 설정
# ===============================================
FONT_PATH = "./fonts/NanumGothic.ttf"   # Streamlit Cloud에서는 반드시 이 경로에 업로드

# Matplotlib 폰트 적용
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
    plt.rc('font', family='NanumGothic')
else:
    st.warning("⚠ NanumGothic.ttf 파일을 찾을 수 없어 기본 폰트를 사용합니다.")

plt.rcParams['axes.unicode_minus'] = False

st.set_page_config(page_title="토마토 생육·수확 통합 분석", layout="wide")
st.title("생육 + 수확 데이터 통합 분석 대시보드")
st.markdown("---")

tab1, tab2, tab3, tab4 = st.tabs(
    ["🌱 생육 데이터 처리", "🍅 수확 데이터 처리", "🔗 생육 + 수확 통합", "📊 상관관계 분석"]
)

# =============================
# TAB 1 — 생육 데이터 처리
# =============================
with tab1:
    st.header("🌱 생육 데이터 처리")
    growth_file = st.file_uploader("📂 생육 데이터 업로드 (CSV)", type=["csv"], key="growth")
    fill_option = st.selectbox("결측치 처리 방법 선택", ["없음", "0", "평균값", "최빈값"], index=0)

    if growth_file:
        growth_df = pd.read_csv(growth_file)
        st.subheader("📌 생육 데이터 미리보기")
        st.dataframe(growth_df.head())

        # =============================
        # 개체번호 필터링
        # =============================
        if "개체번호" in growth_df.columns:
            unique_ids = growth_df["개체번호"].unique().tolist()
            selected_ids = st.multiselect("분석할 개체번호 선택", unique_ids, default=unique_ids)
            growth_df = growth_df[growth_df["개체번호"].isin(selected_ids)]
        else:
            st.error("❌ 생육 데이터에 '개체번호' 컬럼이 필요합니다.")

        # =============================
        # 결측치 처리
        # =============================
        if fill_option != "없음":
            for col in growth_df.columns:
                if growth_df[col].isnull().sum() > 0:
                    if fill_option == "0":
                        growth_df[col] = growth_df[col].fillna(0)
                    elif fill_option == "평균값" and pd.api.types.is_numeric_dtype(growth_df[col]):
                        growth_df[col] = growth_df[col].fillna(growth_df[col].mean())
                    elif fill_option == "최빈값":
                        growth_df[col] = growth_df[col].fillna(growth_df[col].mode()[0])

        # =============================
        # 시계열 그래프 + 이상치 탐색
        # =============================
        st.markdown("## 📈 개체별 시계열 그래프 & 이상치 탐색")

        numeric_cols = [
            col for col in growth_df.columns
            if pd.api.types.is_numeric_dtype(growth_df[col]) and col not in ["개체번호"]
        ]

        # ⚠ 첫 번째 selectbox → key 부여
        selected_feature = st.selectbox(
            "시계열로 볼 생육 지표 선택",
            numeric_cols,
            key="growth_feature_select_1"
        )

        replace_option = st.radio(
            "이상치 처리 방법 선택",
            ["적용 안함", "보간(interpolate)", "이전값(Fill Forward)", "평균값(전체 mean)"],
            horizontal=True
        )

        date_mode = st.radio(
            "X축 날짜 표시 방식",
            ["일 단위 그대로", "1주 단위 표시"],
            horizontal=True
        )

        growth_df["조사일자"] = pd.to_datetime(growth_df["조사일자"])

        fig, ax = plt.subplots(figsize=(10, 4))
        all_outliers_list = []

        for cid in selected_ids:
            sub_df = growth_df[growth_df["개체번호"] == cid].sort_values("조사일자").copy()

            # ================= 안전한 숫자 변환 =================
            # 문자열 → 숫자 / 오류는 NaN 처리
            sub_df[selected_feature] = pd.to_numeric(sub_df[selected_feature], errors="coerce")

            # ================= Z-score 기반 이상치 =================
            series_clean = sub_df[selected_feature].dropna()

            # 데이터가 모두 NaN인 경우 → 이상치 처리 불가
            if series_clean.empty:
                sub_df["Zscore"] = np.nan
                z_outliers = pd.DataFrame()
            else:
                # 안전한 Z-score 계산
                z = stats.zscore(series_clean)
                sub_df.loc[series_clean.index, "Zscore"] = z

                # Zscore 가 ±2 이상인 값
                z_outliers = sub_df[abs(sub_df["Zscore"]) > 2]

            # ================= 이동평균 기반 이상치 =================
            # 이동평균 계산 (window=3)
            sub_df["MA"] = sub_df[selected_feature].rolling(window=3, min_periods=1).mean()
            sub_df["MA_diff"] = abs(sub_df[selected_feature] - sub_df["MA"])

            # 임계값 100 초과
            ma_outliers = sub_df[sub_df["MA_diff"] > 100]

            # ================= 이상치 통합 =================
            if not z_outliers.empty or not ma_outliers.empty:
                outliers = pd.concat([z_outliers, ma_outliers]).drop_duplicates()
            else:
                outliers = pd.DataFrame()

            # 개체번호 추가
            if not outliers.empty:
                outliers["개체번호"] = cid

            # 리스트에 저장
            all_outliers_list.append(outliers)

            # ================= 이상치 처리 =================
            cleaned_df = sub_df.copy()

            if not outliers.empty:

                if replace_option == "보간(interpolate)":
                    cleaned_df.loc[outliers.index, selected_feature] = np.nan
                    cleaned_df[selected_feature] = cleaned_df[selected_feature].interpolate()

                elif replace_option == "이전값(Fill Forward)":
                    cleaned_df.loc[outliers.index, selected_feature] = np.nan
                    cleaned_df[selected_feature] = cleaned_df[selected_feature].fillna(method="ffill")

                elif replace_option == "평균값(전체 mean)":
                    mean_val = cleaned_df[selected_feature].mean()
                    cleaned_df.loc[outliers.index, selected_feature] = mean_val

            # ============= 그래프 =============
            ax.plot(
                cleaned_df["조사일자"],
                cleaned_df[selected_feature],
                marker="o",
                label=f"{cid}"
            )

            ax.scatter(
                outliers["조사일자"],
                outliers[selected_feature],
                color="red",
                s=70,
                label=f"{cid} 이상치"
            )

            for t in outliers["조사일자"]:
                ax.axvspan(
                    t - pd.Timedelta(days=0.5),
                    t + pd.Timedelta(days=0.5),
                    color="red",
                    alpha=0.15
                )

        ax.set_title(f"{selected_feature} 시계열 변화")
        ax.set_xlabel("조사일자")
        ax.set_ylabel(selected_feature)

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        formatter.formats[0] = "%m/%d"
        formatter.formats[1] = "%m/%d"
        formatter.formats[2] = "%m/%d"

        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)

        if date_mode == "1주 단위 표시":
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m/%d"))

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)

        # =============================
        # 이상치 목록 출력
        # =============================
        st.markdown("### 🔍 이상치 목록 (Z-score > 2 또는 이동평균 diff > 100)")
        if len(all_outliers_list) > 0:
            full_outlier_df = pd.concat(all_outliers_list).sort_values(["개체번호", "조사일자"])
            st.dataframe(full_outlier_df)
        else:
            st.info("📭 이상치가 없습니다.")

        # =============================
        # 조사일자 확인
        # =============================
        if "조사일자" not in growth_df.columns:
            st.error("❌ 생육 데이터에 '조사일자' 컬럼이 필요합니다.")
        else:
            growth_df["조사일자"] = pd.to_datetime(growth_df["조사일자"], errors="coerce")

            for col in growth_df.columns:
                if col not in ["개체번호", "조사일자"]:
                    growth_df[col] = pd.to_numeric(growth_df[col], errors="ignore")

            avg_cols_raw = ["초장", "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"]
            avg_cols = [c for c in avg_cols_raw if c in growth_df.columns]

            sum_cols_raw = [
                "화방별총개수", "화방별꽃수", "화방별꽃봉오리수",
                "화방별개화수", "화방별착과수", "화방별적과수", "화방별수확수"
            ]
            sum_cols = [c for c in sum_cols_raw if c in growth_df.columns]

            agg_dict = {}

            for col in growth_df.columns:
                if col in ["개체번호", "조사일자"]:
                    continue

                if col in avg_cols and pd.api.types.is_numeric_dtype(growth_df[col]):
                    agg_dict[col] = "mean"

                elif col in sum_cols and pd.api.types.is_numeric_dtype(growth_df[col]):
                    agg_dict[col] = "sum"

                else:
                    agg_dict[col] = "first"

            growth_group = growth_df.groupby("조사일자").agg(agg_dict).reset_index()

            # ------------------------------------------------------------
            # 🌱 생육 대표값 데이터 (평균 + 합계)
            # ------------------------------------------------------------

            st.subheader("🌱 생육 대표값 데이터 (평균 + 합계)")
            st.dataframe(growth_group)

            # ------------------------------------------------------------
            # 📌 평균값 지표 / 총합 지표 목록
            # ------------------------------------------------------------

            avg_metrics = ["초장", "생장길이", "엽수", "엽장", "엽폭", "줄기굵기", "화방높이"]
            sum_metrics = ["화방별총개수", "화방별꽃수", "화방별꽃봉오리수",
                           "화방별개화수", "화방별착과수", "화방별적과수", "화방별수확수"]

            st.markdown("### 📌 평균값 지표")
            st.write(", ".join(avg_metrics))

            st.markdown("### 📌 총합 지표")
            st.write(", ".join(sum_metrics))

            # ------------------------------------------------------------
            # 📌 총합 계산 (0,000개 형식)
            # ------------------------------------------------------------

            try:
                total_set = int(growth_group["화방별착과수"].sum())
                total_harvest = int(growth_group["화방별수확수"].sum())
            except Exception:
                total_set = 0
                total_harvest = 0

            st.markdown(f"### 🌼 화방별착과수(총합): **{total_set:,} 개**")
            st.markdown(f"### 🍅 화방별수확수(총합): **{total_harvest:,} 개**")

            # ------------------------------------------------------------
            # 📌 총생산량 % 계산
            # ------------------------------------------------------------

            if total_set > 0:
                total_yield_rate = total_harvest / total_set * 100
            else:
                total_yield_rate = 0

            st.markdown(f"### 📊 총생산량률: **{total_yield_rate:.2f}%**")

            # ============================================================
            # 🌱 대표값 이후 — 개체통합 시계열 그래프
            # ============================================================

            st.subheader("📈 개체통합 시계열 그래프 (총합 지표 전용)")

            # ------------------------------------------------------------
            # 총합 지표 리스트
            # ------------------------------------------------------------
            sum_metrics = [
                "화방별총개수", "화방별꽃수", "화방별꽃봉오리수",
                "화방별개화수", "화방별착과수", "화방별적과수", "화방별수확수"
            ]

            # growth_df에 존재하는 컬럼만 사용
            sum_metrics_valid = [col for col in sum_metrics if col in growth_df.columns]

            # 지표 선택
            metric_sum = st.selectbox(
                "시계열로 볼 총합 지표 선택",
                sum_metrics_valid,
                key="integrated_sum_metric"
            )

            # ------------------------------------------------------------
            # 조사일자 기준 개체 합계 만들기
            # ------------------------------------------------------------

            df_sum_daily = (
                growth_df.groupby("조사일자")[sum_metrics_valid]
                .sum()
                .reset_index()
                .sort_values("조사일자")
            )

            # ------------------------------------------------------------
            # 그래프 생성
            # ------------------------------------------------------------

            fig, ax = plt.subplots(figsize=(12, 5))

            ax.plot(
                df_sum_daily["조사일자"],
                df_sum_daily[metric_sum],
                marker="o",
                linewidth=2
            )

            ax.set_title(f"📈 개체통합 시계열 그래프 - {metric_sum}")
            ax.set_xlabel("조사일자")
            ax.set_ylabel(f"{metric_sum} (합계)")
            ax.grid()

            st.pyplot(fig)

            # ------------------------------------------------------------
            # 데이터 테이블 출력
            # ------------------------------------------------------------

            st.markdown("### 📄 조사일자별 합계 데이터")
            st.dataframe(df_sum_daily[["조사일자", metric_sum]])

            # =============================
            # TAB 1 — 데이터 다운로드
            # =============================
            st.download_button(
                "📥 생육 대표값 다운로드",
                growth_group.to_csv(index=False).encode("utf-8-sig"),
                "생육대표값_평균합계.csv",
                "text/csv"
            )

            st.success("✔ 생육 데이터 처리 완료 (평균 + 총합 계산)")

# =============================
# TAB 2 — 수확 데이터 처리
# =============================
with tab2:
    st.header("🍅 수확 데이터 처리")
    harvest_file = st.file_uploader("📂 수확 데이터 업로드 (CSV or Excel)", type=["csv", "xlsx"], key="harvest")
    fill_option_h = st.selectbox("결측치 처리 방법 선택", ["없음", "0", "평균값", "최빈값"], index=0, key="harvest_fill")

    if harvest_file:
        if harvest_file.name.endswith(".csv"):
            harvest_df = pd.read_csv(harvest_file)
        else:
            harvest_df = pd.read_excel(harvest_file)

        st.subheader("📌 수확 데이터 미리보기")
        st.dataframe(harvest_df.head())

        date_col = st.selectbox("📅 조사일자 컬럼 선택", harvest_df.columns)
        weight_col = st.selectbox("⚖ 수확과중 컬럼 선택", harvest_df.columns)

        # 결측치 처리
        if fill_option_h != "없음":
            for col in harvest_df.columns:
                if harvest_df[col].isnull().sum() > 0:
                    if fill_option_h == "0":
                        harvest_df[col] = harvest_df[col].fillna(0)
                    elif fill_option_h == "평균값" and pd.api.types.is_numeric_dtype(harvest_df[col]):
                        harvest_df[col] = harvest_df[col].fillna(harvest_df[col].mean())
                    elif fill_option_h == "최빈값":
                        harvest_df[col] = harvest_df[col].fillna(harvest_df[col].mode()[0])

        harvest_group = harvest_df.groupby(date_col).agg(
            수확수=(weight_col, "count"),
            수확과중평균=(weight_col, "mean")
        ).reset_index()

        st.subheader("🍅 조사일자별 수확 데이터 요약")
        st.dataframe(harvest_group)
        st.download_button("📥 수확 데이터 다운로드",
                           harvest_group.to_csv(index=False).encode("utf-8-sig"),
                           "수확데이터.csv", "text/csv")
        st.success("✔ 수확 데이터 처리 완료")

# =============================
# TAB 3 — 생육 + 수확 통합
# =============================
with tab3:
    st.header("🔗 생육 + 수확 데이터 통합")
    try:
        growth_group
        harvest_group
    except:
        st.warning("⚠ 생육 · 수확 데이터를 먼저 업로드하세요.")
    else:
        # outer merge로 생육/수확 모든 날짜 포함
        if date_col != "조사일자":
            harvest_group = harvest_group.rename(columns={date_col: "조사일자"})
        merged_df = pd.merge(harvest_group, growth_group, on="조사일자", how="outer")

        # 수확 컬럼 결측치 0 처리
        for col in ["수확수", "수확과중평균"]:
            if col in merged_df.columns:
                merged_df[col] = merged_df[col].fillna(0)

        # 컬럼 순서 조정
        harvest_cols = ["조사일자", "수확수", "수확과중평균"]
        other_cols = [c for c in merged_df.columns if c not in harvest_cols]
        merged_df = merged_df[harvest_cols + other_cols]

        st.subheader("🔗 통합 데이터")
        st.dataframe(merged_df)
        st.download_button("📥 통합 데이터 다운로드",
                           merged_df.to_csv(index=False).encode("utf-8-sig"),
                           "생육_수확_통합데이터.csv", "text/csv")
        st.success("✔ 생육 + 수확 통합 완료")

# =============================
# TAB 4 — 상관관계 분석
# =============================
with tab4:
    st.header("📊 상관관계 분석 (생육 + 수확)")
    try:
        merged_df
    except:
        st.warning("⚠ 먼저 통합 데이터를 생성해주세요.")
    else:
        numeric_cols = merged_df.select_dtypes(include="number").columns.tolist()
        selected_cols = st.multiselect("분석할 컬럼 선택", numeric_cols, default=numeric_cols)
        corr_df = merged_df[selected_cols]

        # 결측치/Inf 제거
        corr_df_clean = corr_df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how='any')

        # p-value 임계값
        p_thresh = st.slider("p-value 임계 (%) 선택", 1, 100, 5)
        p_thresh_val = p_thresh / 100

        # 상관계수와 p-value 계산
        corr_matrix = corr_df_clean.corr()
        p_matrix = pd.DataFrame(np.ones(corr_matrix.shape), columns=corr_matrix.columns, index=corr_matrix.index)
        for i in corr_matrix.columns:
            for j in corr_matrix.columns:
                if i != j:
                    try:
                        _, p = stats.pearsonr(corr_df_clean[i], corr_df_clean[j])
                        p_matrix.loc[i, j] = p
                    except:
                        p_matrix.loc[i, j] = np.nan
                else:
                    p_matrix.loc[i, j] = 0

        # p-value 필터링
        mask_p = p_matrix > p_thresh_val

        # 히트맵 색상 강조
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", mask=mask_p,
                    cbar_kws={'label': '상관계수'}, ax=ax)
        st.subheader("🔥 상관관계 히트맵 (p-value < {}%)".format(p_thresh))
        st.pyplot(fig)

        # VIF 계산 및 시각화
        vif_df = pd.DataFrame()
        vif_df["변수"] = corr_df_clean.columns
        vif_df["VIF"] = [variance_inflation_factor(corr_df_clean.values, i) for i in range(corr_df_clean.shape[1])]
        st.subheader("📈 VIF (Variance Inflation Factor)")
        st.dataframe(vif_df.round(3))

        fig_vif, ax_vif = plt.subplots(figsize=(12, 6))
        sns.barplot(x="변수", y="VIF", data=vif_df, palette="magma", ax=ax_vif)
        ax_vif.axhline(5, color='red', linestyle='--', label='VIF=5 기준')
        ax_vif.set_title("VIF 시각화")
        ax_vif.set_ylabel("VIF 값")
        ax_vif.set_xlabel("변수")
        plt.xticks(rotation=45)
        ax_vif.legend()
        st.pyplot(fig_vif)

        # 다운로드
        st.download_button("📥 상관계수표 다운로드",
                           corr_matrix.round(3).to_csv(index=True).encode("utf-8-sig"),
                           "상관계수표.csv", "text/csv")
        fig.savefig("상관관계_히트맵.png")
        with open("상관관계_히트맵.png", "rb") as f:
            st.download_button("📥 상관관계_히트맵 다운로드", f, "상관관계_히트맵.png", "image/png")
