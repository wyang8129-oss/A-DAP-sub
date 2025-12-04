import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import platform
import numpy as np
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ===============================================
# 한글 폰트 설정
# ===============================================
st.markdown("""
<style>
@font-face {
    font-family: 'NanumGothic';
    src: url('NanumGothic.ttf') format('truetype');
}
html, body, [class*="css"] {
    font-family: 'NanumGothic', sans-serif;
}
</style>
""", unsafe_allow_html=True)

st.set_page_config(page_title="토마토 생육·수확 데이터 통합 대시보드", layout="wide")
st.title("생육 + 수확 데이터 통합  대시보드")
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

        if "개체번호" in growth_df.columns:
            unique_ids = growth_df["개체번호"].unique().tolist()
            selected_ids = st.multiselect("분석할 개체번호 선택", unique_ids, default=unique_ids)
            growth_df = growth_df[growth_df["개체번호"].isin(selected_ids)]
        else:
            st.error("❌ 생육 데이터에 '개체번호' 컬럼이 필요합니다.")

        # 결측치 처리
        if fill_option != "없음":
            for col in growth_df.columns:
                if growth_df[col].isnull().sum() > 0:
                    if fill_option == "0":
                        growth_df[col] = growth_df[col].fillna(0)
                    elif fill_option == "평균값" and pd.api.types.is_numeric_dtype(growth_df[col]):
                        growth_df[col] = growth_df[col].fillna(growth_df[col].mean())
                    elif fill_option == "최빈값":
                        growth_df[col] = growth_df[col].fillna(growth_df[col].mode()[0])

        if "조사일자" not in growth_df.columns:
            st.error("❌ 생육 데이터에 '조사일자' 컬럼이 필요합니다.")
        else:
            numeric_cols = [c for c in growth_df.columns
                            if pd.api.types.is_numeric_dtype(growth_df[c]) and c != "개체번호"]
            non_numeric_cols = [c for c in growth_df.columns if c not in numeric_cols and c != "조사일자"]

            growth_group = growth_df.groupby("조사일자").agg(
                {**{col: "mean" for col in numeric_cols},
                 **{col: "first" for col in non_numeric_cols}}
            ).reset_index()

            st.subheader("🌱 생육 대표값 데이터")
            st.dataframe(growth_group)
            st.download_button("📥 생육 대표값 다운로드",
                               growth_group.to_csv(index=False).encode("utf-8-sig"),
                               "생육대표값.csv", "text/csv")
            st.success("✔ 생육 데이터 처리 완료")

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
