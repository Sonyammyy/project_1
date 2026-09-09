import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns

# ---------------------------------------------------------
# 0. 페이지 설정 및 한글 폰트 설정 (Pretendard-Regular.otf)
# ---------------------------------------------------------
st.set_page_config(
    page_title="무역 분석 대시보드",
    page_icon="📊",
    layout="wide"
)

font_path = "Pretendard-Regular.otf"
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    font_prop = fm.FontProperties(fname=font_path)
    plt.rc("font", family=font_prop.get_name())
    st.markdown(
        f"""
        <style>
        @font-face {{
            font-family: 'Pretendard';
            src: url('{font_path}') format('opentype');
        }}
        html, body, [class*="css"], div, span, label {{
            font-family: 'Pretendard', sans-serif !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
else:
    # 폰트 파일이 없을 경우 시스템 기본 한글 폰트 폴백
    plt.rc("font", family="Malgun Gothic" if os.name == "nt" else "AppleGothic")

plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 캐싱
# ---------------------------------------------------------
@st.cache_data
def load_data():
    baci_file = "baci_85_sample.csv"
    country_file = "country_codes_sample.csv"

    # BACI 샘플 데이터 불러오기
    df_baci = pd.read_csv(baci_file)
    # 국가 코드 데이터 불러오기
    df_country = pd.read_csv(country_file)

    # 1) 원본 결측치 확인용 데이터프레임 보존
    missing_df = pd.DataFrame({
        "컬럼명": df_baci.columns,
        "결측치 개수": df_baci.isnull().sum().values,
        "결측치 비율(%)": (df_baci.isnull().sum().values / len(df_baci) * 100).round(2)
    })

    # 2) j열을 기준으로 국가명 매핑
    country_map = dict(zip(df_country["j"], df_country["country_name"]))
    df_baci["country_name"] = df_baci["j"].map(country_map).fillna(df_baci["j"].astype(str))

    # 3) 무역액 등급 구분 (대·중·소) - 3분위수(qcut) 기준 분류
    df_baci["trade_grade"] = pd.qcut(
        df_baci["v"],
        q=3,
        labels=["소", "중", "대"]
    )

    return df_baci, missing_df

df_raw, missing_df = load_data()

# ---------------------------------------------------------
# 2. 사이드바 필터 구성
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 옵션")

# 국가 선택 필터 (j열 기준 국가명)
all_countries = sorted(list(df_raw["country_name"].unique()))
selected_countries = st.sidebar.multiselect(
    "국가 선택",
    options=all_countries,
    default=all_countries
)

# 무역액 등급 선택 (대·중·소)
grade_options = ["대", "중", "소"]
selected_grades = st.sidebar.multiselect(
    "무역액 등급 선택",
    options=grade_options,
    default=grade_options
)

# 필터링 적용
filtered_df = df_raw[
    (df_raw["country_name"].isin(selected_countries)) &
    (df_raw["trade_grade"].isin(selected_grades))
]

# ---------------------------------------------------------
# 3. 오른쪽 화면 메인 콘텐츠
# ---------------------------------------------------------

# 1. 타이틀
st.title("무역 분석 대시보드")
st.markdown("---")

# 2. baci_85_sample.csv 파일의 결측치
st.subheader("📌 baci_85_sample.csv 파일의 결측치")
st.dataframe(missing_df, use_container_width=True)

st.markdown("---")

# 3. 총 거래 건 수 & 총 수출액($) 두 열로 나누어서
st.subheader("📊 거래 현황 요약")
col1, col2 = st.columns(2)

total_count = len(filtered_df)
total_export_val = filtered_df["v"].sum()

with col1:
    st.metric(
        label="총 거래 건 수",
        value=f"{total_count:,} 건"
    )

with col2:
    st.metric(
        label="총 수출액 ($)",
        value=f"${total_export_val:,.2f}"
    )

st.markdown("---")

# 4. 국가*연도 수출액 히트맵(상위 8개국) & 무역액 등급분포 두 열로 나누어서
st.subheader("📈 무역 심층 시각화")
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.markdown("##### 국가 × 연도 수출액 히트맵 (상위 8개국)")
    if not filtered_df.empty:
        # 상위 8개국 추출 (수출액 합계 기준)
        top_8_countries = (
            filtered_df.groupby("country_name")["v"]
            .sum()
            .nlargest(8)
            .index
        )
        heatmap_data = (
            filtered_df[filtered_df["country_name"].isin(top_8_countries)]
            .pivot_table(index="country_name", columns="t", values="v", aggfunc="sum", fill_value=0)
        )
        
        fig_heat, ax_heat = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            annot=True,
            fmt=",.1f",
            linewidths=0.5,
            ax=ax_heat
        )
        ax_heat.set_xlabel("연도")
        ax_heat.set_ylabel("국가명")
        plt.tight_layout()
        st.pyplot(fig_heat)
    else:
        st.info("선택된 데이터가 없습니다.")

with chart_col2:
    st.markdown("##### 무역액 등급 분포")
    if not filtered_df.empty:
        grade_counts = filtered_df["trade_grade"].value_counts().reindex(["대", "중", "소"], fill_value=0)
        
        fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
        colors = ["#2b5c8f", "#4f81bd", "#95b3d7"]
        bars = ax_bar.bar(grade_counts.index, grade_counts.values, color=colors, edgecolor="none", width=0.5)
        
        # 막대 위 수치 표시
        for bar in bars:
            height = bar.get_height()
            ax_bar.annotate(
                f'{int(height):,}건',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=10
            )
            
        ax_bar.set_xlabel("무역액 등급")
        ax_bar.set_ylabel("건수")
        ax_bar.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig_bar)
    else:
        st.info("선택된 데이터가 없습니다.")

st.markdown("---")

# 5. 상위 5개국 * 무역액 등급 교차표 (원본건수 & 정규화비율)
st.subheader("📋 상위 5개국 × 무역액 등급 교차표")

if not filtered_df.empty:
    top_5_countries = (
        filtered_df.groupby("country_name")["v"]
        .sum()
        .nlargest(5)
        .index
    )
    df_top5 = filtered_df[filtered_df["country_name"].isin(top_5_countries)]

    # 원본 건수 교차표
    crosstab_count = pd.crosstab(
        df_top5["country_name"],
        df_top5["trade_grade"],
        margins=True,
        margins_name="합계"
    )

    # 정규화 비율 교차표 (행 기준 비율, 백분율 표기)
    crosstab_norm = pd.crosstab(
        df_top5["country_name"],
        df_top5["trade_grade"],
        normalize="index"
    ) * 100

    tab_col1, tab_col2 = st.columns(2)

    with tab_col1:
        st.markdown("##### [원본 건수]")
        st.dataframe(crosstab_count, use_container_width=True)

    with tab_col2:
        st.markdown("##### [정규화 비율 (%)]")
        st.dataframe(crosstab_norm.style.format("{:.2f}%"), use_container_width=True)
else:
    st.info("선택된 데이터가 없습니다.")