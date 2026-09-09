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
    # Streamlit 전체 CSS에 Pretendard 적용
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
    # 폰트 파일이 없을 경우 시스템 한글 폰트 폴백 설정
    plt.rc("font", family="Malgun Gothic" if os.name == "nt" else "NanumGothic")

plt.rcParams["axes.unicode_minus"] = False

# ---------------------------------------------------------
# 1. 데이터 로드 및 전처리 캐싱
# ---------------------------------------------------------
@st.cache_data
def load_data():
    baci_file = "baci_85_sample.csv"
    country_file = "country_codes_sample.csv"

    # BACI 샘플 데이터 불러오기
    if os.path.exists(baci_file):
        df_baci = pd.read_csv(baci_file)
    else:
        # 파일이 없을 경우 예시 구조 가상 생성 (테스트용)
        np.random.seed(42)
        n = 1000
        countries = [124, 156, 251, 276, 392, 410, 842, 826, 682, 702]
        years = [2018, 2019, 2020, 2021, 2022]
        df_baci = pd.DataFrame({
            "t": np.random.choice(years, n),
            "i": np.random.choice(countries, n),
            "j": np.random.choice(countries, n),
            "k": np.random.randint(100000, 999999, n),
            "v": np.random.exponential(scale=5000, size=n) + 10,
            "q": np.random.exponential(scale=1000, size=n)
        })
        # 결측치 일부 생성
        df_baci.loc[np.random.choice(n, 30, replace=False), "q"] = np.nan

    # 국가 코드 데이터 불러오기
    if os.path.exists(country_file):
        df_country = pd.read_csv(country_file)
    else:
        df_country = pd.DataFrame({
            "country_code": [124, 156, 251, 276, 392, 410, 842, 826, 682, 702],
            "country_name": ["Canada", "China", "France", "Germany", "Japan", "Korea", "USA", "UK", "Saudi Arabia", "Singapore"]
        })

    # 컬럼 표준화
    # BACI: t(연도), i(수출국), j(수입국), v(수출액), q(수량)
    # 국가코드 컬럼 확인 (country_code / i_iso3 / iso3 / id 등)
    c_code_col = [col for col in df_country.columns if "code" in col.lower() or "id" in col.lower() or col in ["i", "country_code_iso3", "iso3"]]
    c_name_col = [col for col in df_country.columns if "name" in col.lower() or "country" in col.lower()]
    
    code_col = c_code_col[0] if c_code_col else df_country.columns[0]
    name_col = c_name_col[0] if c_name_col else (df_country.columns[1] if len(df_country.columns) > 1 else df_country.columns[0])

    # 국가명 매핑 딕셔너리
    country_map = dict(zip(df_country[code_col], df_country[name_col]))

    # 수출국(i) 기준 국가명 생성
    export_col = "i" if "i" in df_baci.columns else ("exporter" if "exporter" in df_baci.columns else df_baci.columns[1])
    year_col = "t" if "t" in df_baci.columns else ("year" if "year" in df_baci.columns else df_baci.columns[0])
    val_col = "v" if "v" in df_baci.columns else ("value" if "value" in df_baci.columns else df_baci.columns[4])

    df_baci["country_name"] = df_baci[export_col].map(country_map).fillna(df_baci[export_col].astype(str))
    
    # 무역액 등급 구분 (대·중·소) - 3분위수(qcut) 기준 분류
    df_baci["trade_grade"] = pd.qcut(
        df_baci[val_col],
        q=3,
        labels=["소", "중", "대"]
    )

    return df_baci, df_country, year_col, export_col, val_col

df_raw, df_country, col_year, col_exporter, col_val = load_data()

# ---------------------------------------------------------
# 2. 사이드바 필터 구성
# ---------------------------------------------------------
st.sidebar.header("🔍 필터 옵션")

# 국가 선택 필터
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
missing_df = pd.DataFrame({
    "컬럼명": df_raw.columns,
    "결측치 개수": df_raw.isnull().sum().values,
    "결측치 비율(%)": (df_raw.isnull().sum().values / len(df_raw) * 100).round(2)
})
st.dataframe(missing_df.T if False else missing_df, use_container_width=True)

st.markdown("---")

# 3. 총 거래 건 수 & 총 수출액($) 두 열로 나누어서
st.subheader("📊 거래 현황 요약")
col1, col2 = st.columns(2)

total_count = len(filtered_df)
total_export_val = filtered_df[col_val].sum()

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
        # 상위 8개 수출국 추출
        top_8_countries = (
            filtered_df.groupby("country_name")[col_val]
            .sum()
            .nlargest(8)
            .index
        )
        heatmap_data = (
            filtered_df[filtered_df["country_name"].isin(top_8_countries)]
            .pivot_table(index="country_name", columns=col_year, values=col_val, aggfunc="sum", fill_value=0)
        )
        
        fig_heat, ax_heat = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            heatmap_data,
            cmap="YlGnBu",
            annot=True,
            fmt=",.0f",
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
        
        # 바 차트 위 수치 표시
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
        filtered_df.groupby("country_name")[col_val]
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
