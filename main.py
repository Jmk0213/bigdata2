import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="배달 구역 클러스터링", layout="wide")

st.title("🚚 배달 위치 클러스터링 지도")
st.markdown("K-Means 알고리즘을 활용해 배달 구역을 구분하고 지도에 표시합니다.")

# 데이터 불러오기
df = pd.read_csv("Delivery.csv")
st.subheader("데이터 미리보기")
st.dataframe(df.head())

# 위도/경도 컬럼명 고정 (데이터 컬럼명에 맞게 변경!)
lat_col = 'latitude'
lon_col = 'longitude'

# 클러스터 개수 선택
n_clusters = st.slider("클러스터 개수 선택 (K)", min_value=2, max_value=10, value=3)

# K-Means 클러스터링
coords = df[[lat_col, lon_col]]
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(coords)

# 지도 그리기
center = [df[lat_col].mean(), df[lon_col].mean()]
m = folium.Map(location=center, zoom_start=12)

colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']

for i, row in df.iterrows():
    folium.CircleMarker(
        location=[row[lat_col], row[lon_col]],
        radius=5,
        color=colors[int(row['Cluster']) % len(colors)],
        fill=True,
        fill_opacity=0.7,
        popup=f"Cluster {row['Cluster']}"
    ).add_to(m)

st.subheader("클러스터링 결과 지도")
st_folium(m, width=800, height=600)

st.subheader("클러스터링 데이터")
st.dataframe(df)
