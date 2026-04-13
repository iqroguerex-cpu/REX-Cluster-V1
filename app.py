import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="REX-Cluster-V1 | IQROGUEREX",
    page_icon="📊",
    layout="wide"
)

# --- DARK UI ---
st.markdown("""
<style>
    .main { background-color: #0B0E14; }
    div[data-testid="stMetricValue"] { 
        font-size: 32px; 
        color: #00FFD1; 
    }
    section[data-testid="stSidebar"] { 
        background-color: #11151C; 
    }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Customer Segmentation Monolith")
st.markdown("Developed by **Chinmay V Chatradamath**")
st.divider()

# --- CONSTANTS ---
FILE_NAME = "Mall_Customers.csv"
INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

# --- LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_NAME)
    df.columns = df.columns.str.strip()
    return df

# --- SAFE LOAD ---
try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded: {FILE_NAME}")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.write("Available files:", __import__("os").listdir())
    st.stop()

# --- VALIDATION ---
if INCOME_COL not in df.columns or SPENDING_COL not in df.columns:
    st.error("❌ Required columns missing")
    st.stop()

# --- FEATURES ---
X = df[[INCOME_COL, SPENDING_COL]]

# --- SIDEBAR ---
st.sidebar.header("REX Control Panel")
k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
show_elbow = st.sidebar.checkbox("Show Elbow Analysis")

# --- MODEL ---
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster'] = [f"Segment {i+1}" for i in kmeans.fit_predict(X)]

# --- LAYOUT ---
col1, col2 = st.columns([1, 3])

# --- METRICS ---
with col1:
    st.subheader("Market Intelligence")
    st.metric("Total Records", len(df))
    st.metric("Active Clusters", k_value)

    cluster_counts = df['Cluster'].value_counts().reset_index()
    cluster_counts.columns = ['Segment', 'Count']
    st.dataframe(cluster_counts, use_container_width=True)

# --- SAFE HOVER DATA ---
hover_cols = [col for col in ['Age', 'Gender'] if col in df.columns]

# --- PLOT ---
with col2:
    fig = px.scatter(
        df,
        x=INCOME_COL,
        y=SPENDING_COL,
        color='Cluster',
        hover_data=hover_cols,
        template="plotly_dark",
        title=f"Customer Distribution (K={k_value})"
    )

    # --- CENTROIDS (FIXED) ---
    centers = kmeans.cluster_centers_
    fig.add_trace(go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode='markers',
        marker=dict(
            color='white',
            size=16,
            symbol='star',
            line=dict(width=2, color='#00FFD1')
        ),
        name='Centroids'
    ))

    st.plotly_chart(fig, use_container_width=True)

# --- ELBOW ---
if show_elbow:
    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=42, n_init=10)
        km.fit(X)
        wcss.append(km.inertia_)

    elbow_fig = px.line(
        x=list(range(1, 11)),
        y=wcss,
        markers=True,
        template="plotly_dark",
        title="Elbow Analysis"
    )

    st.plotly_chart(elbow_fig, use_container_width=True)
