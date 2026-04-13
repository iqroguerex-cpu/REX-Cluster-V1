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
.main { background-color: #0E1117; }
[data-testid="stMetricValue"] { font-size: 28px; color: #00FFD1; }
section[data-testid="stSidebar"] { background-color: #161B22; }
</style>
""", unsafe_allow_html=True)

# --- HEADER ---
st.title("📊 Customer Segmentation Monolith")
st.markdown("Developed by **Chinmay V Chatradamath** | *Vertical AI Analytics Engine*")
st.divider()

# --- CONSTANTS ---
FILE_NAME = "Mall_Customers.csv"
INCOME_COL = "Annual Income (k$)"
SPENDING_COL = "Spending Score (1-100)"

# --- LOAD DATA (GitHub Repo File) ---
@st.cache_data
def load_data():
    df = pd.read_csv(FILE_NAME)
    df.columns = df.columns.str.strip()
    return df

# --- LOAD ---
try:
    df = load_data()
    st.sidebar.success(f"✅ Loaded: {FILE_NAME}")
except Exception as e:
    st.error(f"❌ Failed to load dataset: {e}")
    st.stop()

# --- VALIDATION ---
if INCOME_COL not in df.columns or SPENDING_COL not in df.columns:
    st.error("❌ Required columns not found in CSV.")
    st.stop()

# --- FEATURES ---
X = df[[INCOME_COL, SPENDING_COL]]

# --- SIDEBAR ---
st.sidebar.header("REX Control Panel")
k_value = st.sidebar.slider("Clustering Density (K)", 2, 10, 5)
show_elbow = st.sidebar.checkbox("Show Elbow Analysis")
st.sidebar.divider()
st.sidebar.info("Model: K-Means\nInitialization: k-means++")

# --- MODEL ---
kmeans = KMeans(n_clusters=k_value, random_state=42, n_init=10)
df['Cluster'] = [f"Segment {i+1}" for i in kmeans.fit_predict(X)]

# --- LAYOUT ---
col1, col2 = st.columns([1, 3])

# --- METRICS ---
with col1:
    st.subheader("Segment Metrics")
    st.metric("Total Records", len(df))
    st.metric("Active Clusters", k_value)

    with st.expander("View Raw Assignments"):
        st.dataframe(df[[INCOME_COL, SPENDING_COL, 'Cluster']].head(10))

# --- SCATTER PLOT ---
with col2:
    fig = px.scatter(
        df,
        x=INCOME_COL,
        y=SPENDING_COL,
        color='Cluster',
        hover_data=[col for col in ['Gender', 'Age'] if col in df.columns],
        template="plotly_dark",
        title=f"Customer Distribution (K={k_value})",
        color_discrete_sequence=["#00FFD1", "#FF00E4", "#00D1FF", "#8A2BE2", "#ADFF2F"]
    )

    # --- CENTROIDS ---
    centers = kmeans.cluster_centers_
    fig.add_trace(go.Scatter(
        x=centers[:, 0],
        y=centers[:, 1],
        mode='markers',
        marker=dict(
            color='white',
            size=15,
            symbol='star',
            line=dict(width=2, color='black')
        ),
        name='Centroids'
    ))

    fig.update_layout(legend_title_text='Market Segments')
    st.plotly_chart(fig, use_container_width=True)

# --- ELBOW METHOD ---
if show_elbow:
    st.divider()

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

    elbow_fig.update_traces(line_color='#00FFD1')
    st.plotly_chart(elbow_fig, use_container_width=True)
