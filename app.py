import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import io

# ---------------------------------------------------------
# PAGE SETUP
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation Dashboard",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# STYLING & COLORS
# ---------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Poppins', sans-serif;
    }

    /* Background and containers */
    .stApp {
        background-color: #0F172A;
        color: #F8FAFC;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #1E293B;
        border-right: 1px solid #334155;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.25);
        border-left: 4px solid var(--accent-color, #3B82F6);
        margin-bottom: 10px;
    }
    .kpi-title {
        font-size: 0.85rem;
        color: #94A3B8;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-top: 4px;
    }

    /* Segment Detail Cards */
    .segment-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 14px;
        border-left: 6px solid var(--seg-color, #3B82F6);
    }
    .segment-title {
        font-size: 1.2rem;
        font-weight: 700;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Prediction Result Box */
    .pred-box {
        background: #1E293B;
        border: 2px solid #10B981;
        border-radius: 14px;
        padding: 20px;
        text-align: center;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# COLOR PALETTE FOR CLUSTERS
# ---------------------------------------------------------
CLUSTER_COLORS = {
    "🏆 VIP Customers": "#10B981",          # Green
    "💎 Careful Spenders": "#6366F1",       # Indigo / Purple
    "⚡ Careless / Impulsive": "#EC4899",   # Pink
    "💰 Budget Seekers": "#F59E0B",         # Amber / Orange
    "👥 Standard Shoppers": "#06B6D4",      # Cyan / Blue
    "Cluster 1": "#10B981",
    "Cluster 2": "#6366F1",
    "Cluster 3": "#EC4899",
    "Cluster 4": "#F59E0B",
    "Cluster 5": "#06B6D4",
    "Cluster 6": "#8B5CF6",
    "Cluster 7": "#14B8A6",
    "Cluster 8": "#E11D48"
}

# ---------------------------------------------------------
# LOAD DEFAULT DATASET
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    try:
        return pd.read_csv("dataset/Mall_Customers.csv")
    except:
        return pd.read_csv("Mall_Customers.csv")

default_df = load_default_data()

# ---------------------------------------------------------
# SIDEBAR: DATA SOURCE & NAVIGATION
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛍️ Customer Segmentation")
    st.caption("College Machine Learning Project")
    st.markdown("---")

    st.markdown("### 📁 Dataset Source")
    data_source_mode = st.radio(
        "Choose Data Source:",
        ["🛍️ Default (Mall Customers)", "📤 Upload Custom Dataset"]
    )

    uploaded_file = None
    if data_source_mode == "📤 Upload Custom Dataset":
        uploaded_file = st.file_uploader(
            "Upload CSV or Excel file:",
            type=["csv", "xlsx", "xls"],
            help="Upload any customer dataset containing at least 2 numeric columns."
        )

    st.markdown("---")
    st.markdown("### 📌 Navigation")
    selected_page = st.radio(
        "Go to page:",
        [
            "📊 Dashboard Overview",
            "📂 Upload & Preview Dataset",
            "📉 Finding Optimal K (Elbow Method)",
            "🎯 Customer Clusters (Visuals)",
            "👥 Customer Groups & Strategies",
            "🤖 Predict New Customer",
            "📋 View & Download Data"
        ]
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Controls")
    k_clusters = st.slider("Select Number of Clusters (k)", min_value=2, max_value=8, value=5)

# ---------------------------------------------------------
# PROCESS DATASET (DEFAULT OR UPLOADED)
# ---------------------------------------------------------
is_custom_dataset = False
if data_source_mode == "📤 Upload Custom Dataset" and uploaded_file is not None:
    try:
        if uploaded_file.name.endswith(".csv"):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        is_custom_dataset = True
        st.sidebar.success(f"✅ Loaded: {uploaded_file.name}")
    except Exception as e:
        st.sidebar.error(f"Error reading file: {e}")
        data = default_df.copy()
else:
    data = default_df.copy()

# Auto-detect column roles
numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

if len(numeric_cols) < 2:
    st.error("⚠️ The dataset must contain at least 2 numeric columns for clustering. Reverting to default dataset.")
    data = default_df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    is_custom_dataset = False

# Auto-match or let user select features in sidebar
with st.sidebar:
    if is_custom_dataset:
        st.markdown("### 🔍 Select Features for Clustering")
        
        # Smart default for feature X (income or 1st numeric)
        default_x_idx = 0
        for i, col in enumerate(numeric_cols):
            if "income" in col.lower():
                default_x_idx = i
                break
        feature_x = st.selectbox("Feature 1 (X-Axis / Income):", numeric_cols, index=default_x_idx)

        # Smart default for feature Y (spending/score or 2nd numeric)
        remaining_cols = [c for c in numeric_cols if c != feature_x]
        default_y_idx = 0
        for i, col in enumerate(remaining_cols):
            if any(k in col.lower() for k in ["spend", "score"]):
                default_y_idx = i
                break
        feature_y = st.selectbox("Feature 2 (Y-Axis / Spending):", remaining_cols, index=default_y_idx)
    else:
        feature_x = "Annual Income (k$)"
        feature_y = "Spending Score (1-100)"

    # Detect Gender column if present
    gender_col = None
    for col in data.columns:
        if col.lower() in ["gender", "sex"]:
            gender_col = col
            break

    # Gender filter if available
    gender_filter = "All"
    if gender_col:
        gender_options = ["All"] + list(data[gender_col].dropna().unique())
        gender_filter = st.selectbox("Filter Gender", gender_options)

    # Detect Age column if present
    age_col = None
    for col in data.columns:
        if "age" in col.lower() and col in numeric_cols:
            age_col = col
            break

    # Detect Customer ID column if present
    id_col = None
    for col in data.columns:
        if any(k in col.lower() for k in ["id", "customer"]):
            id_col = col
            break
    if not id_col:
        id_col = "Row_ID"
        data["Row_ID"] = range(1, len(data) + 1)

    st.markdown("---")
    st.markdown("""
    **👨‍🎓 Project By:**  
    **Krishna Piyush Bhavsar**  
    *Customer Segmentation using K-Means*
    """)

# Apply Gender filter if selected
filtered_df = data.copy()
if gender_col and gender_filter != "All":
    filtered_df = filtered_df[filtered_df[gender_col] == gender_filter]

if len(filtered_df) < 5:
    st.warning("Selected filter leaves too few rows. Using unfiltered dataset.")
    filtered_df = data.copy()

# ---------------------------------------------------------
# K-MEANS MODEL TRAINING
# ---------------------------------------------------------
features = [feature_x, feature_y]
X = filtered_df[features].dropna()
filtered_df = filtered_df.loc[X.index].copy()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=k_clusters, random_state=42, n_init=10)
filtered_df["Cluster"] = kmeans.fit_predict(X_scaled)

# Get Centroids in original scale
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
centroid_df = pd.DataFrame(centroids, columns=features)
centroid_df["Cluster"] = range(k_clusters)

# Intelligent Cluster Naming
if k_clusters == 5:
    med_x = filtered_df[feature_x].median()
    med_y = filtered_df[feature_y].median()

    mapping = {}
    for idx, row in centroid_df.iterrows():
        c_id = int(row["Cluster"])
        cx = row[feature_x]
        cy = row[feature_y]

        if cx > med_x and cy > med_y:
            mapping[c_id] = "🏆 VIP Customers"
        elif cx > med_x and cy <= med_y:
            mapping[c_id] = "💎 Careful Spenders"
        elif cx <= med_x and cy > med_y:
            mapping[c_id] = "⚡ Careless / Impulsive"
        elif cx <= med_x and cy <= med_y:
            mapping[c_id] = "💰 Budget Seekers"
        else:
            mapping[c_id] = "👥 Standard Shoppers"

    # Guarantee 5 unique names if centroids overlap in logic
    if len(set(mapping.values())) < 5:
        sorted_by_y = centroid_df.sort_values(by=feature_y)["Cluster"].tolist()
        mapping = {
            sorted_by_y[0]: "💰 Budget Seekers",
            sorted_by_y[1]: "💎 Careful Spenders",
            sorted_by_y[2]: "👥 Standard Shoppers",
            sorted_by_y[3]: "⚡ Careless / Impulsive",
            sorted_by_y[4]: "🏆 VIP Customers"
        }
else:
    mapping = {i: f"Cluster {i+1}" for i in range(k_clusters)}

filtered_df["Customer Category"] = filtered_df["Cluster"].map(mapping)

# ---------------------------------------------------------
# PAGE 1: 📊 DASHBOARD OVERVIEW
# ---------------------------------------------------------
if selected_page == "📊 Dashboard Overview":
    st.title("📊 Customer Segmentation Dashboard")
    st.write(f"Analyzing **{len(filtered_df)}** records based on **{feature_x}** and **{feature_y}**.")

    # KPI Summary Cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color: #3B82F6;">
            <div class="kpi-title">Total Records</div>
            <div class="kpi-value">{len(filtered_df):,}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        val_x = filtered_df[feature_x].mean()
        prefix = "$" if "income" in feature_x.lower() else ""
        suffix = "k" if "income" in feature_x.lower() else ""
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color: #10B981;">
            <div class="kpi-title">Avg {feature_x[:18]}</div>
            <div class="kpi-value">{prefix}{val_x:.1f}{suffix}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        val_y = filtered_df[feature_y].mean()
        st.markdown(f"""
        <div class="kpi-card" style="--accent-color: #EC4899;">
            <div class="kpi-title">Avg {feature_y[:18]}</div>
            <div class="kpi-value">{val_y:.1f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        if age_col:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-color: #F59E0B;">
                <div class="kpi-title">Avg Customer Age</div>
                <div class="kpi-value">{filtered_df[age_col].mean():.1f} Yrs</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="kpi-card" style="--accent-color: #F59E0B;">
                <div class="kpi-title">Active Clusters (k)</div>
                <div class="kpi-value">{k_clusters} Groups</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts Row
    c1, c2 = st.columns([5, 5])
    with c1:
        st.subheader("Customer Distribution by Category")
        counts = filtered_df["Customer Category"].value_counts().reset_index()
        counts.columns = ["Category", "Count"]

        pie_fig = px.pie(
            counts,
            names="Category",
            values="Count",
            hole=0.45,
            color="Category",
            color_discrete_map=CLUSTER_COLORS
        )
        pie_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#FFFFFF"),
            legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
            height=340,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        st.plotly_chart(pie_fig, use_container_width=True)

    with c2:
        if gender_col:
            st.subheader(f"{gender_col} Distribution")
            g_counts = filtered_df[gender_col].value_counts().reset_index()
            g_counts.columns = [gender_col, "Count"]

            bar_g = px.bar(
                g_counts,
                x=gender_col,
                y="Count",
                color=gender_col,
                color_discrete_sequence=["#EC4899", "#3B82F6", "#10B981"],
                text="Count"
            )
            bar_g.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFFFFF"),
                xaxis=dict(showgrid=False),
                yaxis=dict(gridcolor="#334155"),
                height=340,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(bar_g, use_container_width=True)
        else:
            st.subheader(f"{feature_x} Distribution")
            hist_fig = px.histogram(
                filtered_df,
                x=feature_x,
                color="Customer Category",
                color_discrete_map=CLUSTER_COLORS,
                nbins=15
            )
            hist_fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFFFFF"),
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"),
                height=340,
                margin=dict(l=10, r=10, t=10, b=10)
            )
            st.plotly_chart(hist_fig, use_container_width=True)

    st.info("""
    **🎯 Project Objective:**  
    This dashboard applies **K-Means Clustering** to divide shoppers into distinct segments. Retail managers can use these segments to create tailored marketing offers and maximize revenue.
    """)

# ---------------------------------------------------------
# PAGE 2: 📂 UPLOAD & PREVIEW DATASET
# ---------------------------------------------------------
elif selected_page == "📂 Upload & Preview Dataset":
    st.title("📂 Upload & Inspect Dataset")
    st.write("You can analyze the built-in Mall Customers dataset or upload any customer dataset of your choice.")

    u_col1, u_col2 = st.columns([6, 4])
    with u_col1:
        st.markdown("#### 📤 Upload Your File")
        main_upload = st.file_uploader(
            "Upload CSV or Excel file here:",
            type=["csv", "xlsx", "xls"],
            key="main_uploader"
        )
        if main_upload is not None:
            st.success(f"File **{main_upload.name}** uploaded! Use the sidebar to switch active source or select feature columns.")
        else:
            st.caption("Currently using: **" + ("Custom Uploaded Dataset" if is_custom_dataset else "Default Mall Customers (dataset/Mall_Customers.csv)") + "**")

    with u_col2:
        st.markdown("#### ℹ️ Dataset Quick Facts")
        st.markdown(f"""
        - **Total Rows:** {len(data):,}
        - **Total Columns:** {len(data.columns)}
        - **Numeric Columns:** {len(numeric_cols)} ({', '.join(numeric_cols[:4])}...)
        - **Selected Feature 1 (X):** `{feature_x}`
        - **Selected Feature 2 (Y):** `{feature_y}`
        """)

    st.markdown("---")
    st.markdown("#### 📋 Dataset Preview (First 10 Rows)")
    st.dataframe(data.head(10), use_container_width=True)

    with st.expander("📊 View Summary Statistics (Mean, Min, Max, Quantiles)"):
        st.dataframe(data.describe().round(2), use_container_width=True)

# ---------------------------------------------------------
# PAGE 3: 📉 FINDING OPTIMAL K
# ---------------------------------------------------------
elif selected_page == "📉 Finding Optimal K (Elbow Method)":
    st.title("📉 Finding the Best Number of Clusters (k)")
    st.write("In Unsupervised Learning, we use the **Elbow Method** and **Silhouette Score** to determine the optimal number of groups.")

    tab1, tab2 = st.tabs(["📊 Elbow Method Graph", "🎯 Silhouette Score Graph"])

    with tab1:
        wcss = []
        for i in range(1, 11):
            m = KMeans(n_clusters=i, random_state=42, n_init=10)
            m.fit(X_scaled)
            wcss.append(m.inertia_)

        elbow_fig = px.line(
            x=list(range(1, 11)),
            y=wcss,
            markers=True,
            labels={"x": "Number of Clusters (k)", "y": "WCSS (Within-Cluster Sum of Squares)"},
            title="Elbow Method Curve"
        )
        elbow_fig.update_traces(line_color="#3B82F6", marker=dict(size=9, color="#EC4899"))
        elbow_fig.add_vline(x=5, line_dash="dash", line_color="#10B981", annotation_text="Elbow Point (k=5)")
        elbow_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            font=dict(color="#FFFFFF"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            height=400
        )
        st.plotly_chart(elbow_fig, use_container_width=True)

        st.success("""
        **💡 How to explain the Elbow Method in your viva/exam:**  
        - WCSS measures the compactness of points within each cluster.
        - As $k$ increases, WCSS naturally drops.
        - The bend ("elbow") indicates the point of optimal trade-off between cluster tightness and model simplicity.
        """)

    with tab2:
        sil_scores = []
        for i in range(2, 11):
            m = KMeans(n_clusters=i, random_state=42, n_init=10)
            labels = m.fit_predict(X_scaled)
            sil_scores.append(silhouette_score(X_scaled, labels))

        sil_fig = px.line(
            x=list(range(2, 11)),
            y=sil_scores,
            markers=True,
            labels={"x": "Number of Clusters (k)", "y": "Silhouette Score (0 to 1)"},
            title="Silhouette Score Analysis"
        )
        best_k_idx = int(np.argmax(sil_scores)) + 2
        sil_fig.update_traces(line_color="#10B981", marker=dict(size=9, color="#F59E0B"))
        sil_fig.add_vline(x=best_k_idx, line_dash="dash", line_color="#EC4899", annotation_text=f"Peak Score at k={best_k_idx}")
        sil_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            font=dict(color="#FFFFFF"),
            xaxis=dict(gridcolor="#334155"),
            yaxis=dict(gridcolor="#334155"),
            height=400
        )
        st.plotly_chart(sil_fig, use_container_width=True)

        st.info(f"""
        **💡 Silhouette Score Explanation:**  
        The Silhouette Score measures how well separated each cluster is from its neighbors.  
        The highest score is reached at **k = {best_k_idx}** ({max(sil_scores):.3f}), confirming distinct customer groupings.
        """)

# ---------------------------------------------------------
# PAGE 4: 🎯 CUSTOMER CLUSTERS
# ---------------------------------------------------------
elif selected_page == "🎯 Customer Clusters (Visuals)":
    st.title("🎯 Customer Clusters Visualization")
    st.write(f"Interactive scatter plots showing how customers are grouped by **{feature_x}** and **{feature_y}**.")

    col_view, col_info = st.columns([7, 3])

    with col_view:
        hover_items = [c for c in [id_col, age_col, gender_col] if c and c in filtered_df.columns]

        fig2d = px.scatter(
            filtered_df,
            x=feature_x,
            y=feature_y,
            color="Customer Category",
            color_discrete_map=CLUSTER_COLORS,
            hover_data=hover_items,
            title=f"Customer Clusters: {feature_x} vs {feature_y}",
            opacity=0.85
        )
        fig2d.update_traces(marker=dict(size=10, line=dict(width=1, color='#FFFFFF')))

        # Add Centroids on Plot
        fig2d.add_trace(
            go.Scatter(
                x=centroid_df[feature_x],
                y=centroid_df[feature_y],
                mode="markers+text",
                marker=dict(symbol="star", size=16, color="#FFFFFF", line=dict(width=2, color="#000000")),
                text=[f"Centroid {i+1}" for i in centroid_df["Cluster"]],
                textposition="top center",
                name="Centroids ⭐"
            )
        )

        fig2d.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(30, 41, 59, 0.4)",
            font=dict(color="#FFFFFF"),
            xaxis=dict(gridcolor="#334155", title=feature_x),
            yaxis=dict(gridcolor="#334155", title=feature_y),
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
            height=480
        )
        st.plotly_chart(fig2d, use_container_width=True)

    with col_info:
        st.subheader("⭐ Cluster Centers")
        st.caption("Average values for each cluster center point:")
        display_centroids = centroid_df.copy()
        display_centroids["Category"] = display_centroids["Cluster"].map(mapping)
        st.dataframe(
            display_centroids[["Category", feature_x, feature_y]].round(1),
            hide_index=True,
            use_container_width=True
        )

        st.markdown(f"""
        **💡 How to read this plot:**
        - Each colored dot is an individual customer.
        - The white stars (**⭐**) show the mathematical center (centroid) for each group.
        - Any new customer is assigned to the nearest centroid.
        """)

    # 3D View (if Age is available or feature exists)
    if age_col and age_col in filtered_df.columns:
        with st.expander(f"🌐 Click to View 3D Customer Clusters ({feature_x} × {feature_y} × {age_col})"):
            fig3d = px.scatter_3d(
                filtered_df,
                x=feature_x,
                y=feature_y,
                z=age_col,
                color="Customer Category",
                color_discrete_map=CLUSTER_COLORS,
                hover_data=hover_items
            )
            fig3d.update_traces(marker=dict(size=5))
            fig3d.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#FFFFFF"),
                scene=dict(
                    xaxis=dict(backgroundcolor="#1E293B", gridcolor="#334155", title=feature_x),
                    yaxis=dict(backgroundcolor="#1E293B", gridcolor="#334155", title=feature_y),
                    zaxis=dict(backgroundcolor="#1E293B", gridcolor="#334155", title=age_col)
                ),
                height=500
            )
            st.plotly_chart(fig3d, use_container_width=True)

# ---------------------------------------------------------
# PAGE 5: 👥 CUSTOMER GROUPS & STRATEGIES
# ---------------------------------------------------------
elif selected_page == "👥 Customer Groups & Strategies":
    st.title("👥 Customer Groups & Marketing Strategies")
    st.write("Practical business recommendations tailored for each identified customer segment.")

    profiles = [
        {
            "name": "🏆 VIP Customers",
            "color": "#10B981",
            "traits": f"High {feature_x} & High {feature_y}",
            "desc": "The most profitable customers. They appreciate quality, responsive customer service, and shop frequently.",
            "strategy": "Enroll in VIP Club, early access to new product arrivals, dedicated support, and prestige rewards."
        },
        {
            "name": "💎 Careful Spenders",
            "color": "#6366F1",
            "traits": f"High {feature_x} & Low {feature_y}",
            "desc": "Customers with high purchasing power who spend cautiously and research thoroughly before buying.",
            "strategy": "Emphasize product quality, customer reviews, warranties, and premium bundle value discounts."
        },
        {
            "name": "⚡ Careless / Impulsive",
            "color": "#EC4899",
            "traits": f"Low/Moderate {feature_x} & High {feature_y}",
            "desc": "Trend-conscious shoppers responsive to new launches, buzz, viral items, and limited drops.",
            "strategy": "Target with social media ads, countdown flash sales, and trending product alerts."
        },
        {
            "name": "💰 Budget Seekers",
            "color": "#F59E0B",
            "traits": f"Low {feature_x} & Low {feature_y}",
            "desc": "Price-sensitive shoppers looking for discounts, clearance sales, and maximum value per dollar.",
            "strategy": "Offer discount vouchers, cashback promotions, clearance flyers, and BOGO deals."
        },
        {
            "name": "👥 Standard Shoppers",
            "color": "#06B6D4",
            "traits": f"Average {feature_x} & Average {feature_y}",
            "desc": "Middle-market shoppers forming the steady backbone volume of retail footfall.",
            "strategy": "Standard store loyalty points, seasonal holiday offers, and reliable service."
        }
    ]

    for p in profiles:
        st.markdown(f"""
        <div class="segment-card" style="--seg-color: {p['color']};">
            <div class="segment-title">{p['name']} <span style="font-size: 0.9rem; color: #94A3B8; font-weight: normal;">— {p['traits']}</span></div>
            <p style="color: #CBD5E1; margin: 8px 0; font-size: 0.95rem;">{p['desc']}</p>
            <div style="background: rgba(255,255,255,0.05); padding: 10px 14px; border-radius: 8px; font-size: 0.9rem; color: #E2E8F0;">
                <b style="color: #38BDF8;">📢 Recommended Strategy:</b> {p['strategy']}
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# PAGE 6: 🤖 PREDICT NEW CUSTOMER
# ---------------------------------------------------------
elif selected_page == "🤖 Predict New Customer":
    st.title("🤖 Predict Customer Category")
    st.write(f"Enter values for **{feature_x}** and **{feature_y}** to classify any customer in real time.")

    # Calculate dynamic ranges from active dataset
    min_x, max_x = float(filtered_df[feature_x].min()), float(filtered_df[feature_x].max())
    min_y, max_y = float(filtered_df[feature_y].min()), float(filtered_df[feature_y].max())
    mean_x, mean_y = float(filtered_df[feature_x].mean()), float(filtered_df[feature_y].mean())

    # Demo Buttons
    st.write("**💡 Quick Test Presets:**")
    b1, b2, b3, b4 = st.columns(4)
    with b1:
        if st.button("High X & High Y"):
            st.session_state["p_val_x"] = float(min_x + (max_x - min_x) * 0.8)
            st.session_state["p_val_y"] = float(min_y + (max_y - min_y) * 0.8)
    with b2:
        if st.button("High X & Low Y"):
            st.session_state["p_val_x"] = float(min_x + (max_x - min_x) * 0.8)
            st.session_state["p_val_y"] = float(min_y + (max_y - min_y) * 0.2)
    with b3:
        if st.button("Low X & High Y"):
            st.session_state["p_val_x"] = float(min_x + (max_x - min_x) * 0.2)
            st.session_state["p_val_y"] = float(min_y + (max_y - min_y) * 0.8)
    with b4:
        if st.button("Low X & Low Y"):
            st.session_state["p_val_x"] = float(min_x + (max_x - min_x) * 0.2)
            st.session_state["p_val_y"] = float(min_y + (max_y - min_y) * 0.2)

    col_in, col_res = st.columns([5, 5])

    with col_in:
        st.subheader("Customer Input")
        in_x = st.slider(
            f"{feature_x}:",
            min_value=round(min_x, 1),
            max_value=round(max_x, 1),
            value=round(st.session_state.get("p_val_x", mean_x), 1)
        )
        in_y = st.slider(
            f"{feature_y}:",
            min_value=round(min_y, 1),
            max_value=round(max_y, 1),
            value=round(st.session_state.get("p_val_y", mean_y), 1)
        )

        pred_btn = st.button("🔍 Predict Segment", use_container_width=True)

    with col_res:
        st.subheader("Prediction Result")
        scaled_in = scaler.transform([[in_x, in_y]])
        pred_c = kmeans.predict(scaled_in)[0]
        pred_label = mapping[pred_c]
        p_color = CLUSTER_COLORS.get(pred_label, "#3B82F6")

        st.markdown(f"""
        <div class="pred-box" style="border-color: {p_color};">
            <div style="color: #94A3B8; font-size: 0.9rem; text-transform: uppercase;">Assigned Customer Group</div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #FFFFFF; margin: 10px 0;">{pred_label}</div>
            <div style="color: #CBD5E1; font-size: 0.9rem;">
                {feature_x}: <b>{in_x}</b> | {feature_y}: <b>{in_y}</b>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if "VIP" in pred_label:
            st.success("🎯 **Recommended Offer:** Enroll in VIP Club & provide early access to new arrivals.")
        elif "Careful" in pred_label:
            st.info("🎯 **Recommended Offer:** Send verified product reviews, warranties, and bundle discount value.")
        elif "Careless" in pred_label or "Impulsive" in pred_label:
            st.warning("🎯 **Recommended Offer:** Send flash sale vouchers, limited-time countdowns, and trending items.")
        elif "Budget" in pred_label:
            st.error("🎯 **Recommended Offer:** Send discount clearance coupons and BOGO (Buy-One-Get-One) deals.")
        else:
            st.info("🎯 **Recommended Offer:** Send standard loyalty points promotion and seasonal discount email.")

# ---------------------------------------------------------
# PAGE 7: 📋 VIEW & DOWNLOAD DATA
# ---------------------------------------------------------
elif selected_page == "📋 View & Download Data":
    st.title("📋 Customer Data & Export")
    st.write("Browse the active dataset with the newly computed **Customer Category** and **Cluster ID**.")

    search_id = st.text_input(f"🔍 Filter by {id_col} (optional):", placeholder="e.g. 15")
    
    view_data = filtered_df.copy()
    if search_id.strip():
        try:
            target_id = int(search_id.strip())
            view_data = view_data[view_data[id_col] == target_id]
        except:
            st.warning("Please enter a numeric ID.")

    cols_to_show = [c for c in [id_col, gender_col, age_col, feature_x, feature_y, "Customer Category", "Cluster"] if c and c in view_data.columns]
    st.dataframe(view_data[cols_to_show], hide_index=True, use_container_width=True)

    # Download Button
    csv_bytes = filtered_df[cols_to_show].to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Segmented Customers (CSV)",
        data=csv_bytes,
        file_name="segmented_customers.csv",
        mime="text/csv"
    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<center style='color: #64748B; font-size: 0.85rem;'>College Data Science Project • Developed by <b>Krishna Piyush Bhavsar</b> • Built with Streamlit & Scikit-Learn</center>",
    unsafe_allow_html=True
)