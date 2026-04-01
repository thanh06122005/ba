import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────
st.set_page_config(
    page_title="InsightWave | Retention Radar",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium CSS Styling
st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 28px; font-weight: 700; color: #1E3A8A; }
    .main { background-color: #F8FAFC; }
    .stButton>button { border-radius: 10px; height: 3em; background-color: #2563EB; color: white; font-weight: 600; border: none; }
    .strategy-card { background: white; padding: 25px; border-radius: 15px; border: 1px solid #E2E8F0; box-shadow: 0 4px 15px -3px rgba(0, 0, 0, 0.1); }
    .sidebar-header { font-size: 1.4rem; font-weight: 800; color: #1E3A8A; margin-bottom: 10px; }
    .highlight { color: #2563EB; font-weight: 700; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA SYNC ENGINE (Matches Colab Output)
# ─────────────────────────────────────────
@st.cache_data
def load_and_sync_data():
    try:
        # Load exported results from Colab
        df_prob = pd.read_csv("churn_probabilities.csv")
        df_geo = pd.read_csv("telco_preprocessed.csv")
        
        # Merging geo-data for mapping (Syncing with Colab Step 6)
        # Note: Handled potential column suffixing (_x, _y) from pandas merge
        merged = df_prob.merge(
            df_geo[['CustomerID', 'Latitude', 'Longitude', 'City', 'Gender', 'Senior Citizen', 'Partner', 'Dependents']],
            on='CustomerID', how='left'
        )
        
        # Standardizing names if suffixes exist
        if 'Monthly Charges_x' in merged.columns:
            merged.rename(columns={'Monthly Charges_x': 'Monthly Charges', 'Churn Label_x': 'Churn Label'}, inplace=True)
        
        return merged
    except Exception as e:
        st.error(f"Error loading data: {e}. Ensure 'churn_probabilities.csv' and 'telco_preprocessed.csv' are present.")
        return pd.DataFrame()

df = load_and_sync_data()

# ─────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-header">📡 InsightWave Radar</p>', unsafe_allow_html=True)
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/51/IBM_logo.svg/200px-IBM_logo.svg.png", width=70)
    st.markdown("**Enterprise Retention Intelligence**")
    st.divider()

    menu = st.selectbox("Navigation Menu", 
        ["🏠 Executive Dashboard", "🎯 Risk Intelligence", "🗺️ Churn Hotspots", "📈 Revenue Simulation"])
    
    st.divider()
    st.caption("v2.1 · Powered by CatBoost/LGBM Ensemble")
    st.info("System Synced: Probabilities reflect latest model inference.")

# ─────────────────────────────────────────
# PAGE 1: EXECUTIVE DASHBOARD
# ─────────────────────────────────────────
if menu == "🏠 Executive Dashboard":
    st.title("📊 Executive Retention Summary")
    st.markdown("Real-time analysis of customer churn risk and revenue exposure.")
    
    # KPI Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Customers", f"{len(df):,}")
    with m2:
        # Check if Churn Label exists
        c_label = 'Churn Label' if 'Churn Label' in df.columns else 'Churn'
        churn_rate = (df[c_label] == 'Yes').mean() if 'Yes' in df[c_label].values else df[c_label].mean()
        st.metric("Actual Churn Rate", f"{churn_rate:.1%}", delta="-0.8%")
    with m3:
        high_risk_count = len(df[df['Risk_Tier'] == 'High Risk'])
        st.metric("High-Risk Segments", f"{high_risk_count:,}", delta_color="inverse")
    with m4:
        avg_rev = df['Monthly Charges'].mean()
        st.metric("Avg. Monthly Bill", f"${avg_rev:,.2f}")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Risk Tier Distribution")
        fig_risk = px.pie(df, names='Risk_Tier', hole=0.5,
                         color_discrete_map={
                             'High Risk': '#EF4444', 
                             'Medium Risk': '#F59E0B', 
                             'Low Risk': '#10B981', 
                             'Very Low Risk': '#3B82F6'
                         })
        fig_risk.update_layout(showlegend=True, height=450, margin=dict(t=0, b=0, l=0, r=0))
        st.plotly_chart(fig_risk, use_container_width=True)

    with col_right:
        st.subheader("Churn Probability Density")
        fig_hist = px.histogram(df, x="Churn_Probability", nbins=50, 
                               color_discrete_sequence=['#2563EB'],
                               opacity=0.7)
        fig_hist.update_layout(height=450, xaxis_title="Predicted Probability (0.0 - 1.0)", yaxis_title="Customer Count")
        st.plotly_chart(fig_hist, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2: REVENUE SIMULATION (Fixed as requested)
# ─────────────────────────────────────────
elif menu == "📈 Revenue Simulation":
    st.title("📈 Monte Carlo Strategic Simulator")
    st.markdown("Simulate financial impacts of retention strategies using stochastic modeling.")
    
    st.markdown('<div class="strategy-card">', unsafe_allow_html=True)
    c1, c2 = st.columns([1, 2])
    
    with c1:
        st.markdown("### ⚙️ Strategy Configuration")
        # FIXED: Strategy Name Input
        strategy_name = st.text_input("📝 Strategy Name", value="Retention Campaign Alpha")
        
        # FIXED: Simulation Slider (200 - 1000)
        n_sim = st.slider("🎲 Simulations", min_value=200, max_value=1000, value=500, step=100)
        
        st.divider()
        st.markdown("**Intervention Parameters**")
        discount_rate = st.slider("Discount Rate (%)", 0, 40, 15) / 100
        retention_boost = st.slider("Retention Success Rate (%)", 0, 100, 60) / 100
        
        run_sim = st.button("🚀 Run Monte Carlo Simulation", use_container_width=True)
    
    with c2:
        if run_sim:
            # MONTE CARLO ENGINE (Synced with Colab Simulation Logic)
            baseline_revenues = []
            strategy_revenues = []
            
            with st.spinner(f"Running {n_sim} scenarios..."):
                for _ in range(n_sim):
                    # 1. Baseline: Current model probabilities
                    churn_base = np.random.binomial(1, df['Churn_Probability'])
                    baseline_revenues.append(df[churn_base == 0]['Monthly Charges'].sum())
                    
                    # 2. Strategy: Intervention on High/Medium Risk tiers
                    temp_df = df.copy()
                    
                    # Identify target customers (High Risk)
                    target_mask = temp_df['Risk_Tier'] == 'High Risk'
                    
                    # Apply probability reduction (Boost effectiveness)
                    temp_df.loc[target_mask, 'Churn_Probability'] *= (1 - retention_boost)
                    
                    # Run simulation with adjusted probabilities
                    churn_strat = np.random.binomial(1, temp_df['Churn_Probability'].clip(0, 1))
                    
                    # Calculate revenue (Applying discount cost to retained high-risk customers)
                    total_rev = temp_df[churn_strat == 0]['Monthly Charges'].sum()
                    # Apply discount cost to targeted group
                    retained_targeted = (target_mask) & (churn_strat == 0)
                    discount_cost = (temp_df.loc[retained_targeted, 'Monthly Charges'] * discount_rate).sum()
                    
                    strategy_revenues.append(total_rev - discount_cost)

            # Results Display
            res_1, res_2 = st.columns(2)
            res_1.metric("Baseline Avg. Revenue", f"${np.mean(baseline_revenues):,.0f}/mo")
            res_2.metric(f"Projected: {strategy_name}", f"${np.mean(strategy_revenues):,.0f}/mo", 
                         delta=f"${np.mean(strategy_revenues) - np.mean(baseline_revenues):+,.0f}")
            
            # Distribution Plot
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Violin(y=baseline_revenues, name="Baseline", line_color="#94A3B8", box_visible=True))
            fig_dist.add_trace(go.Violin(y=strategy_revenues, name=strategy_name, line_color="#2563EB", box_visible=True))
            fig_dist.update_layout(title="Revenue Probability Distribution Comparison", height=450, yaxis_title="Monthly Revenue ($)")
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.info("👈 Configure the strategy parameters and click **Run Simulation** to project financial outcomes.")
    st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# PAGE 3: RISK INTELLIGENCE (CUSTOMER LOOKUP)
# ─────────────────────────────────────────
elif menu == "🎯 Risk Intelligence":
    st.title("🎯 Customer Risk Profile")
    st.markdown("Deep-dive into individual customer behavioral risk scores.")
    
    search_id = st.text_input("Search Customer ID (e.g., 3668-QPYBK)", "").upper()
    
    if search_id:
        cust = df[df['CustomerID'] == search_id]
        if not cust.empty:
            c = cust.iloc[0]
            
            # Styling card
            st.markdown(f'<div class="strategy-card">', unsafe_allow_html=True)
            l, r = st.columns([1, 1])
            with l:
                st.subheader(f"ID: {c['CustomerID']}")
                risk_color = "#EF4444" if c['Risk_Tier'] == "High Risk" else "#10B981"
                st.markdown(f"**Risk Status:** <span style='color:{risk_color}; font-size: 20px; font-weight:700;'>{c['Risk_Tier']}</span>", unsafe_allow_html=True)
                st.write(f"**Contract:** {c['Contract']}")
                st.write(f"**Internet:** {c['Internet Service']}")
                st.write(f"**Payment:** {c['Payment Method']}")
            
            with r:
                # Gauge Chart for Probability
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = c['Churn_Probability'] * 100,
                    title = {'text': "Churn Probability Score"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': risk_color},
                        'steps': [
                            {'range': [0, 50], 'color': "#E2E8F0"},
                            {'range': [50, 100], 'color': "#FEE2E2"}]
                    },
                    number = {'suffix': "%"}
                ))
                fig_gauge.update_layout(height=250, margin=dict(t=30, b=0, l=30, r=30))
                st.plotly_chart(fig_gauge, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.error("Customer ID not found. Please verify the ID.")

# ─────────────────────────────────────────
# PAGE 4: CHURN HOTSPOTS (GEO MAP)
# ─────────────────────────────────────────
elif menu == "🗺️ Churn Hotspots":
    st.title("🗺️ Geographic Risk Hotspots")
    st.markdown("Visualizing churn risk density across California service areas.")
    
    c1, c2 = st.columns(2)
    with c1:
        tier_filter = st.multiselect("Filter by Risk Tier", df['Risk_Tier'].unique(), default=['High Risk', 'Medium Risk'])
    with c2:
        price_filter = st.slider("Min. Monthly Charges ($)", 0, 120, 0)

    map_df = df[(df['Risk_Tier'].isin(tier_filter)) & (df['Monthly Charges'] >= price_filter)]
    
    fig_map = px.scatter_mapbox(map_df, 
                                lat="Latitude", lon="Longitude", 
                                color="Churn_Probability", 
                                size="Monthly Charges",
                                color_continuous_scale=px.colors.sequential.YlOrRd,
                                size_max=12, zoom=5, height=750,
                                hover_name="CustomerID",
                                hover_data=["Risk_Tier", "Monthly Charges", "Contract"])
    
    fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_map, use_container_width=True)