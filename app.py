import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────
# PAGE SETTINGS
# ─────────────────────────────────────────
st.set_page_config(page_title="InsightWave | Retention Radar", layout="wide")

# Modern Professional Styling
st.markdown("""
<style>
    .main { background-color: #F8FAFC; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    .strategy-container { background-color: #ffffff; padding: 25px; border-radius: 15px; border: 1px solid #E2E8F0; }
    h1, h2, h3 { color: #1E3A8A; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# DATA LOADING & SYNCING
# ─────────────────────────────────────────
@st.cache_data
def get_data():
    try:
        # Sync with your Colab outputs
        df_p = pd.read_csv("churn_probabilities.csv")
        df_g = pd.read_csv("telco_preprocessed.csv")
        
        # Standardize Columns (Handling _x suffixes from Colab merge)
        df = df_p.merge(df_g[['CustomerID', 'Latitude', 'Longitude', 'Gender', 'Contract']], on='CustomerID', how='left')
        
        # Mapping column names to be clean
        cols = {
            'Monthly Charges_x': 'Monthly_Charges',
            'Churn Label_x': 'Churn_Label',
            'Tenure Months_x': 'Tenure'
        }
        df.rename(columns={k: v for k, v in cols.items() if k in df.columns}, inplace=True)
        
        # Fallback if names are different
        if 'Monthly Charges' in df.columns: df.rename(columns={'Monthly Charges': 'Monthly_Charges'}, inplace=True)
        if 'Churn Label' in df.columns: df.rename(columns={'Churn Label': 'Churn_Label'}, inplace=True)
        
        return df
    except Exception as e:
        st.error(f"Data Sync Error: {e}")
        return pd.DataFrame()

df_raw = get_data()

# ─────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────
with st.sidebar:
    st.title("📡 InsightWave")
    st.markdown("---")
    menu = st.radio("Navigation", ["🏠 Overview Dashboard", "📈 Strategic Simulator"])
    st.markdown("---")
    st.caption("v3.0 - Enterprise Predictive Engine")

# ─────────────────────────────────────────
# PAGE 1: OVERVIEW DASHBOARD
# ─────────────────────────────────────────
if menu == "🏠 Overview Dashboard":
    st.title("📊 Customer Churn Overview")
    
    if df_raw.empty:
        st.warning("Please upload 'churn_probabilities.csv' and 'telco_preprocessed.csv' to the app directory.")
    else:
        # KPI Metrics
        c1, c2, c3, c4 = st.columns(4)
        total_cust = len(df_raw)
        actual_churn = len(df_raw[df_raw['Churn_Label'] == 'Yes'])
        churn_rate = actual_churn / total_cust
        revenue_at_risk = df_raw[df_raw['Risk_Tier'] == 'High Risk']['Monthly_Charges'].sum()

        c1.metric("Total Customers", f"{total_cust:,}")
        c2.metric("Actual Churn Rate", f"{churn_rate:.1%}")
        c3.metric("High Risk Customers", len(df_raw[df_raw['Risk_Tier'] == 'High Risk']))
        c4.metric("Revenue at Risk", f"${revenue_at_risk:,.0f}/mo")

        st.divider()

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Risk Tier Distribution")
            # Handling potential value mismatch
            tier_counts = df_raw['Risk_Tier'].value_counts().reset_index()
            fig_pie = px.pie(tier_counts, values='count', names='Risk_Tier', hole=0.4,
                            color_discrete_sequence=px.colors.qualitative.Pastel)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_b:
            st.subheader("Tenure vs. Churn Probability")
            fig_scatter = px.scatter(df_raw.sample(min(1000, total_cust)), 
                                    x="Tenure", y="Churn_Probability", 
                                    color="Risk_Tier", opacity=0.5)
            st.plotly_chart(fig_scatter, use_container_width=True)

# ─────────────────────────────────────────
# PAGE 2: STRATEGIC SIMULATOR (PREDICT)
# ─────────────────────────────────────────
elif menu == "📈 Strategic Simulator":
    st.title("📈 Retention & Pricing Predictor")
    st.markdown("Enter your strategy details below to see the impact on Churn and Revenue.")

    if df_raw.empty:
        st.stop()

    with st.container():
        st.markdown('<div class="strategy-container">', unsafe_allow_html=True)
        col_inp, col_out = st.columns([1, 2])

        with col_inp:
            st.subheader("Strategy Input")
            strat_name = st.text_input("Strategy Name", "FY25 Pricing Adjustment")
            
            # Pricing Strategy Impact
            price_adj = st.slider("Price Adjustment (%)", -30, 30, 0, help="Negative for Discount, Positive for Price Hike")
            
            # Retention Strategy Impact
            ret_boost = st.slider("Retention Success Rate (%)", 0, 100, 20, help="Expected success of proactive retention calls/service")
            
            n_sim = st.slider("Simulations (Accuracy)", 200, 1000, 500)
            
            run = st.button("🚀 Run Prediction", use_container_width=True)

        with col_out:
            if run:
                # SIMULATION LOGIC
                # Elasticity: How much Churn Prob changes per 1% price change (Estimate: 0.5)
                elasticity = 0.5 
                
                sim_churn_delta = []
                sim_rev_delta = []

                with st.spinner("Processing Strategy Impact..."):
                    for _ in range(n_sim):
                        temp = df_raw.copy()
                        
                        # Apply Pricing Strategy to Probabilities
                        # Price hike increases prob, Discount decreases it
                        temp['New_Prob'] = temp['Churn_Probability'] * (1 + (price_adj/100) * elasticity)
                        
                        # Apply Retention Boost (only for targeted high risk)
                        temp.loc[temp['Risk_Tier'] == 'High Risk', 'New_Prob'] *= (1 - (ret_boost/100))
                        temp['New_Prob'] = temp['New_Prob'].clip(0, 1)

                        # Simulate Events
                        baseline_churn = np.random.binomial(1, temp['Churn_Probability']).sum()
                        new_churn = np.random.binomial(1, temp['New_Prob']).sum()
                        
                        # Revenue Calculation
                        baseline_rev = temp[np.random.binomial(1, temp['Churn_Probability']) == 0]['Monthly_Charges'].sum()
                        # Apply new price to those who stay
                        new_rev = temp[np.random.binomial(1, temp['New_Prob']) == 0]['Monthly_Charges'].sum() * (1 + price_adj/100)
                        
                        sim_churn_delta.append(new_churn - baseline_churn)
                        sim_rev_delta.append(new_rev - baseline_rev)

                # Show Results
                st.subheader(f"Impact Analysis: {strat_name}")
                res_a, res_b = st.columns(2)
                
                churn_impact = np.mean(sim_churn_delta)
                rev_impact = np.mean(sim_rev_delta)
                
                res_a.metric("Churn Change (Customers)", 
                            f"{churn_impact:,.0f}", 
                            delta=f"{churn_impact:,.0f}", delta_color="inverse")
                
                res_b.metric("Monthly Revenue Change", 
                            f"${rev_impact:,.0f}", 
                            delta=f"${rev_impact:,.0f}")

                # Visualization
                fig_res = go.Figure()
                fig_res.add_trace(go.Histogram(x=sim_rev_delta, name="Revenue Delta", marker_color='#3B82F6'))
                fig_res.update_layout(title="Revenue Change Distribution", xaxis_title="Revenue Delta ($)", yaxis_title="Frequency")
                st.plotly_chart(fig_res, use_container_width=True)
            else:
                st.info("Adjust the parameters on the left and click **Run Prediction** to see results.")
        st.markdown('</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────
# FIXING THE ERROR: GEO MAP / CUSTOMER LOOKUP (Simplified to avoid crash)
# ─────────────────────────────────────────
st.divider()
if not df_raw.empty:
    st.subheader("📍 Customer Risk Hotspots")
    # Fix the Multiselect Error: Use unique values and dynamic default
    available_tiers = df_raw['Risk_Tier'].unique().tolist()
    default_tiers = [t for t in ['High Risk', 'Medium Risk'] if t in available_tiers]
    
    tier_sel = st.multiselect("Filter Map by Risk Tier", available_tiers, default=default_tiers)
    
    map_df = df_raw[df_raw['Risk_Tier'].isin(tier_sel)]
    if not map_df.empty:
        fig_map = px.scatter_mapbox(map_df, lat="Latitude", lon="Longitude", color="Churn_Probability",
                                   size="Monthly_Charges", hover_name="CustomerID", zoom=4, height=500)
        fig_map.update_layout(mapbox_style="carto-positron", margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(fig_map, use_container_width=True)
