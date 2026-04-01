import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Telco Churn Intelligence",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 26px; font-weight: 700; }
.block-container { padding-top: 1.5rem; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    padding: 8px 20px; border-radius: 8px;
    background: #f0f2f6; font-weight: 500;
}
.stTabs [aria-selected="true"] {
    background: #1f77b4 !important; color: white !important;
}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Loading data...")
def load_data():
    cp = pd.read_csv("churn_probabilities.csv")
    pp = pd.read_csv("telco_preprocessed.csv")
    df = cp.merge(
        pp[["CustomerID", "Latitude", "Longitude",
            "Gender", "Senior Citizen", "Churn"]],
        on="CustomerID", how="left"
    )
    return df


df = load_data()

# ── SIDEBAR ──
with st.sidebar:
    st.markdown("## 📡 Telco Churn Intel")
    st.caption("IBM Telco · 7,043 customers")
    st.divider()
    page = st.radio(
        "", ["📊 Overview Dashboard", "🎯 Strategy Simulator"],
        label_visibility="collapsed"
    )

# ══════════════════════════════════════════
# PAGE 1: OVERVIEW
# ══════════════════════════════════════════
if page == "📊 Overview Dashboard":
    st.title("📊 Customer Churn Overview")

    total        = len(df)
    actual_churn = df["Churn Label"].eq("Yes").sum()
    churn_rate   = actual_churn / total * 100
    high_risk    = df["Risk_Tier"].eq("High Risk").sum()
    rev_at_risk  = (df[df["Risk_Tier"].isin(["High Risk", "Medium Risk"])]
                    ["Monthly Charges"].sum() * 12)
    avg_prob     = df["Churn_Probability"].mean() * 100

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("👥 Total Customers",    f"{total:,}")
    c2.metric("🔴 Actual Churned",     f"{actual_churn:,}", f"{churn_rate:.1f}%")
    c3.metric("🤖 Model Predicted",    f"{df['Predicted_Churn'].sum():,}")
    c4.metric("⚠️ High Risk",          f"{high_risk:,}")
    c5.metric("💸 Annual Rev at Risk", f"${rev_at_risk/1e6:.2f}M")

    st.divider()

    # Row 2
    col1, col2, col3 = st.columns([1.1, 1.1, 1.8])

    with col1:
        st.subheader("Risk Tiers")
        risk_order  = ["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"]
        risk_colors = ["#EF5350", "#FFA726", "#66BB6A", "#42A5F5"]
        rc = df["Risk_Tier"].value_counts().reindex(risk_order).reset_index()
        rc.columns = ["Tier", "Count"]
        fig = px.bar(rc, x="Tier", y="Count", color="Tier",
                     color_discrete_sequence=risk_colors,
                     text="Count", height=300)
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          margin=dict(t=5, b=0),
                          xaxis_title="", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Churn by Contract")
        cc = (df.groupby("Contract")["Churn"]
              .mean()
              .sort_values(ascending=False)
              .reset_index())
        cc["pct"] = (cc["Churn"] * 100).round(1)
        fig2 = px.bar(cc, x="Contract", y="pct",
                      color="pct",
                      color_continuous_scale=["#66BB6A", "#FFA726", "#EF5350"],
                      text=cc["pct"].apply(lambda x: f"{x:.1f}%"),
                      height=300)
        fig2.update_traces(textposition="outside")
        fig2.update_layout(showlegend=False, coloraxis_showscale=False,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=5, b=0),
                           yaxis_title="Churn Rate (%)")
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.subheader("Churn Probability Distribution")
        fig3 = go.Figure()
        for label, color, val in [("Retained", "#42A5F5", 0),
                                   ("Churned",  "#EF5350", 1)]:
            fig3.add_trace(go.Histogram(
                x=df[df["Churn"] == val]["Churn_Probability"],
                name=label, opacity=0.65,
                marker_color=color, nbinsx=40
            ))
        fig3.add_vline(x=0.5, line_dash="dash", line_color="gray",
                       annotation_text="Threshold=0.5")
        fig3.update_layout(barmode="overlay", height=300,
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           margin=dict(t=5, b=0),
                           legend=dict(x=0.65, y=0.95),
                           xaxis_title="Churn Probability")
        st.plotly_chart(fig3, use_container_width=True)

    # Row 3
    col4, col5 = st.columns([1, 1.5])

    with col4:
        st.subheader("Contract × Internet Heatmap")
        pivot = (df.groupby(["Contract", "Internet Service"])["Churn"]
                 .mean().unstack().round(3) * 100)
        fig4 = px.imshow(pivot, text_auto=".1f",
                         color_continuous_scale="RdYlGn_r",
                         zmin=0, zmax=80, aspect="auto", height=300)
        fig4.update_layout(margin=dict(t=5),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig4, use_container_width=True)

    with col5:
        st.subheader("Geographic Churn Map — California")
        geo = df.dropna(subset=["Latitude", "Longitude"])
        geo = geo[(geo["Latitude"].between(32, 42)) &
                  (geo["Longitude"].between(-125, -114))]
        fig5 = px.scatter_mapbox(
            geo, lat="Latitude", lon="Longitude",
            color="Churn_Probability",
            size="Monthly Charges",
            color_continuous_scale=["#66BB6A", "#FFA726", "#EF5350"],
            range_color=[0, 1], size_max=10,
            zoom=5, height=300,
            hover_data={"CustomerID": True,
                        "Churn_Probability": ":.3f",
                        "Risk_Tier": True,
                        "Monthly Charges": ":.0f",
                        "Latitude": False,
                        "Longitude": False},
            mapbox_style="open-street-map"
        )
        fig5.update_layout(margin=dict(t=5, b=0),
                           paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)

    # Row 4: Top risk table
    st.subheader("🔴 Top 20 Highest Risk Customers")
    top20 = (df.nlargest(20, "Churn_Probability")
             [["CustomerID", "Contract", "Monthly Charges",
               "Tenure Months", "Internet Service",
               "Churn_Probability", "Risk_Tier", "CLTV"]]
             .reset_index(drop=True))
    top20.index += 1

    def color_prob(val):
        if isinstance(val, float):
            if val >= 0.75: return "background-color:#ffebee;color:#c62828"
            if val >= 0.50: return "background-color:#fff3e0;color:#e65100"
            return "background-color:#e8f5e9;color:#2e7d32"
        return ""

    st.dataframe(
        top20.style
        .applymap(color_prob, subset=["Churn_Probability"])
        .format({"Churn_Probability": "{:.3f}",
                 "Monthly Charges": "${:.2f}",
                 "CLTV": "${:,.0f}"}),
        use_container_width=True, height=400
    )


# ══════════════════════════════════════════
# PAGE 2: STRATEGY SIMULATOR
# ══════════════════════════════════════════
else:
    st.title("🎯 Pricing Strategy Simulator")
    st.caption("Design your retention strategy → Instantly see projected churn & revenue impact")

    # ── Input form ──
    with st.form("strategy_form"):
        st.subheader("⚙️ Design Your Strategy")

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Target Segment**")
            target_contract = st.multiselect(
                "Contract Type",
                ["Month-to-month", "One year", "Two year"],
                default=["Month-to-month"]
            )
            target_risk = st.multiselect(
                "Risk Tier",
                ["High Risk", "Medium Risk", "Low Risk", "Very Low Risk"],
                default=["High Risk", "Medium Risk"]
            )
            target_internet = st.multiselect(
                "Internet Service",
                ["Fiber optic", "DSL", "No"],
                default=["Fiber optic", "DSL"]
            )

        with col_b:
            st.markdown("**Strategy Parameters**")
            discount_pct = st.slider(
                "💰 Price Discount (%)",
                0, 50, 10, 5,
                help="% bill reduction for targeted customers"
            )
            retention_rate = st.slider(
                "🎯 Expected Retention Rate (%)",
                0, 100, 50, 10,
                help="% of at-risk customers retained by this strategy"
            )
            n_sim = st.select_slider(
                "🎲 Simulations",
                options=[200, 500, 1000], value=500
            )
            strategy_desc = st.text_input(
                "📝 Strategy Name",
                placeholder="e.g. 10% loyalty discount for Month-to-month Fiber churners"
            )

        submitted = st.form_submit_button(
            "🚀 Run Simulation", type="primary", use_container_width=True
        )

    if not submitted:
        st.info("👆 Fill in your strategy above and click **Run Simulation**")
        with st.expander("ℹ️ How the simulation works"):
            st.markdown("""
**Monte Carlo Logic (500 runs):**
1. Each customer's churn is randomly sampled from their predicted probability
2. **Discount**: targeted customers get bill reduced by `discount_%`
3. **Retention**: `retention_%` of churned target customers are saved
4. **Revenue** = sum of monthly charges for non-churned customers
5. Repeat 200–1,000 times → confidence intervals (P5/P95)

**Why this matters:**  
A single deterministic estimate hides risk. The P5/P95 range shows your  
worst-case and best-case revenue outcomes under the same strategy.
            """)

    else:
        if not target_contract or not target_risk or not target_internet:
            st.error("Please select at least one option in each filter!")
            st.stop()

        # Build mask
        mask = (
            df["Contract"].isin(target_contract) &
            df["Risk_Tier"].isin(target_risk) &
            df["Internet Service"].isin(target_internet)
        )
        n_targeted = int(mask.sum())

        if n_targeted == 0:
            st.error("No customers match these criteria. Try broadening your filters.")
            st.stop()

        discount  = discount_pct / 100
        retention = retention_rate / 100

        # ── Vectorized Monte Carlo ──
        probs      = df["Churn_Probability"].values
        charges    = df["Monthly Charges"].values.astype(float)
        is_target  = mask.values

        rng = np.random.default_rng(42)

        with st.spinner(f"Running {n_sim} simulations..."):
            # Pre-generate all random samples at once (fast)
            churn_base_all  = rng.binomial(1, np.tile(probs, (n_sim, 1)))
            churn_strat_all = rng.binomial(1, np.tile(probs, (n_sim, 1)))

            base_revs, strat_revs   = [], []
            base_churns, strat_churns = [], []

            disc_charges = charges.copy()
            disc_charges[is_target] *= (1 - discount)

            for i in range(n_sim):
                cb = churn_base_all[i]
                base_revs.append(charges[cb == 0].sum())
                base_churns.append(int(cb.sum()))

                cs = churn_strat_all[i].astype(float)
                churned_targets = np.where(is_target & (cs == 1))[0]
                if len(churned_targets) > 0:
                    n_save = int(len(churned_targets) * retention)
                    if n_save > 0:
                        saved = rng.choice(churned_targets, n_save, replace=False)
                        cs[saved] = 0
                strat_revs.append(disc_charges[cs == 0].sum())
                strat_churns.append(int(cs.sum()))

        base_revs   = np.array(base_revs)
        strat_revs  = np.array(strat_revs)
        base_churns = np.array(base_churns)
        strat_churns = np.array(strat_churns)

        # ── Summary stats ──
        b_rev  = base_revs.mean()
        s_rev  = strat_revs.mean()
        b_ch   = base_churns.mean()
        s_ch   = strat_churns.mean()
        uplift = s_rev - b_rev
        saved  = b_ch - s_ch
        cost   = n_targeted * df[mask]["Monthly Charges"].mean() * discount
        roi    = (uplift / cost * 100) if cost > 0 else float("inf")

        # ── KPIs ──
        st.divider()
        if strategy_desc:
            st.info(f"📝 **{strategy_desc}**")

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Customers Targeted",   f"{n_targeted:,}")
        k2.metric("Baseline Rev/mo",      f"${b_rev:,.0f}")
        k3.metric("Strategy Rev/mo",
                  f"${s_rev:,.0f}",
                  delta=f"${uplift:+,.0f}",
                  delta_color="normal")
        k4.metric("Avg Churn Saved/mo",
                  f"{saved:.0f}",
                  delta=f"↓{saved/b_ch*100:.1f}%",
                  delta_color="normal")
        k5.metric("ROI on Discount Cost", f"{roi:.0f}%")

        st.divider()

        # ── Charts row ──
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("💰 Revenue Distribution")
            fig_r = go.Figure()
            for revs, name, color in [
                (base_revs,  "Baseline",       "#42A5F5"),
                (strat_revs, "With Strategy",  "#66BB6A"),
            ]:
                fig_r.add_trace(go.Violin(
                    y=revs, name=name, fillcolor=color,
                    opacity=0.65, box_visible=True,
                    meanline_visible=True,
                    line_color="gray"
                ))
            fig_r.update_layout(
                height=370, yaxis_title="Monthly Revenue ($)",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10),
                yaxis=dict(tickformat="$,.0f")
            )
            st.plotly_chart(fig_r, use_container_width=True)

        with col2:
            st.subheader("📉 Churn Count Distribution")
            fig_c = go.Figure()
            for churns, name, color in [
                (base_churns,  "Baseline",      "#EF5350"),
                (strat_churns, "With Strategy", "#66BB6A"),
            ]:
                fig_c.add_trace(go.Histogram(
                    x=churns, name=name, opacity=0.65,
                    marker_color=color, nbinsx=30
                ))
            fig_c.update_layout(
                barmode="overlay", height=370,
                xaxis_title="Churned Customers / month",
                yaxis_title="Frequency",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(t=10),
                legend=dict(x=0.65, y=0.95)
            )
            st.plotly_chart(fig_c, use_container_width=True)

        # ── Stats table ──
        st.subheader("📋 Statistical Summary")
        summary = pd.DataFrame({
            "Metric": ["Monthly Revenue ($)", "Monthly Churn Count", "Annual Revenue ($)"],
            "Baseline Mean": [f"${b_rev:,.0f}", f"{b_ch:.0f}", f"${b_rev*12:,.0f}"],
            "Baseline P5–P95": [
                f"${np.percentile(base_revs,5):,.0f} – ${np.percentile(base_revs,95):,.0f}",
                f"{np.percentile(base_churns,5):.0f} – {np.percentile(base_churns,95):.0f}",
                f"${np.percentile(base_revs,5)*12:,.0f} – ${np.percentile(base_revs,95)*12:,.0f}",
            ],
            "Strategy Mean": [f"${s_rev:,.0f}", f"{s_ch:.0f}", f"${s_rev*12:,.0f}"],
            "Strategy P5–P95": [
                f"${np.percentile(strat_revs,5):,.0f} – ${np.percentile(strat_revs,95):,.0f}",
                f"{np.percentile(strat_churns,5):.0f} – {np.percentile(strat_churns,95):.0f}",
                f"${np.percentile(strat_revs,5)*12:,.0f} – ${np.percentile(strat_revs,95)*12:,.0f}",
            ],
            "Δ Change": [
                f"${uplift:+,.0f}/mo",
                f"{saved:+.0f} saved/mo",
                f"${uplift*12:+,.0f}/yr",
            ],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

        # ── Sensitivity ──
        st.subheader("🔍 Sensitivity: Discount Rate vs Revenue Uplift")
        disc_range = [0, 5, 10, 15, 20, 25, 30]
        u_mean, u_p5, u_p95 = [], [], []

        for d in disc_range:
            d_charges = charges.copy()
            d_charges[is_target] *= (1 - d/100)
            d_revs = []
            for i in range(200):
                cs = churn_strat_all[i % n_sim].astype(float)
                churned_t = np.where(is_target & (cs == 1))[0]
                if len(churned_t) > 0:
                    n_save = int(len(churned_t) * retention)
                    if n_save > 0:
                        saved_idx = rng.choice(churned_t, n_save, replace=False)
                        cs[saved_idx] = 0
                d_revs.append(d_charges[cs == 0].sum())
            d_revs = np.array(d_revs)
            uplift_d = d_revs - base_revs[:200]
            u_mean.append(uplift_d.mean())
            u_p5.append(np.percentile(uplift_d, 5))
            u_p95.append(np.percentile(uplift_d, 95))

        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=disc_range + disc_range[::-1],
            y=u_p95 + u_p5[::-1],
            fill="toself", fillcolor="rgba(31,119,180,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="P5–P95"
        ))
        fig_s.add_trace(go.Scatter(
            x=disc_range, y=u_mean,
            mode="lines+markers", name="Mean Uplift",
            line=dict(color="#1f77b4", width=2.5),
            marker=dict(size=8)
        ))
        fig_s.add_hline(y=0, line_dash="dash", line_color="red",
                        annotation_text="Break-even")
        fig_s.update_layout(
            height=330,
            xaxis_title="Discount Rate (%)",
            yaxis_title="Revenue Uplift vs Baseline ($)",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(tickformat="$,.0f"),
            margin=dict(t=10),
            legend=dict(x=0.75, y=0.95)
        )
        st.plotly_chart(fig_s, use_container_width=True)
        st.caption("💡 Stay above the red line for positive ROI. Blue band = uncertainty range.")
