import streamlit as st
import pandas as pd
import json
import altair as alt

# --- CONFIGURATION ---
SUMMARY_FILE = "run_summary.json" 
ASSIGNMENTS_FILE = "assignments.csv" 
COMPARE_FILE = "compare_summary.json"

# --- FUNCTIONS ---
@st.cache_data 
def load_summary_data():
    """Load the JSON summary with overall metrics."""
    try:
        with open(SUMMARY_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: {SUMMARY_FILE} not found. Run main_sa.py first.")
        return None

@st.cache_data
def load_assignments_data():
    """Load the detailed assignments CSV."""
    try:
        return pd.read_csv(ASSIGNMENTS_FILE, parse_dates=['start_time', 'finish_time'])
    except FileNotFoundError:
        return None

@st.cache_data
def load_compare_data():
    """Load the comparison JSON for the optimization proof."""
    try:
        with open(COMPARE_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        # It's okay if this is missing, we just won't show the graph
        return None

# --- STREAMLIT APP ---
st.title("🤖 AI Helpdesk Scheduling Results")

summary = load_summary_data()
assignments_df = load_assignments_data()
compare_data = load_compare_data()

if summary is not None and assignments_df is not None:
    
    # --- HEADER ---
    st.header(f"Scheduler: {summary['algorithm']['scheduler'].upper()}")
    st.caption(f"Tickets Processed: {summary['datasets']['n_tickets']} | Triage Method: {summary['algorithm']['triage']['method']}")
    
    # --- 1. KEY METRICS ---
    col1, col2, col3 = st.columns(3)
    metrics = summary['metrics_overall']
    col1.metric("SLA Breach Rate", f"{metrics['sla_breach_rate']:.2%}")
    col2.metric("Workload Variance", f"{metrics['workload_variance']:.2f}")
    col3.metric("Total Tardiness", f"{metrics['total_tardiness_minutes']} min")
    
    st.divider()

    # --- 2. OPTIMIZATION PROOF: COMPARE CHART ---
    st.subheader("Optimization Proof: Algo Comparison")
    
    if compare_data:
        st.caption("Comparing Random vs. Greedy vs. Simulated Annealing (Greedy+SA)")
        
        # Parse the nested JSON structure into a flat list for plotting
        results = compare_data['results']
        rows = []
        
        # 1. Random (Mean)
        rows.append({
            "Algorithm": "Random (Mean)", 
            "Metric": "SLA Breach Rate (%)", 
            "Value": results['random_mean']['sla_breach_rate'] * 100
        })
        rows.append({
            "Algorithm": "Random (Mean)", 
            "Metric": "Workload Variance", 
            "Value": results['random_mean']['workload_variance']
        })
        
        # 2. Greedy
        rows.append({
            "Algorithm": "Greedy", 
            "Metric": "SLA Breach Rate (%)", 
            "Value": results['greedy']['sla_breach_rate'] * 100
        })
        rows.append({
            "Algorithm": "Greedy", 
            "Metric": "Workload Variance", 
            "Value": results['greedy']['workload_variance']
        })
        
        # 3. Greedy+SA
        rows.append({
            "Algorithm": "Greedy+SA", 
            "Metric": "SLA Breach Rate (%)", 
            "Value": results['greedy+sa']['sla_breach_rate'] * 100
        })
        rows.append({
            "Algorithm": "Greedy+SA", 
            "Metric": "Workload Variance", 
            "Value": results['greedy+sa']['workload_variance']
        })
        
        chart_df = pd.DataFrame(rows)
        
        # Create a grouped bar chart using Altair
        c = alt.Chart(chart_df).mark_bar().encode(
            x=alt.X('Algorithm', axis=None),
            y=alt.Y('Value', title='Score'),
            color='Algorithm',
            column=alt.Column('Metric', title=None),
            tooltip=['Algorithm', 'Metric', 'Value']
        ).properties(width=250)
        
        st.altair_chart(c)
        
    else:
        st.info("Run `python run_compare.py ...` to see the comparison chart here.")

    st.divider()

    # --- 3. TRIAGE PROOF: CATEGORY BREAKDOWN ---
    st.subheader("Triage Proof: Category Breakdown")
    
    col_chart, col_text = st.columns([2, 1])
    
    with col_chart:
        # Count tickets per category
        cat_counts = assignments_df['predicted_category'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        
        # Create a Donut Chart
        base = alt.Chart(cat_counts).encode(theta=alt.Theta("Count", stack=True))
        pie = base.mark_arc(outerRadius=120, innerRadius=80).encode(
            color=alt.Color("Category"),
            order=alt.Order("Count", sort="descending"),
            tooltip=["Category", "Count"]
        )
        text = base.mark_text(radius=140).encode(
            text=alt.Text("Count"),
            order=alt.Order("Count", sort="descending"),
            color=alt.value("white")  
        )
        st.altair_chart(pie + text, use_container_width=True)

    with col_text:
        st.write("Distribution of predicted categories.")
        st.dataframe(cat_counts, hide_index=True)

    st.divider()

    # --- 4. HELPER WORKLOAD ---
    st.subheader("Helper Workload Distribution")
    helper_load_data = pd.DataFrame(summary['metrics_by_helper'])
    helper_load_data = helper_load_data.set_index('helper_id')
    st.bar_chart(helper_load_data, y='tickets')

    st.divider()
    
    # --- 5. FULL TABLE ---
    st.subheader("Full Assignment Schedule")
    display_df = assignments_df.drop(columns=['customer_id', 'text', 'sla_hours', 'created_at'], errors='ignore')
    if st.checkbox('Show raw data table'):
        st.dataframe(display_df, use_container_width=True)