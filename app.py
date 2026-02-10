import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

st.set_page_config(page_title="Etihad Cargo ML Predictor", layout="wide")

# LOAD DATA
@st.cache_resource
def load_data():
    return joblib.load('final_results.pkl')

try:
    res_df = load_data()
except:
    st.error("Error: 'final_results.pkl' not found. Run the trainer script first!")
    st.stop()

st.title("✈️ Etihad Cargo: Dual-Model ML Predictor")

# SIDEBAR
st.sidebar.header("Model Parameters")
min_date, max_date = res_df['DEPARTURE_DATE'].min(), res_df['DEPARTURE_DATE'].max()

test_range = st.sidebar.date_input(
    "Evaluation Period", 
    [min_date, min_date + timedelta(days=14)],
    min_value=min_date, max_value=max_date
)

# FILTERING (Dynamic for ALL routes)
col1, col2 = st.columns(2)
origins = sorted(res_df['LEG_ORIGIN'].unique())
sel_orig = col1.selectbox("Filter Origin", origins, index=origins.index("BCN") if "BCN" in origins else 0)

available_dests = sorted(res_df[res_df['LEG_ORIGIN'] == sel_orig]['LEG_DESTINATION'].unique())
sel_dest = col2.selectbox("Filter Destination", available_dests, index=available_dests.index("AUH") if "AUH" in available_dests else 0)

# APPLY FILTERS
view_df = res_df[(res_df['LEG_ORIGIN'] == sel_orig) & 
                 (res_df['LEG_DESTINATION'] == sel_dest) &
                 (res_df['DEPARTURE_DATE'] >= pd.to_datetime(test_range[0])) &
                 (res_df['DEPARTURE_DATE'] <= pd.to_datetime(test_range[1]))].copy()

if not view_df.empty:
    st.header(f"Route Performance: {sel_orig} ➔ {sel_dest}")
    
    # 1. METRICS
    mae_fj = mean_absolute_error(view_df['actual_ake_fj'], view_df['pred_fj'])
    mae_y = mean_absolute_error(view_df['actual_ake_y'], view_df['pred_y'])
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Premium MAE", f"{mae_fj:.2f}")
    c2.metric("Economy MAE", f"{mae_y:.2f}")
    c3.metric("Hierarchy Level", view_df['hierarchy_level'].iloc[0])

    st.divider()

    # 2. TABLE FIRST (Per your request)
    st.subheader("Baggage Load (Actual / Predicted)")
    
    # Create the display string
    view_df['Display'] = (view_df['actual_total'].astype(int).astype(str) + 
                          " / " + 
                          view_df['final_pred_total'].astype(int).astype(str))
    
    # Create the pivot
    pivot = view_df.pivot(index='FLTNO', columns='DEPARTURE_DATE', values='Display').fillna("-")
    
    # FIX DATE FORMATTING: Change "2023-05-01 00:00:00" to "May-01"
    pivot.columns = [d.strftime('%b-%d') for d in pivot.columns]
    
    st.caption("Legend: Actual / Predicted")
    st.dataframe(pivot, use_container_width=True)

    st.divider()

    # 3. CHART BELOW
    st.subheader("Trends Over Time")
    # Clean up dates for the chart axis as well
    chart_data = view_df.copy()
    chart_data['Date'] = chart_data['DEPARTURE_DATE'].dt.strftime('%b-%d')
    st.line_chart(chart_data.set_index('Date')[['actual_total', 'final_pred_total']])

else:
    st.warning("No data found for this selection.")
    