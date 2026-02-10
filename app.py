import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# --- 1. APP CONFIGURATION ---
st.set_page_config(page_title="Etihad Cargo ML Predictor", layout="wide")
st.title("✈️ Etihad Cargo: Dual-Model XGBoost Predictor")

# --- 2. DATA LOADING & ML ENGINE ---
@st.cache_data
def run_ml_pipeline():
    # Loading the specific ML dataset
    df = pd.read_csv('etihad_ml_data.csv')
    df['DEPARTURE_DATE'] = pd.to_datetime(df['DEPARTURE_DATE'])
    
    # 1. PRE-PROCESSING
    df['route'] = (df['LEG_ORIGIN'] + '-' + df['LEG_DESTINATION']).str.upper()
    df['country_pair'] = (df['Orig Coun'] + '-' + df['Dest Coun']).str.upper()
    df['month'] = df['DEPARTURE_DATE'].dt.month
    df['dow'] = df['DEPARTURE_DATE'].dt.dayofweek
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Targets
    df['actual_ake_fj'] = df['Bag Count L3E Actl F/B'].fillna(0)
    df['actual_ake_y']  = df['Bag Count L3E Actl Y'].fillna(0)
    if 'MAX_LD3' not in df.columns: df['MAX_LD3'] = 30

    # 2. SPLIT (Training Knowledge < May)
    train = df[df['DEPARTURE_DATE'] < '2023-05-01'].copy()
    test_full = df[df['DEPARTURE_DATE'] >= '2023-05-01'].copy()

    # 3. FEATURE ENGINEERING
    train['eff_fj'] = train['booked_fj_d1'] / train['actual_ake_fj'].replace(0, np.nan)
    train['eff_y']  = train['booked_y_d1'] / train['actual_ake_y'].replace(0, np.nan)
    train['dens_fj'] = train['Bag Wgt Actl F/B'] / train['booked_fj_d1'].replace(0, np.nan)
    train['dens_y']  = train['Bag Wgt Actl Y'] / train['booked_y_d1'].replace(0, np.nan)

    # 4. HIERARCHY AGGREGATION
    def get_lvl(df_in, group_cols, prefix):
        res = df_in.groupby(group_cols).agg({
            'actual_ake_fj': 'mean', 'actual_ake_y': 'mean',
            'eff_fj': 'median', 'eff_y': 'median',
            'dens_fj': 'median', 'dens_y': 'median'
        }).reset_index()
        res.columns = group_cols + [f'{prefix}_avg_fj', f'{prefix}_avg_y', f'{prefix}_eff_fj', f'{prefix}_eff_y', f'{prefix}_dens_fj', f'{prefix}_dens_y']
        return res

    lvl_a = get_lvl(train, ['route', 'Equip Fcst', 'dow'], 'A')
    lvl_b = get_lvl(train, ['country_pair', 'Equip Fcst', 'dow'], 'B')
    lvl_c = get_lvl(train, ['Equip Fcst'], 'C')

    # 5. FALLBACKS
    def apply_hierarchy(input_df):
        df_out = input_df.merge(lvl_a, on=['route', 'Equip Fcst', 'dow'], how='left')
        df_out = df_out.merge(lvl_b, on=['country_pair', 'Equip Fcst', 'dow'], how='left')
        df_out = df_out.merge(lvl_c, on=['Equip Fcst'], how='left')
        
        for cat in ['fj', 'y']:
            for feat in ['avg', 'eff', 'dens']:
                df_out[f'hist_{feat}_{cat}'] = df_out[f'A_{feat}_{cat}'].fillna(df_out[f'B_{feat}_{cat}']).fillna(df_out[f'C_{feat}_{cat}']).fillna(0)
        
        df_out['hierarchy_level'] = np.where(df_out['A_avg_fj'].notna(), 'A', np.where(df_out['B_avg_fj'].notna(), 'B', 'C'))
        return df_out

    train = apply_hierarchy(train)
    test_processed = apply_hierarchy(test_full)

    # 6. TRAINING
    feat_fj = ['booked_fj_d1', 'pax_fcst_fj_d1', 'mom_booked_fj', 'hist_avg_fj', 'hist_eff_fj', 'hist_dens_fj', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'MAX_LD3']
    feat_y = ['booked_y_d1', 'pax_fcst_y_d1', 'mom_booked_y', 'hist_avg_y', 'hist_eff_y', 'hist_dens_y', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'MAX_LD3']
    
    params = {'n_estimators': 1200, 'learning_rate': 0.01, 'max_depth': 8, 'random_state': 42}
    
    m_fj = XGBRegressor(**params).fit(train[feat_fj], train['actual_ake_fj'])
    m_y = XGBRegressor(**params).fit(train[feat_y], train['actual_ake_y'])
    
    test_processed['pred_fj'] = m_fj.predict(test_processed[feat_fj])
    test_processed['pred_y'] = m_y.predict(test_processed[feat_y])
    test_processed['final_pred_total'] = (test_processed['pred_fj'] + test_processed['pred_y']).round()
    test_processed['actual_total'] = test_processed['actual_ake_fj'] + test_processed['actual_ake_y']
    
    return test_processed

# --- 3. SIDEBAR CONTROLS ---
st.sidebar.header("Model Parameters")

# Restricted Calendar: May 1 2023 to July 31 2023
min_date = pd.to_datetime('2023-05-01')
max_date = pd.to_datetime('2023-07-31')

test_range = st.sidebar.date_input(
    "Evaluation Period", 
    [min_date, min_date + timedelta(days=14)],
    min_value=min_date,
    max_value=max_date
)

run_forecast = st.sidebar.button("Run Forecast Model")

# --- 4. DATA PROCESSING ON BUTTON CLICK ---
if run_forecast:
    with st.spinner("Processing ML Forecast..."):
        st.session_state['res_df'] = run_ml_pipeline()
    st.success("Analysis Complete!")

# --- 5. UI & VISUALIZATION ---
if 'res_df' in st.session_state:
    res_df = st.session_state['res_df']
    
    st.divider()
    col1, col2 = st.columns(2)
    
    # Logic to default to AUH
    origins = sorted(res_df['LEG_ORIGIN'].unique())
    default_orig_idx = origins.index("BCN") if "BCN" in origins else 0
    sel_orig = col1.selectbox("Filter Origin", origins, index=default_orig_idx)
    
    # Logic to default to LHR
    available_dests = sorted(res_df[res_df['LEG_ORIGIN'] == sel_orig]['LEG_DESTINATION'].unique())
    default_dest_idx = available_dests.index("AUH") if "AUH" in available_dests else 0
    sel_dest = col2.selectbox("Filter Destination", available_dests, index=default_dest_idx)

    # Filter View Data
    view_df = res_df[(res_df['LEG_ORIGIN'] == sel_orig) & 
                     (res_df['LEG_DESTINATION'] == sel_dest) &
                     (res_df['DEPARTURE_DATE'] >= pd.to_datetime(test_range[0])) &
                     (res_df['DEPARTURE_DATE'] <= pd.to_datetime(test_range[1]))].copy()

    if not view_df.empty:
        st.header(f"Route Performance: {sel_orig} ➔ {sel_dest}")
        
        # Cabin-Wise MAE for the specific route
        mae_route_fj = mean_absolute_error(view_df['actual_ake_fj'], view_df['pred_fj'])
        mae_route_y = mean_absolute_error(view_df['actual_ake_y'], view_df['pred_y'])
        mse_route_total = mean_squared_error(view_df['actual_total'], view_df['final_pred_total'])
        h_level = view_df['hierarchy_level'].iloc[0]

        # Metric Display
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("F/J (Premium) MAE", f"{mae_route_fj:.2f}")
        c2.metric("Y (Economy) MAE", f"{mae_route_y:.2f}")
        c3.metric("Route MSE", f"{mse_route_total:.2f}")
        c4.metric("Hierarchy", h_level)

        st.divider()
        
        # Pivot Table Display
        st.subheader("Baggage Load (Actual / Predicted)")
        view_df['Display'] = (view_df['actual_total'].astype(int).astype(str) + 
                              " / " + view_df['final_pred_total'].astype(int).astype(str))
        
        pivot = view_df.pivot(index='FLTNO', columns='DEPARTURE_DATE', values='Display').fillna("-")
        pivot.columns = [d.strftime('%b-%d') for d in pivot.columns]
        
        st.caption("Legend: Actual / Predicted")
        st.dataframe(pivot, use_container_width=True)
        
        # Comparison Chart
        st.line_chart(view_df.set_index('DEPARTURE_DATE')[['actual_total', 'final_pred_total']])
    else:
        st.warning(f"No flight data found for {sel_orig}-{sel_dest} in the selected dates.")
else:
    st.info("Please click 'Run Forecast Model' in the sidebar to begin.")