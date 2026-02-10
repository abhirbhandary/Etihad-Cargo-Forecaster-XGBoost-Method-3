import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import joblib

def run_training_pipeline():
    print("Loading data and starting training...")
    df = pd.read_csv('etihad_ml_data.csv')
    df['DEPARTURE_DATE'] = pd.to_datetime(df['DEPARTURE_DATE'])
    
    # 1. PRE-PROCESSING
    df['route'] = (df['LEG_ORIGIN'] + '-' + df['LEG_DESTINATION']).str.upper()
    df['country_pair'] = (df['Orig Coun'] + '-' + df['Dest Coun']).str.upper()
    df['month'] = df['DEPARTURE_DATE'].dt.month
    df['dow'] = df['DEPARTURE_DATE'].dt.dayofweek
    
    # Cyclic encoding for time features
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    df['actual_ake_fj'] = df['Bag Count L3E Actl F/B'].fillna(0)
    df['actual_ake_y']  = df['Bag Count L3E Actl Y'].fillna(0)
    if 'MAX_LD3' not in df.columns: df['MAX_LD3'] = 30

    # 2. SPLIT
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

    # 5. TRAINING
    feat_fj = ['booked_fj_d1', 'pax_fcst_fj_d1', 'mom_booked_fj', 'hist_avg_fj', 'hist_eff_fj', 'hist_dens_fj', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'MAX_LD3']
    feat_y = ['booked_y_d1', 'pax_fcst_y_d1', 'mom_booked_y', 'hist_avg_y', 'hist_eff_y', 'hist_dens_y', 'month_sin', 'month_cos', 'dow_sin', 'dow_cos', 'MAX_LD3']
    
    params = {'n_estimators': 1200, 'learning_rate': 0.01, 'max_depth': 8, 'random_state': 42}
    
    m_fj = XGBRegressor(**params).fit(train[feat_fj], train['actual_ake_fj'])
    m_y = XGBRegressor(**params).fit(train[feat_y], train['actual_ake_y'])
    
    # 6. GENERATE FINAL DF
    test_processed['pred_fj'] = m_fj.predict(test_processed[feat_fj])
    test_processed['pred_y'] = m_y.predict(test_processed[feat_y])
    test_processed['final_pred_total'] = (test_processed['pred_fj'] + test_processed['pred_y']).round()
    test_processed['actual_total'] = test_processed['actual_ake_fj'] + test_processed['actual_ake_y']
    
    # 7. EXPORT
    joblib.dump(test_processed, 'final_results.pkl')
    print("Model trained and data saved to final_results.pkl")

if __name__ == "__main__":
    run_training_pipeline()