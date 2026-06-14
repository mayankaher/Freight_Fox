"""FreightFox Model v3 — feature prep, training, and live inference."""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder

FESTIVAL_PRESSURE = {
    ('Uttar Pradesh', 10): 1.10, ('Uttar Pradesh', 11): 1.08, ('Uttar Pradesh', 3): 1.06,
    ('Uttar Pradesh', 4): 1.04, ('Maharashtra', 10): 1.06, ('Maharashtra', 11): 1.05,
    ('Punjab', 3): 1.08, ('Punjab', 4): 1.06, ('Haryana', 3): 1.07, ('Haryana', 4): 1.05,
    ('Madhya Pradesh', 3): 1.05, ('Madhya Pradesh', 10): 1.04,
    ('West Bengal', 10): 1.07, ('West Bengal', 9): 1.04,
    ('Kerala', 8): 1.05, ('Kerala', 9): 1.03,
    ('Tamil Nadu', 1): 1.05, ('Tamil Nadu', 9): 1.04,
    ('Gujarat', 10): 1.05, ('Bihar', 10): 1.06, ('Bihar', 11): 1.05,
    ('Rajasthan', 10): 1.04, ('Andhra Pradesh', 9): 1.04,
}

TIER_COMP = {
    '1_1': 1.3, '1_2': 1.1, '1_3': 0.9, '2_1': 1.0, '2_2': 1.0,
    '2_3': 0.85, '3_1': 0.8, '3_2': 0.75, '3_3': 0.70,
}

LGB_PARAMS = dict(
    n_estimators=600, max_depth=8, learning_rate=0.02, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=15,
    reg_alpha=0.1, reg_lambda=0.2, random_state=42, verbose=-1,
)

FEATURES_V3 = [
    'capacity_mt', 'is_container', 'is_twoway', 'is_oneway',
    'is_interstate', 'is_same_city', 'body_ft', 'n_pallets',
    'is_flatbed', 'is_sxl', 'is_mxl', 'is_refer',
    'approx_dist_km', 'log_dist', 'log_capacity',
    'dist_x_capacity', 'log_dist_x_log_cap', 'dist_bucket',
    'orig_tier', 'dest_tier', 'tier_combo_enc', 'tier_competition_idx',
    'orig_odi', 'orig_iac', 'orig_vif', 'orig_sdpi',
    'orig_state_out_rank', 'orig_out_ewb', 'orig_total_ewb',
    'orig_odi_3m', 'orig_sdpi_3m',
    'dest_odi', 'dest_iac', 'dest_vif', 'dest_sdpi',
    'dest_state_out_rank', 'dest_out_ewb', 'dest_total_ewb',
    'dest_odi_3m', 'dest_sdpi_3m',
    'sdpi_x_dist', 'opi_x_capacity', 'odi_x_capacity', 'dest_odi_x_iac',
    'ewb_seasonal_factor', 'festival_pressure', 'combined_seasonal',
    'month_sin', 'month_cos',
    'lane_median', 'lane_count', 'lane_per_km', 'lane_data_density',
    'effective_lane_median', 'effective_rate_per_km',
    'city_lane_median', 'city_lane_count', 'has_city_lane',
    'rfq_bid_count', 'rfq_spread_pct', 'rfq_cv', 'rfq_median', 'rfq_std',
    'tsp_median', 'tsp_mean', 'tsp_bid_count',
    'tsp_lane_diversity', 'tsp_is_large',
    'tsp_vs_lane', 'tsp_specialization', 'tsp_rate_per_km',
    'orig_state_le', 'dest_state_le',
]

MODEL_R2 = 0.928
MODEL_MAPE = 17.8


def _clean_city(c):
    if pd.isna(c):
        return None
    import re
    return re.sub(r'\s*\(.*', '', str(c)).strip().title()


def build_seasonal_lookup(ewb):
    ewb_sl = (
        ewb.groupby(['state_name', 'month'])['seasonal_factor']
        .mean()
        .reset_index()
    )
    ewb_sl['state_name'] = ewb_sl['state_name'].str.strip().str.title()
    return {
        (r['state_name'], int(r['month'])): float(r['seasonal_factor'])
        for _, r in ewb_sl.iterrows()
    }


def prepare_rfq(rfq, ewb):
    """Apply v3 feature engineering to RFQ training data."""
    rfq = rfq.copy()
    rfq['_orig_city'] = rfq['origin_city'].apply(_clean_city)
    rfq['_dest_city'] = rfq['destination_city'].apply(_clean_city)

    seasonal_lookup = build_seasonal_lookup(ewb)

    rfq['festival_pressure'] = rfq.apply(
        lambda r: FESTIVAL_PRESSURE.get((r['origin_state'], int(r['month'])), 1.0), axis=1)

    rfq['origin_state_title'] = rfq['origin_state'].str.strip().str.title()
    sl_df = pd.DataFrame(
        [(s, m, v) for (s, m), v in seasonal_lookup.items()],
        columns=['origin_state_title', 'month', 'ewb_seasonal_factor'],
    )
    rfq = rfq.merge(sl_df, on=['origin_state_title', 'month'], how='left')
    rfq['ewb_seasonal_factor'] = rfq['ewb_seasonal_factor'].fillna(1.0)

    le_orig = LabelEncoder()
    le_dest = LabelEncoder()
    rfq['orig_state_le'] = le_orig.fit_transform(rfq['origin_state'])
    rfq['dest_state_le'] = le_dest.fit_transform(rfq['destination_state'])

    rfq['tsp_vs_lane'] = rfq['tsp_median'] / rfq['lane_median'].replace(0, 1)
    rfq['tsp_specialization'] = rfq['tsp_bid_count'] / rfq['tsp_lane_diversity'].replace(0, 1)
    rfq['effective_rate_per_km'] = (
        rfq['effective_lane_median'] / rfq['approx_dist_km'].replace(0, 1))
    rfq['tsp_rate_per_km'] = rfq['tsp_median'] / rfq['approx_dist_km'].replace(0, 1)
    rfq['tier_competition_idx'] = rfq['tier_combo'].map(TIER_COMP).fillna(1.0)
    rfq['combined_seasonal'] = rfq['ewb_seasonal_factor'] * rfq['festival_pressure']

    features = [f for f in FEATURES_V3 if f in rfq.columns]
    return rfq, seasonal_lookup, le_orig, le_dest, features


def train_v3_models(rfq_prepared, features):
    """Train main + quantile models on all inlier bids (production inference)."""
    q1, q99 = rfq_prepared['quote'].quantile(0.01), rfq_prepared['quote'].quantile(0.99)
    df = rfq_prepared[(rfq_prepared['quote'] >= q1) & (rfq_prepared['quote'] <= q99)].copy()

    X = df[features].fillna(df[features].median())
    y_log = np.log1p(df['quote'])
    y_orig = df['quote']

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X, y_log)

    model_q10 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.10, n_estimators=400, learning_rate=0.03,
        num_leaves=31, random_state=42, verbose=-1)
    model_q90 = lgb.LGBMRegressor(
        objective='quantile', alpha=0.90, n_estimators=400, learning_rate=0.03,
        num_leaves=31, random_state=42, verbose=-1)
    model_q10.fit(X, y_orig)
    model_q90.fit(X, y_orig)

    return model, model_q10, model_q90, X.median().to_dict(), df


def _encode_state(state, le):
    if state in le.classes_:
        return int(le.transform([state])[0])
    return int(np.median(le.transform(le.classes_)))


def season_f(month_num, state, seasonal_lookup):
    key = (state.strip().title(), int(month_num))
    val = seasonal_lookup.get(key)
    if val is None:
        base = {1: 1.02, 2: 1.00, 3: 1.05, 4: 0.91, 5: 0.94, 6: 0.95,
                7: 0.98, 8: 1.03, 9: 1.08, 10: 1.04, 11: 1.00, 12: 1.02}
        val = base.get(int(month_num), 1.0)
    return round(float(val), 4)


def build_predict_row(
    origin_city, origin_state, dest_city, dest_state,
    vtype, month_num, competition, dist,
    ewb_orig, ewb_dest, rfq, feat_med,
    seasonal_lookup, le_orig, le_dest,
):
    """Build a single feature row for live inference."""
    cap = float(re_extract_cap(vtype))
    is_c = int('container' in vtype.lower())
    is_i = int(origin_state != dest_state)
    same_city = origin_city == dest_city
    db = 0 if dist < 250 else (1 if dist < 750 else 2)

    city_lane = rfq[(rfq['_orig_city'] == origin_city) & (rfq['_dest_city'] == dest_city)]
    state_lane = rfq[
        (rfq['origin_state'] == origin_state) & (rfq['destination_state'] == dest_state)
    ]
    lm = city_lane['quote'].median()
    if np.isnan(lm):
        lm = state_lane['quote'].median()
    lm = lm if not np.isnan(lm) else rfq['quote'].median()

    lane_count = len(state_lane) if len(state_lane) else feat_med.get('lane_count', 5)
    city_count = len(city_lane) if len(city_lane) else feat_med.get('city_lane_count', 3)
    has_city = int(len(city_lane) > 0)

    sf = season_f(month_num, origin_state, seasonal_lookup)
    fp = FESTIVAL_PRESSURE.get((origin_state.strip().title(), int(month_num)), 1.0)
    tsp_med = rfq['quote'].median()
    tsp_mean = rfq['quote'].mean()

    orig_tier = feat_med.get('orig_tier', 2)
    dest_tier = feat_med.get('dest_tier', 2)
    tier_combo = f"{int(orig_tier)}_{int(dest_tier)}"

    month_sin = np.sin(2 * np.pi * month_num / 12)
    month_cos = np.cos(2 * np.pi * month_num / 12)

    vtype_l = vtype.lower()
    row = {
        'capacity_mt': cap, 'is_container': is_c,
        'is_twoway': feat_med.get('is_twoway', 0),
        'is_oneway': feat_med.get('is_oneway', 1),
        'is_interstate': is_i, 'is_same_city': int(same_city),
        'body_ft': feat_med.get('body_ft', 32),
        'n_pallets': feat_med.get('n_pallets', 0),
        'is_flatbed': int('flat' in vtype_l or 'bed' in vtype_l),
        'is_sxl': int('sxl' in vtype_l or 'one way' in vtype_l),
        'is_mxl': int('mxl' in vtype_l or 'two way' in vtype_l),
        'is_refer': int('refer' in vtype_l),
        'approx_dist_km': dist, 'log_dist': np.log1p(dist),
        'log_capacity': np.log1p(cap), 'dist_x_capacity': dist * cap,
        'log_dist_x_log_cap': np.log1p(dist) * np.log1p(cap),
        'dist_bucket': db,
        'orig_tier': orig_tier, 'dest_tier': dest_tier,
        'tier_combo_enc': feat_med.get('tier_combo_enc', 0),
        'tier_competition_idx': TIER_COMP.get(tier_combo, 1.0),
        'orig_odi': ewb_orig['odi'], 'orig_iac': ewb_orig['iac'],
        'orig_vif': ewb_orig['vif'], 'orig_sdpi': ewb_orig['sdpi'],
        'orig_state_out_rank': ewb_orig['state_out_rank'],
        'orig_out_ewb': ewb_orig['out_ewb'], 'orig_total_ewb': ewb_orig['total_ewb'],
        'orig_odi_3m': ewb_orig['odi_3m_avg'], 'orig_sdpi_3m': ewb_orig['sdpi_3m_avg'],
        'dest_odi': ewb_dest['odi'], 'dest_iac': ewb_dest['iac'],
        'dest_vif': ewb_dest.get('vif', feat_med.get('dest_vif', 1.0)),
        'dest_sdpi': ewb_dest['sdpi'],
        'dest_state_out_rank': ewb_dest.get('state_out_rank', feat_med.get('dest_state_out_rank', 10)),
        'dest_out_ewb': ewb_dest['out_ewb'], 'dest_total_ewb': ewb_dest['total_ewb'],
        'dest_odi_3m': ewb_dest['odi_3m_avg'], 'dest_sdpi_3m': ewb_dest['sdpi_3m_avg'],
        'sdpi_x_dist': ewb_orig['sdpi'] * dist,
        'opi_x_capacity': ewb_orig['opi'] * cap,
        'odi_x_capacity': ewb_orig['odi'] * cap,
        'dest_odi_x_iac': ewb_dest['odi'] * ewb_dest['iac'],
        'ewb_seasonal_factor': sf, 'festival_pressure': fp,
        'combined_seasonal': sf * fp,
        'month_sin': month_sin, 'month_cos': month_cos,
        'lane_median': lm, 'lane_count': lane_count,
        'lane_per_km': lm / max(dist, 1),
        'lane_data_density': feat_med.get('lane_data_density', 0.5),
        'effective_lane_median': lm,
        'effective_rate_per_km': lm / max(dist, 1),
        'city_lane_median': lm, 'city_lane_count': city_count,
        'has_city_lane': has_city,
        'rfq_bid_count': competition,
        'rfq_spread_pct': feat_med.get('rfq_spread_pct', 1.5),
        'rfq_cv': feat_med.get('rfq_cv', 0.5),
        'rfq_median': lm, 'rfq_std': feat_med.get('rfq_std', 5000),
        'tsp_median': tsp_med, 'tsp_mean': tsp_mean,
        'tsp_bid_count': feat_med.get('tsp_bid_count', 20),
        'tsp_lane_diversity': feat_med.get('tsp_lane_diversity', 10),
        'tsp_is_large': feat_med.get('tsp_is_large', 0),
        'tsp_vs_lane': tsp_med / max(lm, 1),
        'tsp_specialization': feat_med.get('tsp_specialization', 5.0),
        'tsp_rate_per_km': tsp_med / max(dist, 1),
        'orig_state_le': _encode_state(origin_state, le_orig),
        'dest_state_le': _encode_state(dest_state, le_dest),
    }
    return row


def re_extract_cap(v):
    import re
    m = re.search(r'(\d+(?:\.\d+)?)\s*MT', str(v), re.I)
    return float(m.group(1)) if m else 16.0


def predict_v3(model, model_q10, model_q90, features, feat_med, row):
    X = pd.DataFrame([row])[features].fillna(pd.Series(feat_med))
    price = float(np.expm1(model.predict(X)[0]))
    ci_lo = float(model_q10.predict(X)[0])
    ci_hi = float(model_q90.predict(X)[0])
    ci_lo = max(ci_lo, price * 0.5)
    ci_hi = max(ci_hi, ci_lo)
    return price, ci_lo, ci_hi
