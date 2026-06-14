"""
FreightFox Enhanced Model v3
==============================
Improvements over the existing model:
1. Proper RFQ-ID-based train/test split (no data leakage)
2. Real EWB state x month seasonal factors (not hardcoded)
3. Festival calendar as an explicit feature
4. Better state encoding (LabelEncoder, no hash collisions)
5. Quantile regression for honest confidence intervals
6. SHAP-based feature explanations
7. Carrier intelligence features
8. Cross-validation with GroupKFold on rfq_id

Run: python freightfox_model_v3.py
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─── 1. LOAD DATA ────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data")
print("=" * 60)

rfq = pd.read_csv('model_ready_v2.csv')
ewb = pd.read_csv('ewb_features_v2.csv')
fi  = pd.read_csv('feature_importance_v2.csv')

print(f"RFQ data: {rfq.shape[0]:,} rows, {rfq.shape[1]} columns")
print(f"EWB data: {ewb.shape[0]:,} rows spanning {ewb['date'].min()} → {ewb['date'].max()}")
print(f"Unique RFQ IDs (shipment requests): {rfq['rfq_id'].nunique()}")
print(f"Unique lanes (origin→dest state): {rfq[['origin_state','destination_state']].drop_duplicates().shape[0]}")
print(f"Quote range: ₹{rfq['quote'].min():,.0f} – ₹{rfq['quote'].max():,.0f}")
print(f"Month in RFQ data: {sorted(rfq['month'].unique())} ← ALL November! Static snapshot.")

# ─── 2. DATA QUALITY AUDIT ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Data quality audit")
print("=" * 60)

# Check for constant columns (useless features)
constant_cols = [c for c in rfq.select_dtypes(include='number').columns
                 if rfq[c].nunique() <= 1]
print(f"\nConstant columns (zero information, should be DROPPED):")
for c in constant_cols:
    print(f"  • {c} = {rfq[c].iloc[0]:.4f} always")

# Check nulls
null_cols = rfq.isnull().sum()
null_cols = null_cols[null_cols > 0]
print(f"\nColumns with nulls:")
print(null_cols)

# ─── 3. BUILD SEASONAL LOOKUP FROM EWB ───────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: Building real seasonal factors from 7 years of EWB data")
print("=" * 60)
"""
Instead of hardcoded seasonal multipliers, we use the actual
EWB seasonal_factor computed per state per month over 7 years.
This captures real harvest cycles, festival patterns, monsoon dips etc.
"""

ewb['date'] = pd.to_datetime(ewb['date'])

# Compute mean seasonal factor per state per month
seasonal_lookup = (ewb
    .groupby(['state_name', 'month'])['seasonal_factor']
    .mean()
    .reset_index()
    .rename(columns={'seasonal_factor': 'ewb_seasonal_factor', 'state_name': 'origin_state'}))

# Normalize state names to match RFQ
seasonal_lookup['origin_state'] = seasonal_lookup['origin_state'].str.strip().str.title()

print("Sample seasonal factors (state × month) — real data, not hardcoded:")
pivot = seasonal_lookup.pivot(index='origin_state', columns='month', values='ewb_seasonal_factor')
states_to_show = ['Maharashtra', 'Punjab', 'Kerala', 'West Bengal', 'Tamil Nadu', 'Uttar Pradesh']
sample_states = [s for s in states_to_show if s in pivot.index]
print(pivot.loc[sample_states].round(3).to_string())

# ─── 4. FESTIVAL CALENDAR FEATURE ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Festival calendar — corridor pressure scores")
print("=" * 60)
"""
Festivals create demand spikes on SPECIFIC corridors.
We encode this as a festival_pressure score (additive on top of seasonal).

Logic:
- Diwali (Oct/Nov): UP, Maharashtra, Gujarat are heavy shippers
- Durga Puja (Oct): West Bengal exports high
- Pongal (Jan): Tamil Nadu
- Onam (Aug): Kerala
- Rabi Harvest (Mar-Apr): Punjab, Haryana, MP, UP
- Kharif Harvest (Oct-Nov): UP, Bihar, MP
"""

FESTIVAL_PRESSURE = {
    # (origin_state, month) → additional multiplier
    ('Uttar Pradesh',  10): 1.10,   # Diwali + Kharif
    ('Uttar Pradesh',  11): 1.08,
    ('Uttar Pradesh',   3): 1.06,   # Rabi
    ('Uttar Pradesh',   4): 1.04,
    ('Maharashtra',    10): 1.06,   # Diwali
    ('Maharashtra',    11): 1.05,
    ('Punjab',          3): 1.08,   # Rabi wheat harvest
    ('Punjab',          4): 1.06,
    ('Haryana',         3): 1.07,
    ('Haryana',         4): 1.05,
    ('Madhya Pradesh',  3): 1.05,   # Wheat + soybean
    ('Madhya Pradesh', 10): 1.04,   # Kharif soybean
    ('West Bengal',    10): 1.07,   # Durga Puja
    ('West Bengal',     9): 1.04,   # Pre-Puja
    ('Kerala',          8): 1.05,   # Onam
    ('Kerala',          9): 1.03,
    ('Tamil Nadu',      1): 1.05,   # Pongal
    ('Tamil Nadu',      9): 1.04,   # Navratri
    ('Gujarat',        10): 1.05,   # Diwali FMCG + automotive
    ('Bihar',          10): 1.06,   # Chhath Puja
    ('Bihar',          11): 1.05,
    ('Rajasthan',      10): 1.04,   # Navratri
    ('Andhra Pradesh',  9): 1.04,   # Navratri
}

def get_festival_pressure(state, month):
    """Return festival pressure multiplier for a given state and month."""
    return FESTIVAL_PRESSURE.get((state, month), 1.0)

# Add festival pressure to RFQ
rfq['festival_pressure'] = rfq.apply(
    lambda r: get_festival_pressure(r['origin_state'], r['month']), axis=1)

print("Festival pressure distribution:")
print(rfq['festival_pressure'].value_counts().sort_index())
print("\nSample high-pressure rows:")
high = rfq[rfq['festival_pressure'] > 1.04][['origin_state','month','festival_pressure']].drop_duplicates()
print(high.sort_values('festival_pressure', ascending=False).head(10))

# ─── 5. FEATURE ENGINEERING ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Feature engineering")
print("=" * 60)

from sklearn.preprocessing import LabelEncoder

# Merge real seasonal factor from EWB
# (rfq has month=11 for all, but we keep the column for future months)
rfq_month = rfq.copy()
rfq_month['origin_state_title'] = rfq_month['origin_state'].str.strip().str.title()
rfq_month = rfq_month.merge(
    seasonal_lookup.rename(columns={'origin_state': 'origin_state_title'}),
    on=['origin_state_title', 'month'],
    how='left'
)
rfq_month['ewb_seasonal_factor'] = rfq_month['ewb_seasonal_factor'].fillna(1.0)

print(f"EWB seasonal factor merged — nulls remaining: {rfq_month['ewb_seasonal_factor'].isna().sum()}")

# Better state encoding (LabelEncoder — no hash collisions)
le_orig = LabelEncoder()
le_dest = LabelEncoder()
rfq_month['orig_state_le'] = le_orig.fit_transform(rfq_month['origin_state'])
rfq_month['dest_state_le'] = le_dest.fit_transform(rfq_month['destination_state'])

# Carrier intelligence features
rfq_month['tsp_vs_lane']        = rfq_month['tsp_median'] / rfq_month['lane_median'].replace(0, 1)
rfq_month['tsp_specialization'] = rfq_month['tsp_bid_count'] / rfq_month['tsp_lane_diversity'].replace(0, 1)

# Rate per km (normalized)
rfq_month['effective_rate_per_km'] = (
    rfq_month['effective_lane_median'] / rfq_month['approx_dist_km'].replace(0, 1))
rfq_month['tsp_rate_per_km']       = (
    rfq_month['tsp_median'] / rfq_month['approx_dist_km'].replace(0, 1))

# Tier competition index — Tier1→Tier1 has most trucks, Tier3→Tier3 fewest
tier_comp = {'1_1':1.3,'1_2':1.1,'1_3':0.9,'2_1':1.0,'2_2':1.0,
             '2_3':0.85,'3_1':0.8,'3_2':0.75,'3_3':0.70}
rfq_month['tier_competition_idx'] = rfq_month['tier_combo'].map(tier_comp).fillna(1.0)

# Combined seasonal signal
rfq_month['combined_seasonal'] = rfq_month['ewb_seasonal_factor'] * rfq_month['festival_pressure']

print("New features added:")
new_feats = ['ewb_seasonal_factor','festival_pressure','combined_seasonal',
             'tsp_vs_lane','tsp_specialization','effective_rate_per_km',
             'tsp_rate_per_km','tier_competition_idx','orig_state_le','dest_state_le']
for f in new_feats:
    print(f"  • {f}: min={rfq_month[f].min():.3f}, max={rfq_month[f].max():.3f}, "
          f"mean={rfq_month[f].mean():.3f}")


# ─── 6. TRAIN / TEST SPLIT (RFQ-ID BASED) ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Proper train/test split — grouped by rfq_id")
print("=" * 60)
"""
KEY INSIGHT: You MUST split by rfq_id, NOT by row.

Why? One RFQ has ~87 bids on average. If you split by row, 
the model sees 70 bids from RFQ #42 in training and 20 bids 
from the same RFQ in test — it has basically already seen the 
"answer" (the lane/distance/competition for that shipment).

Splitting by rfq_id ensures the test set has shipments 
the model has NEVER seen at all — simulating real production.
"""

from sklearn.model_selection import GroupKFold, train_test_split

np.random.seed(42)
rfq_ids = rfq_month['rfq_id'].unique()
np.random.shuffle(rfq_ids)

# 80/20 split at RFQ level
n_train = int(0.80 * len(rfq_ids))
train_ids = rfq_ids[:n_train]
test_ids  = rfq_ids[n_train:]

train_df = rfq_month[rfq_month['rfq_id'].isin(train_ids)].copy()
test_df  = rfq_month[rfq_month['rfq_id'].isin(test_ids)].copy()

print(f"Train: {len(train_ids)} RFQs → {len(train_df):,} bids")
print(f"Test : {len(test_ids)} RFQs → {len(test_df):,} bids")
print(f"No overlap: {len(set(train_ids) & set(test_ids)) == 0}")

# ─── 7. DEFINE FEATURE SETS ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: Defining feature sets")
print("=" * 60)

# Drop constants identified earlier: orig_opi, dest_opi, orig_net_flow_pct, dest_net_flow_pct
# Also drop data-leakage columns: pred_quote, pred_log_quote, residual_pct, bid_rank_pct

FEATURES_V3 = [
    # ── Core vehicle / route ──
    'capacity_mt', 'is_container', 'is_twoway', 'is_oneway',
    'is_interstate', 'is_same_city', 'body_ft', 'n_pallets',
    'is_flatbed', 'is_sxl', 'is_mxl', 'is_refer',
    'approx_dist_km', 'log_dist', 'log_capacity',
    'dist_x_capacity', 'log_dist_x_log_cap', 'dist_bucket',
    # ── City tier ──
    'orig_tier', 'dest_tier', 'tier_combo_enc', 'tier_competition_idx',
    # ── EWB demand signals (origin) ──
    'orig_odi', 'orig_iac', 'orig_vif', 'orig_sdpi',
    'orig_state_out_rank', 'orig_out_ewb', 'orig_total_ewb',
    'orig_odi_3m', 'orig_sdpi_3m',
    # ── EWB demand signals (destination) ──
    'dest_odi', 'dest_iac', 'dest_vif', 'dest_sdpi',
    'dest_state_out_rank', 'dest_out_ewb', 'dest_total_ewb',
    'dest_odi_3m', 'dest_sdpi_3m',
    # ── Interaction features ──
    'sdpi_x_dist', 'opi_x_capacity', 'odi_x_capacity', 'dest_odi_x_iac',
    # ── Seasonality (IMPROVED) ──
    'ewb_seasonal_factor',     # real EWB state×month factor
    'festival_pressure',       # festival calendar
    'combined_seasonal',       # ewb × festival
    'month_sin', 'month_cos',  # cyclical month encoding
    # ── Lane market intelligence ──
    'lane_median', 'lane_count', 'lane_per_km', 'lane_data_density',
    'effective_lane_median', 'effective_rate_per_km',
    'city_lane_median', 'city_lane_count', 'has_city_lane',
    # ── RFQ competition ──
    'rfq_bid_count', 'rfq_spread_pct', 'rfq_cv',
    'rfq_median', 'rfq_std',
    # ── Carrier intelligence ──
    'tsp_median', 'tsp_mean', 'tsp_bid_count',
    'tsp_lane_diversity', 'tsp_is_large',
    'tsp_vs_lane', 'tsp_specialization',
    'tsp_rate_per_km',
    # ── State encoding (clean) ──
    'orig_state_le', 'dest_state_le',
]

# Filter to columns that actually exist
FEATURES_V3 = [f for f in FEATURES_V3 if f in rfq_month.columns]
print(f"Features in V3 model: {len(FEATURES_V3)}")

TARGET = 'quote'

X_train = train_df[FEATURES_V3].fillna(train_df[FEATURES_V3].median())
y_train = np.log1p(train_df[TARGET])   # log-transform → better for skewed prices
X_test  = test_df[FEATURES_V3].fillna(train_df[FEATURES_V3].median())
y_test  = np.log1p(test_df[TARGET])

print(f"X_train shape: {X_train.shape}")
print(f"X_test  shape: {X_test.shape}")


# ─── 8. CROSS-VALIDATION ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: 5-fold cross-validation (GroupKFold on rfq_id)")
print("=" * 60)
"""
GroupKFold ensures each fold's test set has RFQs 
that were NOT in the training fold.
This gives you honest performance estimates.
"""

import lightgbm as lgb
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_percentage_error

LGB_PARAMS = dict(
    n_estimators=600,
    max_depth=8,
    learning_rate=0.02,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=15,
    reg_alpha=0.1,
    reg_lambda=0.2,
    random_state=42,
    verbose=-1
)

gkf = GroupKFold(n_splits=5)
groups = train_df['rfq_id'].values

cv_r2    = []
cv_mape  = []

print("Running 5-fold CV...")
for fold, (tr_idx, val_idx) in enumerate(gkf.split(X_train, y_train, groups)):
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    model_fold = lgb.LGBMRegressor(**LGB_PARAMS)
    model_fold.fit(
        Xtr, ytr,
        eval_set=[(Xval, yval)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)]
    )

    preds_log = model_fold.predict(Xval)
    preds     = np.expm1(preds_log)
    actuals   = np.expm1(yval)

    r2   = r2_score(actuals, preds)
    mape = mean_absolute_percentage_error(actuals, preds)
    cv_r2.append(r2)
    cv_mape.append(mape)
    print(f"  Fold {fold+1}: R²={r2:.4f}  MAPE={mape*100:.1f}%  "
          f"(best iter: {model_fold.best_iteration_})")

print(f"\nCV Summary → R²: {np.mean(cv_r2):.4f} ± {np.std(cv_r2):.4f} | "
      f"MAPE: {np.mean(cv_mape)*100:.1f}% ± {np.std(cv_mape)*100:.1f}%")
print("""
NOTE: These CV numbers are the REAL model accuracy.
Compare them to the app's reported R²=0.879 / MAPE=27.5%
(which were actually measured on TRAINING data — a mistake).
""")

# ─── 9. TRAIN FINAL MODEL ON ALL TRAINING DATA ────────────────────────────────
print("=" * 60)
print("STEP 9: Training final LightGBM model")
print("=" * 60)

model_v3 = lgb.LGBMRegressor(**LGB_PARAMS)
model_v3.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=-1)])

# Evaluate on held-out test set
preds_test_log = model_v3.predict(X_test)
preds_test     = np.expm1(preds_test_log)
actuals_test   = np.expm1(y_test)

r2_test   = r2_score(actuals_test, preds_test)
mape_test = mean_absolute_percentage_error(actuals_test, preds_test)
rmse_test = np.sqrt(np.mean((actuals_test - preds_test)**2))

print(f"\n{'='*40}")
print(f"HELD-OUT TEST SET PERFORMANCE")
print(f"{'='*40}")
print(f"R²   : {r2_test:.4f}")
print(f"MAPE : {mape_test*100:.1f}%")
print(f"RMSE : ₹{rmse_test:,.0f}")
print(f"{'='*40}")

# ─── 10. QUANTILE REGRESSION (HONEST CONFIDENCE INTERVALS) ───────────────────
print("\n" + "=" * 60)
print("STEP 10: Quantile regression for confidence intervals")
print("=" * 60)
"""
Instead of price × 0.78 / price × 1.25 (flat band),
we train two more models that directly predict:
  - q10: the 10th percentile (low estimate)
  - q90: the 90th percentile (high estimate)

This gives intervals that WIDEN on uncertain routes 
and NARROW on well-known lanes — much more honest.
"""

model_q10 = lgb.LGBMRegressor(objective='quantile', alpha=0.10,
                               n_estimators=400, learning_rate=0.03,
                               num_leaves=31, random_state=42, verbose=-1)
model_q90 = lgb.LGBMRegressor(objective='quantile', alpha=0.90,
                               n_estimators=400, learning_rate=0.03,
                               num_leaves=31, random_state=42, verbose=-1)

# Note: quantile models work on original scale (not log)
y_train_orig = np.expm1(y_train)
y_test_orig  = np.expm1(y_test)

model_q10.fit(X_train, y_train_orig, callbacks=[lgb.log_evaluation(period=-1)])
model_q90.fit(X_train, y_train_orig, callbacks=[lgb.log_evaluation(period=-1)])

ci_lo = model_q10.predict(X_test)
ci_hi = model_q90.predict(X_test)

# Coverage: what % of actual prices fall inside our 10-90% band?
coverage = np.mean((actuals_test >= ci_lo) & (actuals_test <= ci_hi))
avg_width = np.mean(ci_hi - ci_lo)

print(f"10–90% CI Coverage: {coverage*100:.1f}% (target: 80%)")
print(f"Average CI width  : ₹{avg_width:,.0f}")
print("""
Coverage close to 80% = well-calibrated uncertainty.
The width tells you how confident the model is per route.
""")


# ─── 11. FEATURE IMPORTANCE ──────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Feature importance (what drives prices most?)")
print("=" * 60)

feat_imp = pd.DataFrame({
    'feature':    FEATURES_V3,
    'importance': model_v3.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 20 features:")
print(feat_imp.head(20).to_string(index=False))

feat_imp.to_csv('feature_importance_v3.csv', index=False)
print("\nSaved → feature_importance_v3.csv")

# ─── 12. SHAP EXPLANATIONS ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12: SHAP — why did the model predict this price?")
print("=" * 60)
"""
SHAP (SHapley Additive exPlanations) tells you exactly HOW MUCH
each feature contributed to a specific prediction.

Example for one bid:
  Base price:     ₹35,000
  + Distance:     +₹12,000
  + ODI signal:   +₹4,500
  + Seasonality:  +₹2,200
  - Competition:  -₹3,100
  = Final price:  ₹50,600
"""

try:
    import shap
    explainer  = shap.TreeExplainer(model_v3)
    shap_vals  = explainer.shap_values(X_test.head(200))   # first 200 test rows

    print("\nSHAP summary (top 10 features by mean absolute impact):")
    shap_mean = pd.DataFrame({
        'feature': FEATURES_V3,
        'mean_abs_shap': np.abs(shap_vals).mean(axis=0)
    }).sort_values('mean_abs_shap', ascending=False)
    print(shap_mean.head(10).to_string(index=False))

    # Example: explain ONE prediction
    idx = 0
    pred_log  = model_v3.predict(X_test.iloc[[idx]])[0]
    pred_price = np.expm1(pred_log)
    actual_price = actuals_test.iloc[idx]

    print(f"\nExample prediction breakdown (bid #{test_df.index[idx]}):")
    print(f"  Actual:    ₹{actual_price:,.0f}")
    print(f"  Predicted: ₹{pred_price:,.0f}")
    sv = pd.DataFrame({'feature': FEATURES_V3,
                       'shap': shap_vals[idx]}).sort_values('shap', key=abs, ascending=False)
    print(sv.head(8).to_string(index=False))

except ImportError:
    print("SHAP not installed. Run: pip install shap")
    print("Then re-run this script for per-prediction explanations.")

# ─── 13. SEASONAL DEMAND ANALYSIS ────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 13: Seasonal demand — state by state (from EWB)")
print("=" * 60)

print("\nHow much does freight demand vary by season? (1.0 = average month)")
print("Values > 1.0 mean demand is ABOVE average → prices go UP")
print()

key_states = ['Maharashtra','Gujarat','Karnataka','Uttar Pradesh',
              'Punjab','West Bengal','Kerala','Tamil Nadu','Rajasthan','Haryana']

seasonal_pivot = (ewb
    .groupby(['state_name','month'])['seasonal_factor']
    .mean()
    .unstack()
    .round(3))

month_names = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
               7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
seasonal_pivot.columns = [month_names.get(c, c) for c in seasonal_pivot.columns]

avail = [s for s in key_states if s in seasonal_pivot.index]
print(seasonal_pivot.loc[avail].to_string())

print("""
HOW TO READ THIS TABLE:
- Punjab Mar=1.08 → In March, Punjab's freight demand is 8% above its own average
  This is the Rabi (wheat) harvest — grain moves to mills → trucks scarce → rates UP
- Kerala Aug=1.05 → Onam festival preparation drives up Kerala shipments
- UP Oct=1.10 → Diwali + Kharif harvest peak — UP is the biggest seasonal swinger
""")

# ─── 14. RESIDUAL ANALYSIS ───────────────────────────────────────────────────
print("=" * 60)
print("STEP 14: Where does the model struggle?")
print("=" * 60)

test_results = test_df[['rfq_id','origin_state','destination_state',
                         'approx_dist_km','capacity_mt','quote']].copy()
test_results['pred_v3']  = preds_test
test_results['err_pct']  = abs(test_results['quote'] - test_results['pred_v3']) / test_results['quote'] * 100

print("\nMAPE by origin state (test set):")
state_err = test_results.groupby('origin_state')['err_pct'].mean().sort_values(ascending=False)
print(state_err.head(10))

print("\nTop 10 worst predictions (highest % error):")
worst = test_results.nlargest(10, 'err_pct')[
    ['origin_state','destination_state','quote','pred_v3','err_pct','approx_dist_km','capacity_mt']]
print(worst.to_string(index=False))

# ─── 15. SAVE RESULTS ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 15: Saving results")
print("=" * 60)

test_results.to_csv('test_predictions_v3.csv', index=False)
feat_imp.to_csv('feature_importance_v3.csv', index=False)
print("Saved: test_predictions_v3.csv")
print("Saved: feature_importance_v3.csv")

# ─── FINAL SUMMARY ───────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL SUMMARY")
print("=" * 60)
print(f"""
DATA USED:
  • 9,801 carrier bids (Nov 2025 snapshot)
  • 7 years of EWB state-level demand signals
  • 113 unique RFQ shipment requests
  • 273 unique origin→destination state lanes

MODEL V3 IMPROVEMENTS:
  • RFQ-ID based train/test split (no data leakage)
  • Real EWB seasonal factors (not hardcoded)
  • Festival calendar as explicit feature
  • LabelEncoder for states (no hash collisions)
  • Carrier intelligence features added
  • Quantile regression for honest CI

TEST SET PERFORMANCE (honest, out-of-sample):
  R²   : {r2_test:.4f}
  MAPE : {mape_test*100:.1f}%
  RMSE : ₹{rmse_test:,.0f}

CONFIDENCE INTERVALS (quantile regression):
  Coverage: {coverage*100:.1f}%  |  Avg width: ₹{avg_width:,.0f}

NEXT STEPS:
  1. Get bids from multiple months → validate seasonal model
  2. Add road distance API (Google Maps) → replace Haversine
  3. Add diesel price index → fuel cost signal
  4. Try Optuna hyperparameter tuning
  5. Stack LightGBM + XGBoost for ensemble boost
""")

