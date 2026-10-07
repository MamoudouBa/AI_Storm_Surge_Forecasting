# Developed with the assistance of the Gemini AI code assistant.
#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
This script extracts and compiles continuous hindcast tracking series from the storm surge models
and generates multi-panel performance charts featuring an integrated validation statistics sidebar.
Ultimate Edition: Pure ETSS Error Correction Framework Evaluation (Dynamically Scaled Mode).
'''

import os, sys, gc
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import joblib

# Force headless Matplotlib execution BEFORE importing pyplot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

STATION_ID = sys.argv[1].strip() if len(sys.argv) > 1 else "8726520"

# 🌟 OPERATIONAL MODE: Pass "RESID" or "BIAS" as the second argument
MODE = sys.argv[2].strip().upper() if len(sys.argv) > 2 else "RESID"
if MODE not in ["RESID", "BIAS"]:
    print(f"⚠️ Unknown mode '{MODE}'. Defaulting to RESID.")
    MODE = "RESID"

# 🌟 TARGET SCALER SELECTION LINK (Supports "ROBUST" or "MAXABS")
TARGET_SCALER = "ROBUST"

EVAL_START = "2026-01-01 01:00:00"
EVAL_END = "2026-03-01 17:00:00"
c_val = 18

MODELS_PATH = '../storm_surge/'
OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FUTURE_TRAINED = 12
N_FUTURE_OP = 12
STN_MAP = {"8726520": "St_Petersburg_FL", "8557380": "Lewes_DE", "8761724": "Grand_Isle_LA"}
STATION_NAME = STN_MAP.get(STATION_ID, f"Station_{STATION_ID}")

def load_and_parse(file_path):
    df = pd.read_csv(file_path)
    t_col = 'timestamp_gui' if 'timestamp_gui' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

# --- 1. Load, Standardize, and Low-Pass Filter Input Files ---
try:
    print(f"🔄 Loading Master Datasets for {STATION_NAME} Ground Truth Sync...")
    df_g = load_and_parse(os.path.join(MODELS_PATH, f'test_gfs_etss_data_{STATION_ID}_2026.csv'))
    df_o = load_and_parse(os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2021_2026.csv'))

    # Clean duplicate frames safely
    df_o = df_o.drop_duplicates(subset=['dt']).set_index('dt').sort_index()
    df_g_unique = df_g.sort_values('cycle').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

    # Create the unified database matrix shell
    data_full = pd.DataFrame(index=df_g_unique.index.union(df_o.index)).sort_index()

    if 'etss_tide' in df_g_unique.columns:
        data_full['etss_tide'] = df_g_unique['etss_tide']
    else:
        t_col = [c for c in df_g_unique.columns if 'tide' in c.lower()]
        data_full['etss_tide'] = df_g_unique[t_col[0]] if t_col else 0.0

    data_full['water_level'] = df_o['water_level']

    # Calculate pure numerical surge realities
    data_full['etss_surge'] = df_g_unique['etss_water_level'] - data_full['etss_tide']
    raw_obs_surge = data_full['water_level'] - data_full['etss_tide']

    # Apply low-pass filter to observational surge truth
    data_full['obs_surge_filtered'] = pd.Series(raw_obs_surge.values, index=raw_obs_surge.index)\
                                        .rolling(window=12, center=True, min_periods=1).mean()

    # Create the requested pure physical targets
    if MODE == "RESID":
        data_full['target_metric'] = data_full['obs_surge_filtered'] - data_full['etss_surge']
    else:
        data_full['target_metric'] = data_full['etss_surge'] - data_full['obs_surge_filtered']

    if 'etss_water_level' in df_g_unique.columns:
        data_full['etss_water_level_baseline'] = df_g_unique['etss_water_level']
    else:
        data_full['etss_water_level_baseline'] = data_full['etss_tide']

    data_full = data_full.ffill().bfill()
    print("✅ Hardened Data Alignment and Target Transformation completed successfully.")
except Exception as e:
    print(f"❌ Plotter Data Mapping Alignment Error: {e}"); sys.exit()

available_cycles = sorted([pd.to_datetime(x) for x in df_g[(df_g['dt'] >= pd.to_datetime(EVAL_START)) & (df_g['dt'] <= pd.to_datetime(EVAL_END)) & (df_g['cycle'] == c_val)]['dt'].unique()])
print(f"📅 Tracked {len(available_cycles)} validation anchors inside target parameters.")

# --- 2. CACHED MODEL STORAGE ---
print(f"🧠 Pre-loading Pure ETSS {MODE} Error Models ({TARGET_SCALER} track) into static cache...")
cached_models = {}
cached_scalers = {}
lookbacks = [24, 48]
logics = ["PIVOT", "DIRECT"]

# Match the structural nomenclature of the target scaler mode
s_type_fn = "robustscaler" if TARGET_SCALER == "ROBUST" else "scaler"

for l_type in logics:
    for n_past in lookbacks:
        key = f"{l_type}_{n_past}_{TARGET_SCALER}"

        m_base = f"{STATION_NAME}_104_ETSS_DEV_{MODE}_{l_type}_{TARGET_SCALER}"
        m_name = f"{m_base}_{n_past}past_cycle{c_val:02d}Z"

        s_f = f"{STATION_NAME}_{s_type_fn}_etss_dev_{MODE.lower()}_{l_type.lower()}_f_{n_past}h_cycle{c_val:02d}Z.joblib"
        s_t = f"{STATION_NAME}_{s_type_fn}_etss_dev_{MODE.lower()}_{l_type.lower()}_t_{n_past}h_cycle{c_val:02d}Z.joblib"

        with open(os.path.join(MODELS_PATH, f"{m_name}.json"), 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(os.path.join(MODELS_PATH, f"{m_name}.h5"))

        cached_models[key] = model
        cached_scalers[f"{key}_f"] = joblib.load(os.path.join(MODELS_PATH, s_f))
        cached_scalers[f"{key}_t"] = joblib.load(os.path.join(MODELS_PATH, s_t))

print("✅ Global structure memory cache ready.")

# --- 3. Optimized Inference Engine ---
GUI_MET = ['etss_surge']
TARGET = 'target_metric'
feat_cols = GUI_MET + [TARGET]

def generate_isolated_engine_blend(logic_type, fc_origin, op_gui, anchor_target):
    local_horizons = {}
    target_len = len(op_gui)

    for n_past in lookbacks:
        key = f"{logic_type.upper()}_{n_past}_{TARGET_SCALER}"
        model = cached_models[key]
        sc_f = cached_scalers[f"{key}_f"]
        sc_t = cached_scalers[f"{key}_t"]

        raw_past = data_full.loc[:fc_origin][feat_cols].copy().values
        past_block = raw_past[-n_past:] if len(raw_past) >= n_past else np.vstack([np.repeat(raw_past[:1, :], n_past - len(raw_past), axis=0), raw_past])

        fut_etss_surge = (op_gui['etss_water_level'] - data_full.loc[op_gui.index, 'etss_tide']).copy().values.reshape(-1, 1)
        fut_fb = np.full((target_len, 1), anchor_target) if logic_type.upper() == "PIVOT" else np.zeros((target_len, 1))
        op_fut_block = np.hstack([fut_etss_surge, fut_fb])
        fut_sequence = op_fut_block[:N_FUTURE_TRAINED] if len(op_fut_block) >= N_FUTURE_TRAINED else np.vstack([op_fut_block, np.repeat(op_fut_block[-1:, :], N_FUTURE_TRAINED - len(op_fut_block), axis=0)])

        inp = np.ascontiguousarray(np.vstack([past_block, fut_sequence]))

        # --- 🌟 FIXED: Dynamic Forward Scale Layer transformation ---
        if TARGET_SCALER == "ROBUST" and hasattr(sc_f, 'center_') and hasattr(sc_f, 'scale_'):
            scaled_inp = (inp - np.atleast_1d(sc_f.center_)[0]) / np.atleast_1d(sc_f.scale_)[0]
        elif TARGET_SCALER == "MAXABS" and hasattr(sc_f, 'scale_'):
            scaled_inp = inp / np.atleast_1d(sc_f.scale_)[0]
        else:
            scaled_inp = (inp - np.mean(inp)) / (np.std(inp) + 1e-6)

        raw_pred = model.predict(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), verbose=0).flatten()
        aligned_pred = np.zeros(target_len)
        fill_len = min(len(raw_pred), target_len)
        aligned_pred[:fill_len] = raw_pred[:fill_len]

        # --- 🌟 FIXED: Dynamic Inverse Scale Layer transformation ---
        if TARGET_SCALER == "ROBUST" and hasattr(sc_t, 'center_') and hasattr(sc_t, 'scale_'):
            pred_final = (aligned_pred * np.atleast_1d(sc_t.scale_)[0]) + np.atleast_1d(sc_t.center_)[0]
        elif TARGET_SCALER == "MAXABS" and hasattr(sc_t, 'scale_'):
            pred_final = aligned_pred * np.atleast_1d(sc_t.scale_)[0]
        else:
            pred_final = aligned_pred

        local_horizons[n_past] = (pred_final.flatten()[:target_len] - pred_final[0]) + anchor_target

    timeline = np.arange(target_len)
    weight_24h = 1.0 / (1.0 + np.exp((timeline - 36) / 8))
    return ((local_horizons[24].flatten()[:target_len] * weight_24h) + (local_horizons[48].flatten()[:target_len] * (1.0 - weight_24h))).flatten()[:target_len]

# --- 4. Chronological Array Extraction Engine ---
print(f"📊 Running GRU array extraction across {len(available_cycles)} forecast steps...")
time_axis_records = []

for idx, fc_origin in enumerate(available_cycles):
    if idx % 6 != 0: continue

    if idx % 30 == 0:
        print(f"   ⏳ Processing timeline marker step {idx}/{len(available_cycles)}...")
        sys.stdout.flush()

    try:
        op_gui = df_g[(df_g['dt'] >= fc_origin) & (df_g['cycle'] == c_val)].copy().set_index('dt').sort_index().iloc[:N_FUTURE_OP]
        if len(op_gui) < N_FUTURE_OP: continue

        target_len = len(op_gui)

        anchor_target = data_full.loc[fc_origin, 'target_metric']
        if isinstance(anchor_target, pd.Series): anchor_target = anchor_target.iloc[0]

        piv_track = generate_isolated_engine_blend("PIVOT", fc_origin, op_gui, anchor_target).flatten()[:target_len]
        dir_track = generate_isolated_engine_blend("DIRECT", fc_origin, op_gui, anchor_target).flatten()[:target_len]
        cons_track = ((piv_track * 0.50) + (dir_track * 0.50)).flatten()[:target_len]

        obs_slice = data_full.loc[op_gui.index]

        # Reconstruct full reconstructed storm surge values based on Option physics routing
        etss_s_vals = obs_slice['etss_surge'].values.flatten()[:target_len]
        if MODE == "RESID":
            recon_ai_surge = etss_s_vals + cons_track
            recon_piv_surge = etss_s_vals + piv_track
            recon_dir_surge = etss_s_vals + dir_track
        else:
            recon_ai_surge = etss_s_vals - cons_track
            recon_piv_surge = etss_s_vals - piv_track
            recon_dir_surge = etss_s_vals - dir_track

        time_axis_records.append(pd.DataFrame({
            'obs_tide': obs_slice['etss_tide'].values.flatten()[:target_len],
            'obs_water_level': obs_slice['water_level'].values.flatten()[:target_len],
            'obs_surge': obs_slice['obs_surge_filtered'].values.flatten()[:target_len],
            'etss_surge': etss_s_vals,
            'etss_water_level': obs_slice['etss_water_level_baseline'].values.flatten()[:target_len],
            'pivot_surge': recon_piv_surge,
            'direct_surge': recon_dir_surge,
            'consensus_surge': recon_ai_surge
        }, index=op_gui.index[:target_len]))
    except Exception as inner_e:
        print(f"   ⚠️ Loop structural failure at step {idx}: {str(inner_e)}")
        sys.stdout.flush()
        continue

if not time_axis_records:
    print("❌ Error: Verification engine returned empty records. Cannot plot.")
    sys.exit()

print(f"✅ Extraction completed successfully. Merging sheets...")
master_plot_df = pd.concat(time_axis_records).groupby(level=0).mean()
master_plot_df['year_month'] = master_plot_df.index.to_period('M')

# Recompose Total Water Levels for AI metrics
master_plot_df['pivot_wl'] = master_plot_df['pivot_surge'] + master_plot_df['obs_tide']
master_plot_df['direct_wl'] = master_plot_df['direct_surge'] + master_plot_df['obs_tide']
master_plot_df['consensus_wl'] = master_plot_df['consensus_surge'] + master_plot_df['obs_tide']

# --- 5. Monthly Dual-Panel Plot + Side Panel Statistics Core ---
print("🎨 Rendering monthly performance charts with integrated statistics sidebar...")
for current_month in master_plot_df['year_month'].unique():
    m_slice = master_plot_df[master_plot_df['year_month'] == current_month].dropna()
    if len(m_slice) < 24: continue

    month_str = current_month.strftime('%B %Y')

    obs_surge   = m_slice['obs_surge'].values
    cons_surge  = m_slice['consensus_surge'].values
    piv_surge   = m_slice['pivot_surge'].values
    dir_surge  = m_slice['direct_surge'].values
    etss_surge  = m_slice['etss_surge'].values

    rmse_cons_s = np.sqrt(np.mean((cons_surge - obs_surge) ** 2))
    rmse_piv_s  = np.sqrt(np.mean((piv_surge - obs_surge) ** 2))
    rmse_dir_s  = np.sqrt(np.mean((dir_surge - obs_surge) ** 2))
    rmse_etss_s = np.sqrt(np.mean((etss_surge - obs_surge) ** 2))

    obs_wl   = m_slice['obs_water_level'].values
    cons_wl  = m_slice['consensus_wl'].values
    piv_wl   = m_slice['pivot_wl'].values
    dir_wl   = m_slice['direct_wl'].values
    etss_wl  = m_slice['etss_water_level'].values

    rmse_cons_w = np.sqrt(np.mean((cons_wl - obs_wl) ** 2))
    rmse_piv_w  = np.sqrt(np.mean((piv_wl - obs_wl) ** 2))
    rmse_dir_w  = np.sqrt(np.mean((dir_wl - obs_wl) ** 2))
    rmse_etss_w = np.sqrt(np.mean((etss_wl - obs_wl) ** 2))

    bias_cons_w = np.mean(cons_wl - obs_wl)
    bias_etss_w = np.mean(etss_wl - obs_wl)

    skill_wl = ((rmse_etss_w - rmse_cons_w) / rmse_etss_w) * 100.0

    fig = plt.figure(figsize=(21, 12))

    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[0.75, 0.25], hspace=0.15, wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_stat = fig.add_subplot(gs[:, 1])
    ax_stat.axis('off')

    # --- PANEL 1: Total Water Level Space ---
    ax1.plot(m_slice.index, m_slice['obs_water_level'], label='Observed Total Water (Truth)', color='#2c3e50', linewidth=2.5)
    ax1.plot(m_slice.index, m_slice['consensus_wl'], label='AI GRAND CONSENSUS (Total Water)', color='#27ae60', linewidth=1.8)
    ax1.plot(m_slice.index, m_slice['pivot_wl'], label='AI Ultimate Pivot', color='#2980b9', linestyle=':', linewidth=1.5)
    ax1.plot(m_slice.index, m_slice['direct_wl'], label='AI Ultimate Direct', color='#8e44ad', linestyle=':', linewidth=1.5)
    ax1.plot(m_slice.index, m_slice['etss_water_level'], label='ETSS Physics Baseline WL', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)
    ax1.plot(m_slice.index, m_slice['obs_tide'], label='Astronomical Tide Baseline', color='#7f8c8d', linestyle='-', linewidth=1.0, alpha=0.3)

    ax1.set_title(f"Deep Learning Storm Surge Forecast Suite (ETSS {MODE} Correction Framework): {STATION_NAME} ({month_str})", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

    # --- PANEL 2: Cleaned Meteorological Surge Residual Space ---
    ax2.plot(m_slice.index, m_slice['obs_surge'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.5)
    ax2.plot(m_slice.index, m_slice['consensus_surge'], label='AI GRAND CONSENSUS (Reconstructed Surge)', color='#27ae60', linewidth=1.8)
    ax2.plot(m_slice.index, m_slice['pivot_surge'], label='AI Ultimate Pivot Model', color='#2980b9', linestyle=':', linewidth=1.5)
    ax2.plot(m_slice.index, m_slice['direct_surge'], label='AI Ultimate Direct Model', color='#8e44ad', linestyle=':', linewidth=1.5)
    ax2.plot(m_slice.index, m_slice['etss_surge'], label='ETSS Physics Baseline Model', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)

    ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
    ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
    ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

    fig.autofmt_xdate()

    # --- SIDE PANEL: Validation Scorecard ---
    stat_box_text = (
        f"     VALIDATION CARD
"
        f"=========================
"
        f"  Month: {current_month.strftime('%m/%Y')}
"
        f"  Station Target: {STATION_ID}
"
        f"  ETSS Dev Mode: {MODE}
"
        f"  Forced Scaler: {TARGET_SCALER}

"
        f"  [1] TOTAL WATER LEVEL (WL)
"
        f"  -----------------------
"
        f"  * AI GRAND CONSENSUS:
"
        f"    > RMSE : {rmse_cons_w:.3f} ft
"
        f"    > Bias : {bias_cons_w:+.3f} ft
"
        f"  * AI ULTIMATE PIVOT:
"
        f"    > RMSE : {rmse_piv_w:.3f} ft
"
        f"  * AI ULTIMATE DIRECT:
"
        f"    > RMSE : {rmse_dir_w:.3f} ft
"
        f"  * ETSS PHYSICS BASELINE:
"
        f"    > RMSE : {rmse_etss_w:.3f} ft
"
        f"    > Bias : {bias_etss_w:+.3f} ft

"
        f"  [2] PURE SURGE RESIDUAL
"
        f"  -----------------------
"
        f"  * AI GRAND CONSENSUS:
"
        f"    > RMSE : {rmse_cons_s:.3f} ft
"
        f"  * AI ULTIMATE PIVOT:
"
        f"    > RMSE : {rmse_piv_s:.3f} ft
"
        f"  * AI ULTIMATE DIRECT:
"
        f"    > RMSE : {rmse_dir_s:.3f} ft
"
        f"  * ETSS PHYSICS BASELINE:
"
        f"    > RMSE : {rmse_etss_s:.3f} ft

"
        f"  OPERATIONAL PERFORMANCE
"
        f"  -----------------------
"
        f"  AI WL Error Reduction:
"
        f"    * {skill_wl:+.1f}% Improvement
"
        f"      over Physics Baseline"
    )

    ax_stat.text(0.02, 0.98, stat_box_text, transform=ax_stat.transAxes,
                 fontsize=9.0, verticalalignment='top', fontname='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#cfd8dc', linewidth=1.2))

    plot_out = os.path.join(OUTPUT_DIR, f"etss_dev_{TARGET_SCALER}_{MODE.lower()}_wl_metrics_{STATION_ID}_{current_month.strftime('%Y_%m')}.png")
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')

    plt.close(fig)
    fig.clear()
    ax1.clear()
    ax2.clear()
    ax_stat.clear()
    del fig, ax1, ax2, ax_stat, m_slice
    K.clear_session()
    gc.collect()

    print(f" 💾 Summary chart compiled successfully: {plot_out}")

print("🎉 Complete error correction validation consensus dashboard charts rendered successfully.")
