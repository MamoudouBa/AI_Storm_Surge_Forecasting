#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Script developed with the assistance of the Gemini AI code assistant.
This script extracts and compiles continuous hindcast tracking series from the storm surge models
and generates multi-panel performance charts featuring an integrated validation statistics sidebar.
Ultimate Edition: Pure GFS Weather-Only Surrogate Model Multi-Station Validation Engine.
Tailored for the Simplified DTW Loss variant outputs.
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

# 🌟 TARGET SCALER & SYSTEM DEFINITIONS
TARGET_SCALER = "ROBUST"
EVAL_START = "2026-01-01 01:00:00"
EVAL_END = "2026-03-01 17:00:00"
c_val = 18

MODELS_PATH = '../storm_surge/'
CONFIG_FILE = '../storm_surge/stations_config_gulf_atlantic.csv'
OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FUTURE_TRAINED = 12
N_FUTURE_OP = 12

def load_and_parse(file_path):
    df = pd.read_csv(file_path)
    t_col = 'timestamp_gui' if 'timestamp_gui' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

# --- DYNAMIC MASTER MATRIX CONFIGURATION LOADER ---
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Master station configuration matrix missing at: {CONFIG_FILE}")
    sys.exit(1)

print(f"📖 Parsing master CSV registry tracks from {CONFIG_FILE}...")
df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
df_config['station_id'] = df_config['station_id'].str.strip()
df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')\n# --- 🚀 MASTER PLOTTER OUTER LOOP ---
for idx_stn, row in df_config.iterrows():
    STATION_ID = row['station_id']
    STATION_NAME = row['station_name']
    if STATION_ID != "8557380": continue

    df_g_path = os.path.join(MODELS_PATH, f'test_gfs_etss_data_{STATION_ID}_2026.csv')
    df_o_path = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2021_2026.csv')

    if not (os.path.exists(df_g_path) and os.path.exists(df_o_path)):
        print(f"⚠️ [Skipping] Data files absent for station: {STATION_NAME} ({STATION_ID})")
        continue

    print("\n" + "="*80)
    print(f"🎨 COMPILING MET-ONLY SURROGATE PLOTS FOR: {STATION_NAME} ({STATION_ID})")
    print("="*80)

    # --- 1. Load, Standardize, and Low-Pass Filter Input Files ---
    try:
        df_g = load_and_parse(df_g_path)
        df_o = load_and_parse(df_o_path)

        df_o = df_o.drop_duplicates(subset=['dt']).set_index('dt').sort_index()
        df_g_unique = df_g.sort_values('cycle').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

        data_full = pd.DataFrame(index=df_g_unique.index.union(df_o.index)).sort_index()

        if 'etss_tide' in df_g_unique.columns:
            data_full['etss_tide'] = df_g_unique['etss_tide']
        else:
            t_col = [c for c in df_g_unique.columns if 'tide' in c.lower()]
            data_full['etss_tide'] = df_g_unique[t_col[0]] if t_col else 0.0

        data_full['water_level'] = df_o['water_level']
        raw_surge = data_full['water_level'] - data_full['etss_tide']

        # Center-rolling 12-hour low-pass tidal filter to define true verification target
        data_full['surge_residual'] = pd.Series(raw_surge.values, index=raw_surge.index)\
                                            .rolling(window=12, center=True, min_periods=1).mean()

        if 'etss_water_level' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['etss_water_level']
        else:
            data_full['etss_water_level_baseline'] = data_full['etss_tide']

        # Map dynamic weather profile fields
        for col in ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui']:
            if col in df_g_unique.columns:
                data_full[col] = df_g_unique[col]

        # =====================================================================
        # 🌟 VALIDATION LAYER WIND DECOMPOSITION VECTORS (U & V Mapping)
        # =====================================================================
        rad_val = np.radians(data_full['wind_direction_gui'].values)
        data_full['wind_u'] = data_full['wind_speed_gui'].values * np.cos(rad_val)
        data_full['wind_v'] = data_full['wind_speed_gui'].values * np.sin(rad_val)

        data_full = data_full.ffill().bfill()
    except Exception as e:
        print(f"❌ Plotter Data Alignment Exception on {STATION_NAME}: {e}")
        continue

    available_cycles = sorted([pd.to_datetime(x) for x in df_g[(df_g['dt'] >= pd.to_datetime(EVAL_START)) & (df_g['dt'] <= pd.to_datetime(EVAL_END)) & (df_g['cycle'] == c_val)]['dt'].unique()])
    if not available_cycles:
        print(f"⚠️ No forecast validation anchor cycles found within specified window for {STATION_NAME}.")
        continue

    # --- 2. CACHED PURE ATMOSPHERIC SURROGATE STORAGE ---
    cached_models = {}
    cached_scalers = {}
    lookbacks = [24, 48]  
    logics = ["PIVOT", "DIRECT"]

    load_failure = False
    for l_type in logics:
        for n_past in lookbacks:
            key = f"{l_type}_{n_past}_{TARGET_SCALER}"

            # 🌟 UPDATED: Matches your exact training token patterns and standalone target scaler file arrays
            m_name = f"{STATION_NAME}_{N_FUTURE_TRAINED}_OPERATIONAL_GFS_{l_type.upper()}_SIMPLIFIED_DILATE_{n_past}past"
            s_f = f"{STATION_NAME}_scaler_operational_simplified_{l_type.lower()}_robust_{n_past}h.joblib"
            s_t = f"{STATION_NAME}_scaler_operational_simplified_{l_type.lower()}_target_robust_{n_past}h.joblib"

            try:
                with open(os.path.join(MODELS_PATH, f"{m_name}.json"), 'r') as f:
                    model = model_from_json(f.read())
                model.load_weights(os.path.join(MODELS_PATH, f"{m_name}.h5"))

                cached_models[key] = model
                cached_scalers[f"{key}_f"] = joblib.load(os.path.join(MODELS_PATH, s_f))
                cached_scalers[f"{key}_t"] = joblib.load(os.path.join(MODELS_PATH, s_t))
                print(f"   > Successfully loaded components for key tracking node: {key}")
            except Exception as model_err:
                print(f"⚠️ Missing deep learning components on key {key}: {model_err}")
                load_failure = True
                break
        if load_failure: break
    if load_failure: continue

    # --- 3. Pure Weather Inference Matrix Engine ---
    GUI_MET = ['air_temp_gui', 'air_pressure_gui', 'wind_u', 'wind_v']
    TARGET = 'surge_residual'
    feat_cols = GUI_MET + [TARGET]

    def generate_isolated_engine_blend(logic_type, fc_origin, op_gui, anchor_surge):
        local_horizons = {}
        target_len = len(op_gui)

        for n_past in lookbacks:
            key = f"{logic_type.upper()}_{n_past}_{TARGET_SCALER}"
            model = cached_models[key]
            sc_f = cached_scalers[f"{key}_f"]
            sc_t = cached_scalers[f"{key}_t"]

            raw_past = data_full.loc[:fc_origin][feat_cols].copy().values
            past_block = raw_past[-n_past:] if len(raw_past) >= n_past else np.vstack([np.repeat(raw_past[:1, :], n_past - len(raw_past), axis=0), raw_past])

            rad_fut = np.radians(op_gui['wind_direction_gui'].values)
            fut_u = op_gui['wind_speed_gui'].values * np.cos(rad_fut)
            fut_v = op_gui['wind_speed_gui'].values * np.sin(rad_fut)

            fut_met_profile = np.hstack([
                op_gui['air_temp_gui'].values.reshape(-1, 1),
                op_gui['air_pressure_gui'].values.reshape(-1, 1),
                fut_u.reshape(-1, 1),
                fut_v.reshape(-1, 1)
            ])

            fut_fb = np.full((target_len, 1), anchor_surge) if logic_type.upper() == "PIVOT" else np.zeros((target_len, 1))
            op_fut_block = np.hstack([fut_met_profile, fut_fb])
            fut_sequence = op_fut_block[:N_FUTURE_TRAINED] if len(op_fut_block) >= N_FUTURE_TRAINED else np.vstack([op_fut_block, np.repeat(op_fut_block[-1:, :], N_FUTURE_TRAINED - len(op_fut_block), axis=0)])

            inp = np.ascontiguousarray(np.vstack([past_block, fut_sequence]))
            scaled_inp = sc_f.transform(inp)

            raw_pred = model.predict(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), verbose=0).flatten()
            aligned_pred = np.zeros(target_len)
            fill_len = min(len(raw_pred), target_len)
            aligned_pred[:fill_len] = raw_pred[:fill_len]

            pred_final = sc_t.inverse_transform(aligned_pred.reshape(-1, 1)).flatten()
            local_horizons[n_past] = (pred_final.flatten()[:target_len] - pred_final[0]) + anchor_surge

        timeline = np.arange(target_len)
        weight_24h = 1.0 / (1.0 + np.exp((timeline - (target_len / 2)) / 2))
        return ((local_horizons[24].flatten()[:target_len] * weight_24h) + (local_horizons[48].flatten()[:target_len] * (1.0 - weight_24h))).flatten()[:target_len]

    # --- 4. Chronological Verification Extraction Engine ---
    time_axis_records = []

    for idx, fc_origin in enumerate(available_cycles):
        if idx % 6 != 0: continue

        try:
            op_gui = df_g[(df_g['dt'] >= fc_origin) & (df_g['cycle'] == c_val)].copy().set_index('dt').sort_index().iloc[:N_FUTURE_OP]
            if len(op_gui) < N_FUTURE_OP: continue

            target_len = len(op_gui)
            obs_raw = df_o.reindex(op_gui.index)
            tide_vals = df_g_unique.reindex(op_gui.index)['etss_tide'].fillna(0.0).values.flatten()[:target_len]

            obs_surge_clean = (obs_raw['water_level'].values - tide_vals)
            obs_surge_filtered = pd.Series(obs_surge_clean).rolling(window=12, center=True, min_periods=1).mean().values

            anchor_surge = data_full.loc[fc_origin, 'surge_residual']
            if isinstance(anchor_surge, pd.Series): anchor_surge = anchor_surge.iloc[0]

            piv_track = generate_isolated_engine_blend("PIVOT", fc_origin, op_gui, anchor_surge).flatten()[:target_len]
            dir_track = generate_isolated_engine_blend("DIRECT", fc_origin, op_gui, anchor_surge).flatten()[:target_len]
            cons_track = ((piv_track * 0.50) + (dir_track * 0.50)).flatten()[:target_len]

            time_axis_records.append(pd.DataFrame({
                'obs_tide': tide_vals,
                'obs_water_level': obs_raw['water_level'].values.flatten()[:target_len],
                'obs_surge': obs_surge_filtered,
                'etss_surge': (data_full.loc[op_gui.index, 'etss_water_level_baseline'] - tide_vals).values.flatten()[:target_len],
                'etss_water_level': data_full.loc[op_gui.index, 'etss_water_level_baseline'].values.flatten()[:target_len],
                'pivot_surge': piv_track,
                'direct_surge': dir_track,
                'consensus_surge': cons_track
            }, index=op_gui.index[:target_len]))
        except Exception:
            continue

    if not time_axis_records:
        print(f"⚠️ Warning: Verification vector stack completely empty for {STATION_NAME}. No matching times.")
        continue

    master_plot_df = pd.concat(time_axis_records).groupby(level=0).mean()
    master_plot_df['year_month'] = master_plot_df.index.to_period('M')

    master_plot_df['pivot_wl'] = master_plot_df['pivot_surge'] + master_plot_df['obs_tide']
    master_plot_df['direct_wl'] = master_plot_df['direct_surge'] + master_plot_df['obs_tide']
    master_plot_df['consensus_wl'] = master_plot_df['consensus_surge'] + master_plot_df['obs_tide']

    # =====================================================================
    # 🌟 PRODUCTION PATCH: HARDENED DATA VALIDATION & EXTRACTION AUDIT
    # =====================================================================
    print(f"    📊 Data Validation Audit Matrix for {STATION_NAME}:")
    print(f"      > Total Extracted Rows    : {master_plot_df.shape[0]}")
    print(f"      > Timestamp Coverage Span : {master_plot_df.index.min()} to {master_plot_df.index.max()}")
    print(f"      > Missing Observed WL NaN: {master_plot_df['obs_water_level'].isna().sum()} rows")
    print(f"      > Missing AI Consensus NaN: {master_plot_df['consensus_surge'].isna().sum()} rows")
    sys.stdout.flush()

    # --- 5. Monthly Dual-Panel Plot Generating Layer ---
    for current_month in master_plot_df['year_month'].unique():
        m_slice = master_plot_df[master_plot_df['year_month'] == current_month].dropna()

        if len(m_slice) < 1:
            print(f"    ⚠️ [Skipping Month {current_month}] Zero valid records available after dropping NaN rows.")
            continue

        print(f"    🎨 Rendering target chart for tracking window: {current_month} ({len(m_slice)} valid hours)")
        sys.stdout.flush()

        month_str = current_month.strftime('%B %Y')

        obs_surge = m_slice['obs_surge'].values
        cons_surge = m_slice['consensus_surge'].values
        piv_surge = m_slice['pivot_surge'].values
        dir_surge = m_slice['direct_surge'].values
        etss_surge = m_slice['etss_surge'].values

        rmse_cons_s = np.sqrt(np.mean((cons_surge - obs_surge) ** 2))
        rmse_piv_s = np.sqrt(np.mean((piv_surge - obs_surge) ** 2))
        rmse_dir_s = np.sqrt(np.mean((dir_surge - obs_surge) ** 2))
        rmse_etss_s = np.sqrt(np.mean((etss_surge - obs_surge) ** 2))

        obs_wl = m_slice['obs_water_level'].values
        cons_wl = m_slice['consensus_wl'].values
        piv_wl = m_slice['pivot_wl'].values
        dir_wl = m_slice['direct_wl'].values
        etss_wl = m_slice['etss_water_level'].values

        rmse_cons_w = np.sqrt(np.mean((cons_wl - obs_wl) ** 2))
        rmse_piv_w = np.sqrt(np.mean((piv_wl - obs_wl) ** 2))
        rmse_dir_w = np.sqrt(np.mean((dir_wl - obs_wl) ** 2))
        rmse_etss_w = np.sqrt(np.mean((etss_wl - obs_wl) ** 2))

        bias_cons_w = np.mean(cons_wl - obs_wl)
        bias_etss_w = np.mean(etss_wl - obs_wl)

        skill_wl = ((rmse_etss_w - rmse_cons_w) / rmse_etss_w) * 100.0 if rmse_etss_w > 0 else 0.0

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

        ax1.set_title(f"Deep Learning Storm Surge Forecast Suite (Pure GFS Weather-Only Surrogate Mode): {STATION_NAME} ({month_str})", fontsize=13, fontweight='bold', pad=12)
        ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
        ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

        # --- PANEL 2: Cleaned Meteorological Surge Residual Space ---
        ax2.plot(m_slice.index, m_slice['obs_surge'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.5)
        ax2.plot(m_slice.index, m_slice['consensus_surge'], label='AI GRAND CONSENSUS (Surrogate Surge)', color='#27ae60', linewidth=1.8)
        ax2.plot(m_slice.index, m_slice['pivot_surge'], label='AI Ultimate Pivot Model', color='#2980b9', linestyle=':', linewidth=1.5)
        ax2.plot(m_slice.index, m_slice['direct_surge'], label='AI Ultimate Direct Model', color='#8e44ad', linestyle=':', linewidth=1.5)
        ax2.plot(m_slice.index, m_slice['etss_surge'], label='ETSS Physics Baseline Model', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)

        ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
        ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
        ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

        fig.autofmt_xdate()

        # --- SIDE PANEL: CLEAN VALIDATION SCORECARD SIDEBAR ---
        stat_box_text = (
            f"      VALIDATION CARD\n"
            f"=========================\n"
            f"  Month: {current_month.strftime('%m/%Y')}\n"
            f"  Station Target: {STATION_ID}\n"
            f"  Surrogate Mode: GFS-ONLY\n"
            f"  Forced Scaler: {TARGET_SCALER}\n\n"
            f"  [1] TOTAL WATER LEVEL (WL)\n"
            f"  -----------------------\n"
            f"  * AI GRAND CONSENSUS:\n"
            f"    > RMSE : {rmse_cons_w:.3f} ft\n"
            f"    > Bias : {bias_cons_w:+.3f} ft\n"
            f"  * AI ULTIMATE PIVOT:\n"
            f"    > RMSE : {rmse_piv_w:.3f} ft\n"
            f"  * AI ULTIMATE DIRECT:\n"
            f"    > RMSE : {rmse_dir_w:.3f} ft\n"
            f"  * ETSS PHYSICS BASELINE:\n"
            f"    > RMSE : {rmse_etss_w:.3f} ft\n"
            f"    > Bias : {bias_etss_w:+.3f} ft\n\n"
            f"  [2] PURE SURGE RESIDUAL\n"
            f"  -----------------------\n"
            f"  * AI GRAND CONSENSUS:\n"
            f"    > RMSE : {rmse_cons_s:.3f} ft\n"
            f"  * AI ULTIMATE PIVOT:\n"
            f"    > RMSE : {rmse_piv_s:.3f} ft\n"
            f"  * AI ULTIMATE DIRECT:\n"
            f"    > RMSE : {rmse_dir_s:.3f} ft\n"
            f"  * ETSS PHYSICS BASELINE:\n"
            f"    > RMSE : {rmse_etss_s:.3f} ft\n\n"
            f"  OPERATIONAL PERFORMANCE\n"
            f"  -----------------------\n"
            f"  AI WL Error Reduction:\n"
            f"    * {skill_wl:+.1f}% Improvement"
        )

        ax_stat.text(0.05, 0.95, stat_box_text, transform=ax_stat.transAxes,
                     fontsize=9.5, verticalalignment='top', fontname='monospace',
                     bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#eceff1', linewidth=1.0))

        plot_out = os.path.join(OUTPUT_DIR, f"gfs_only_simplified_dilate_metrics_{STATION_ID}_{current_month.strftime('%Y_%m')}.png")
        plt.savefig(plot_out, dpi=300, bbox_inches='tight')
        print(f"    💾 Operational hydrograph chart successfully exported:\n       ➡️ {plot_out}")
        
        plt.close(fig)
        fig.clear()
        ax1.clear()
        ax2.clear()
        ax_stat.clear()

    K.clear_session()
    gc.collect()

print("\n🎉 Master Simplified Loss Plot Engine Complete.")
