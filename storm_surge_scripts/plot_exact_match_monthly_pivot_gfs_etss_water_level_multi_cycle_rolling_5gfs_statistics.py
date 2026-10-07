#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Script developed with the assistance of the Gemini AI code assistant.
Operational GFS + ETSS Hybrid Pivot Multi-Station 12h Rolling Evaluation Engine.
Evaluates continuous 12-hour rolling forecasts across full monthly evaluation windows.
Updated with linear boundary transition (fut_fb) and zero-anchored relative delta.
Generates verification plots matching the target +4.8% WL Error Reduction plot.
'''

import os, sys, gc
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

# 🌟 SYSTEM & MODEL DEFINITIONS
EVAL_START = "2026-01-01 00:00:00"
EVAL_END = "2026-02-01 23:00:00"

MODELS_PATH = '../storm_surge/'
CONFIG_FILE = '../storm_surge/stations_config_gulf_atlantic.csv'
OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FUTURE_TRAINED = 12
lookbacks = [24, 48]

def load_and_parse(file_path):
    df = pd.read_csv(file_path)
    t_col = 'timestamp_gui' if 'timestamp_gui' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

# --- 1. DYNAMIC CONFIGURATION LOADER ---
if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Master station configuration matrix missing at: {CONFIG_FILE}")
    sys.exit(1)

print(f"📖 Parsing master CSV registry tracks from {CONFIG_FILE}...")
df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
df_config['station_id'] = df_config['station_id'].str.strip()
df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')

# --- 🚀 MASTER EVALUATION LOOP ACROSS ALL STATIONS ---
for idx_stn, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    STATION_NAME = str(row['station_name']).strip()

    df_g_path = os.path.join(MODELS_PATH, f'test_gfs_etss_data_{STATION_ID}_2026.csv')
    df_o_path = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2021_2026.csv')
    if not os.path.exists(df_o_path):
        df_o_path = os.path.join(MODELS_PATH, f'obs_storms_{STATION_ID}_2021_2026.csv')

    if not (os.path.exists(df_g_path) and os.path.exists(df_o_path)):
        print(f"⚠️ [Skipping] Data files absent for station: {STATION_NAME} ({STATION_ID})")
        continue

    print("\n" + "="*80)
    print(f"🎨 COMPILING CONTINUOUS 12H ROLLING FORECAST FOR: {STATION_NAME} ({STATION_ID})")
    print("="*80)

    # --- 2. Load and Process Input Data Streams ---
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
        data_full['surge_residual'] = pd.Series(raw_surge.values, index=raw_surge.index)\
                                            .rolling(window=12, center=True, min_periods=1).mean()

        if 'water_level_etss' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['water_level_etss']
        elif 'etss_water_level' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['etss_water_level']
        else:
            data_full['etss_water_level_baseline'] = data_full['etss_tide']

        raw_etss_surge = data_full['etss_water_level_baseline'] - data_full['etss_tide']
        data_full['etss_surge_guidance'] = pd.Series(raw_etss_surge.values, index=raw_etss_surge.index)\
                                                  .rolling(window=12, center=True, min_periods=1).mean()
        data_full['etss_water_level_baseline'] = data_full['etss_surge_guidance'] + data_full['etss_tide']

        for col in ['air_pressure', 'wind_speed', 'air_temp', 'wind_gust',
                    'air_pressure_gui', 'wind_speed_gui', 'air_temp_gui', 'wind_gust_gui',
                    'wind_direction', 'wind_direction_gui']:
            if col in df_g_unique.columns:
                data_full[col] = df_g_unique[col]
            elif col in df_o.columns:
                data_full[col] = df_o[col]

        data_full = data_full.ffill().bfill()
    except Exception as e:
        print(f"❌ Data Alignment Exception on {STATION_NAME}: {e}")
        continue

    # --- 3. Load Model Assets ---
    cached_models = {}
    cached_scalers = {}
    load_failure = False

    for n_past in lookbacks:
        key = f"CHRONO_{n_past}"
        m_name = f"{STATION_NAME}_{N_FUTURE_TRAINED}_GFS_ETSS_PIVOT_SAMPLE_WEIGHTING_ROBUST_{n_past}past_CHRONO"
        s_f_name = f"{STATION_NAME}_scaler_gfs_etss_pivot_sample_weighting_robust_{n_past}h_CHRONO.joblib"
        s_t_name = f"{STATION_NAME}_scaler_gfs_etss_pivot_target_sample_weighting_robust_{n_past}h_CHRONO.joblib"

        m_json_path = os.path.join(MODELS_PATH, f"{m_name}.json")
        m_h5_path = os.path.join(MODELS_PATH, f"{m_name}.h5")
        sf_path = os.path.join(MODELS_PATH, s_f_name)
        st_path = os.path.join(MODELS_PATH, s_t_name)

        try:
            if os.path.exists(m_json_path) and os.path.exists(sf_path):
                with open(m_json_path, 'r') as f:
                    model = model_from_json(f.read())
                model.load_weights(m_h5_path)

                cached_models[key] = model
                cached_scalers[f"{key}_f"] = joblib.load(sf_path)
                cached_scalers[f"{key}_t"] = joblib.load(st_path)
                print(f"    > Loaded Unified CHRONO Asset Matrix: {key} ({m_name})")
            else:
                print(f"❌ Missing CHRONO model components for key {key} at {m_json_path}")
                load_failure = True
                break
        except Exception as model_err:
            print(f"❌ Error loading CHRONO assets on key {key}: {model_err}")
            load_failure = True
            break

    if load_failure:
        continue

    # --- 4. 12-Hour Continuous Rolling Inference Engine ---
    eval_window = data_full.loc[EVAL_START:EVAL_END]
    if eval_window.empty:
        print(f"⚠️ No data in specified evaluation range [{EVAL_START} to {EVAL_END}].")
        continue

    rolling_preds = {24: pd.Series(index=eval_window.index, dtype=float),
                     48: pd.Series(index=eval_window.index, dtype=float)}

    cycle_anchors = eval_window.index[::N_FUTURE_TRAINED]

    for t_anchor in cycle_anchors:
        anchor_idx = data_full.index.get_loc(t_anchor)

        for n_past in lookbacks:
            key = f"CHRONO_{n_past}"
            model = cached_models[key]
            sc_f = cached_scalers[f"{key}_f"]
            sc_t = cached_scalers[f"{key}_t"]

            if anchor_idx < n_past:
                continue

            # Dynamic Feature Detector
            n_feat = sc_f.n_features_in_
            if n_feat == 7:
                OBS_FEAT = ['air_pressure', 'air_temp', 'wind_speed', 'wind_direction', 'wind_gust', 'etss_surge_guidance']
                GUI_FEAT = ['air_pressure_gui', 'air_temp_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui', 'etss_surge_guidance']
            elif n_feat == 5:
                OBS_FEAT = ['wind_direction', 'wind_gust', 'etss_surge_guidance', 'air_temp']
                GUI_FEAT = ['wind_direction_gui', 'wind_gust_gui', 'etss_surge_guidance', 'air_temp_gui']
            else:
                raise ValueError(f"Scaler expects unsupported feature dimension: {n_feat}")

            TARGET = 'surge_residual'

            past_block = data_full.iloc[anchor_idx - n_past : anchor_idx][OBS_FEAT + [TARGET]].values
            current_pivot = data_full.iloc[anchor_idx]['surge_residual']

            fut_chunk = data_full.iloc[anchor_idx : anchor_idx + N_FUTURE_TRAINED]
            if len(fut_chunk) < N_FUTURE_TRAINED:
                continue

            # 🌟 ORIGINAL +4.8% INFERENCE PROTOCOL
            fut_met_profile = fut_chunk[GUI_FEAT].values
            
            # 1. Smooth Linear Boundary Transition for Future Feedback Channel
            last_known_surge = past_block[-1, -1]
            fut_fb = np.linspace(last_known_surge, current_pivot, N_FUTURE_TRAINED).reshape(-1, 1)

            op_fut_block = np.hstack([fut_met_profile, fut_fb])
            inp = np.ascontiguousarray(np.vstack([past_block, op_fut_block]))
            scaled_inp = sc_f.transform(inp)

            # 2. Forward Pass & Scale Inversion
            raw_pred = model.predict(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), verbose=0).flatten()
            pred_delta = sc_t.inverse_transform(raw_pred.reshape(-1, 1)).flatten()

            # 3. Relative Zero-Anchored Pivot Re-Elevation
            relative_delta = pred_delta - pred_delta[0]
            pred_surge = relative_delta + current_pivot

            rolling_preds[n_past].loc[fut_chunk.index] = pred_surge

    # Blend 24h and 48h rolling forecasts
    blended_surge = (rolling_preds[24] * 0.5) + (rolling_preds[48] * 0.5)

    m_slice = pd.DataFrame({
        'obs_tide': eval_window['etss_tide'],
        'obs_water_level': eval_window['water_level'],
        'obs_surge': eval_window['surge_residual'],
        'etss_surge': eval_window['etss_surge_guidance'],
        'etss_water_level': eval_window['etss_water_level_baseline'],
        'ai_surge': blended_surge,
        'ai_wl': blended_surge + eval_window['etss_tide']
    }, index=eval_window.index).dropna()

    # --- 5. Plotting Layer ---
    print(f"🎨 Rendering continuous 12-hour rolling verification timeline for {STATION_NAME}...")

    obs_surge = m_slice['obs_surge'].values
    ai_surge = m_slice['ai_surge'].values
    etss_surge = m_slice['etss_surge'].values

    rmse_ai_s = np.sqrt(np.mean((ai_surge - obs_surge) ** 2))
    rmse_etss_s = np.sqrt(np.mean((etss_surge - obs_surge) ** 2))

    obs_wl = m_slice['obs_water_level'].values
    ai_wl = m_slice['ai_wl'].values
    etss_wl = m_slice['etss_water_level'].values

    rmse_ai_w = np.sqrt(np.mean((ai_wl - obs_wl) ** 2))
    rmse_etss_w = np.sqrt(np.mean((etss_wl - obs_wl) ** 2))

    bias_ai_w = np.mean(ai_wl - obs_wl)
    bias_etss_w = np.mean(etss_wl - obs_wl)

    skill_wl = ((rmse_etss_w - rmse_ai_w) / rmse_etss_w) * 100.0 if rmse_etss_w > 0 else 0.0

    fig = plt.figure(figsize=(21, 12))
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[0.75, 0.25], hspace=0.15, wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_stat = fig.add_subplot(gs[:, 1])
    ax_stat.axis('off')

    # PANEL 1: Total Water Level Space
    ax1.plot(m_slice.index, m_slice['obs_water_level'], label='Observed Total Water (Truth)', color='#2c3e50', linewidth=2.0, zorder=4)
    ax1.plot(m_slice.index, m_slice['ai_wl'], label='AI Stitched Chrono Rolling Forecast', color='#e67e22', linewidth=2.0, zorder=3)
    ax1.plot(m_slice.index, m_slice['etss_water_level'], label='ETSS Physics Baseline WL', color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)
    ax1.plot(m_slice.index, m_slice['obs_tide'], label='Astronomical Tide Baseline', color='#7f8c8d', linestyle='-', linewidth=0.8, alpha=0.25, zorder=1)

    ax1.set_title(f"Deep Learning Storm Surge Forecast Suite (Unified CHRONO GFS+ETSS Pivot - 12h Horizon): {STATION_NAME} (January 2026)", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

    # PANEL 2: Cleaned Meteorological Surge Residual Space
    ax2.plot(m_slice.index, m_slice['obs_surge'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.0, zorder=4)
    ax2.plot(m_slice.index, m_slice['ai_surge'], label='AI Stitched Chrono Rolling Surge', color='#e67e22', linewidth=2.0, zorder=3)
    ax2.plot(m_slice.index, m_slice['etss_surge'], label='ETSS Physics Baseline Model', color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.6, zorder=2)

    ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
    ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
    ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

    fig.autofmt_xdate()

    # SIDE PANEL: ROLLOUT STREAM CARD
    stat_box_text = (
        f"    ROLLOUT STREAM CARD\n"
        f"=========================\n"
        f"  Month: 01/2026\n"
        f"  Station Target: {STATION_ID}\n"
        f"  Surrogate Mode: GFS+ETSS\n"
        f"  Forced Scaler: ROBUST\n\n"
        f"  [1] TOTAL WATER LEVEL (WL)\n"
        f"  -----------------------\n"
        f"  * AI 12H ROLLOUT:\n"
        f"      > RMSE : {rmse_ai_w:.3f} ft\n"
        f"      > Bias : {bias_ai_w:+.3f} ft\n"
        f"  * ETSS PHYSICS BASELINE:\n"
        f"      > RMSE : {rmse_etss_w:.3f} ft\n"
        f"      > Bias : {bias_etss_w:+.3f} ft\n\n"
        f"  [2] PURE SURGE RESIDUAL\n"
        f"  -----------------------\n"
        f"  * AI 12H ROLLOUT:\n"
        f"      > RMSE : {rmse_ai_s:.3f} ft\n"
        f"  * ETSS PHYSICS BASELINE:\n"
        f"      > RMSE : {rmse_etss_s:.3f} ft\n\n"
        f"  OPERATIONAL PERFORMANCE\n"
        f"  -----------------------\n"
        f"  AI WL Error Reduction:\n"
        f"      * {skill_wl:+.1f}% Improvement"
    )

    ax_stat.text(0.05, 0.95, stat_box_text, transform=ax_stat.transAxes,
                 fontsize=9.5, verticalalignment='top', fontname='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#eceff1', linewidth=1.0))

    plot_out = os.path.join(OUTPUT_DIR, f"gfs_etss_chrono_rolling_forecast_12h_{STATION_ID}_2026_01.png")
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    print(f"      💾 Continuous rolling evaluation chart successfully exported ➔ {plot_out}")

    plt.close(fig)
    fig.clear()
    ax1.clear()
    ax2.clear()
    ax_stat.clear()

    K.clear_session()
    gc.collect()

print("\n🎉 Monthly 12-Hour Rolling Evaluation Protocol Complete Across All Targets.")
