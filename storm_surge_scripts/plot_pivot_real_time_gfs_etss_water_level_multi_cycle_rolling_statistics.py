#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Script developed with the assistance of the Gemini AI code assistant.
Operational GFS + ETSS Hybrid Pivot Multi-Station Real-Time Prediction Engine.
Evaluates multi-cycle unified CHRONO models across configured stations using real-time streams.
Displays a 12-hour stitched rolling hindcast across 102 hours prior to t0, alongside an extended
forward operational forecast out to N_FUTURE_OP hours after t0. Draws a vertical partition line at t0.
Dynamic Inverse-Variance Weighting Integration: Dynamically blends 24h & 48h lookback ensemble members.
Hindcast Evaluation Panel: Calculates and displays historical AI vs ETSS performance statistics in the sidebar.
'''

import os, sys, gc, warnings
# 🌟 FORCE KERAS 2 DESERIALIZER (Leaves NumPy 1.26.4 untouched)
os.environ["TF_USE_LEGACY_KERAS"] = "1"
warnings.filterwarnings('ignore')

# 🌟 SUPPRESS ALL TENSORFLOW & C++ COMMAND LINE NOISE BEFORE IMPORTING TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Clean CPU execution
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_NUM_INTRAOP_THREADS'] = '1'
os.environ['TF_NUM_INTEROP_THREADS'] = '1'

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

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
TARGET_SCALER = "ROBUST"
MODELS_PATH = '../storm_surge/'
CONFIG_FILE = '../storm_surge/stations_config_gulf_atlantic.csv'
OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 🌟 CONFIGURATION PARAMETERS
N_FUTURE_TRAINED = 12   # Trained step horizon: 12h
N_FUTURE_OP = 12        # Total operational forecast rollout: 102h
HINDCAST_HOURS = 12     # Rolling historical hindcast window prior to t0
HINDCAST_STEP = 12       # 12-hour rolling step cadence for historical stitching
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

# --- 🚀 MASTER PLOTTER OUTER LOOP ACROSS ALL STATIONS ---
for idx_stn, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    STATION_NAME = str(row['station_name']).strip()

    df_g_path = os.path.join(MODELS_PATH, f'realtime_gfs_etss_data_{STATION_ID}_2026.csv')
    df_o_path = os.path.join(MODELS_PATH, f'realtime_obs_{STATION_ID}_2026.csv')

    if not (os.path.exists(df_g_path) and os.path.exists(df_o_path)):
        print(f"⚠️ [Skipping] Real-time data files absent for station: {STATION_NAME} ({STATION_ID})")
        continue

    print("\n" + "="*80)
    print(f"🎨 COMPILING REAL-TIME HYBRID GFS+ETSS {N_FUTURE_OP}H FORECAST FOR: {STATION_NAME} ({STATION_ID})")
    print(f"   > Configuration: Trained Horizon = {N_FUTURE_TRAINED}h | Operational Horizon = {N_FUTURE_OP}h")
    print("="*80)

    # --- 2. Load, Standardize, and Low-Pass Filter Input Files ---
    try:
        df_g = load_and_parse(df_g_path)
        df_o = load_and_parse(df_o_path)

        df_o = df_o.drop_duplicates(subset=['dt']).set_index('dt').sort_index()
        df_g_unique = df_g.sort_values('cycle').drop_duplicates(subset=['dt'], keep='last').set_index('dt').sort_index()

        # Identify forecast origin t0 as the latest available ground observation
        t0 = df_o.index.max()
        print(f"📍 Forecast Origin Anchor (t0): {t0}")

        hindcast_start = t0 - pd.Timedelta(hours=HINDCAST_HOURS)

        data_full = pd.DataFrame(index=df_g_unique.index.union(df_o.index)).sort_index()

        # Extract Tide Baseline
        if 'etss_tide' in df_g_unique.columns:
            data_full['etss_tide'] = df_g_unique['etss_tide']
        else:
            t_col = [c for c in df_g_unique.columns if 'tide' in c.lower()]
            data_full['etss_tide'] = df_g_unique[t_col[0]] if t_col else 0.0

        # Extract Observed Water Level & Compute Filtered Surge Residual
        data_full['water_level'] = df_o['water_level']
        raw_surge = data_full['water_level'] - data_full['etss_tide']
        data_full['surge_residual'] = pd.Series(raw_surge.values, index=raw_surge.index)\
                                            .rolling(window=12, center=True, min_periods=1).mean()

        # Robust ETSS Water Level Baseline Resolver
        if 'water_level_etss' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['water_level_etss']
        elif 'etss_water_level' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['etss_water_level']
        else:
            wl_cols = [c for c in df_g_unique.columns if 'etss' in c.lower() and ('water' in c.lower() or 'wl' in c.lower())]
            data_full['etss_water_level_baseline'] = df_g_unique[wl_cols[0]] if wl_cols else data_full['etss_tide']

        # Forward/Back fill prior to guidance computation
        data_full['etss_tide'] = data_full['etss_tide'].ffill().bfill()
        data_full['etss_water_level_baseline'] = data_full['etss_water_level_baseline'].ffill().bfill()

        raw_etss_surge = data_full['etss_water_level_baseline'] - data_full['etss_tide']
        data_full['etss_surge_guidance'] = pd.Series(raw_etss_surge.values, index=raw_etss_surge.index)\
                                                  .rolling(window=12, center=True, min_periods=1).mean()
        data_full['etss_water_level_baseline'] = data_full['etss_surge_guidance'] + data_full['etss_tide']

        # Extract GFS & Observed Meteorological Fields
        for col in ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust',
                    'air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']:
            if col in df_g_unique.columns:
                data_full[col] = df_g_unique[col]
            elif col in df_o.columns:
                data_full[col] = df_o[col]

        if 'wind_gust_gui' not in data_full.columns and 'wind_speed_gui' in data_full.columns:
            data_full['wind_gust_gui'] = data_full['wind_speed_gui'] * 1.2
        if 'wind_gust' not in data_full.columns and 'wind_speed' in data_full.columns:
            data_full['wind_gust'] = data_full['wind_speed'] * 1.2

        data_full = data_full.ffill().bfill()
    except Exception as e:
        print(f"❌ Real-time data alignment exception on {STATION_NAME}: {e}")
        continue

    # --- 3. LOAD UNIFIED CHRONO MODEL ASSETS ---
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

    # --- 4. HYBRID AUTOREGRESSIVE INFERENCE ENGINE WITH DYNAMIC INVERSE-VARIANCE WEIGHTING ---
    OBS_FEAT = ['wind_direction', 'wind_gust', 'etss_surge_guidance', 'air_temp']
    GUI_FEAT = ['wind_direction_gui', 'wind_gust_gui', 'etss_surge_guidance', 'air_temp_gui']
    TARGET = 'surge_residual'

    def run_inference_window(anchor_time, input_df, max_horizon):
        valid_anchor_time = data_full.index.asof(anchor_time)
        if pd.isna(valid_anchor_time):
            valid_anchor_time = data_full.index[0]

        anchor_surge = data_full.loc[valid_anchor_time, 'surge_residual']
        if isinstance(anchor_surge, pd.Series):
            anchor_surge = anchor_surge.iloc[0]

        target_len = min(max_horizon, len(input_df))
        local_horizons = {}
        block_variances = {24: [], 48: []}

        for n_past in lookbacks:
            key = f"CHRONO_{n_past}"
            model = cached_models[key]
            sc_f = cached_scalers[f"{key}_f"]
            sc_t = cached_scalers[f"{key}_t"]

            raw_past = data_full.loc[:valid_anchor_time][OBS_FEAT + [TARGET]].copy().values
            past_block = raw_past[-n_past:] if len(raw_past) >= n_past else np.vstack([np.repeat(raw_past[:1, :], n_past - len(raw_past), axis=0), raw_past])

            current_pivot = anchor_surge
            full_pred_surge = []

            for step_start in range(0, target_len, N_FUTURE_TRAINED):
                step_end = min(step_start + N_FUTURE_TRAINED, target_len)
                chunk_len = step_end - step_start

                gui_chunk = input_df.iloc[step_start:step_end]

                fut_met_list = []
                for col in GUI_FEAT:
                    if col in gui_chunk.columns:
                        val = gui_chunk[col].values.reshape(-1, 1)
                    elif col in data_full.columns:
                        val = data_full.loc[gui_chunk.index, col].values.reshape(-1, 1)
                    else:
                        val = np.zeros((len(gui_chunk), 1))
                    fut_met_list.append(val)

                fut_met_profile = np.hstack(fut_met_list)

                if chunk_len < N_FUTURE_TRAINED:
                    pad_size = N_FUTURE_TRAINED - chunk_len
                    pad_met = np.repeat(fut_met_profile[-1:, :], pad_size, axis=0)
                    fut_met_profile = np.vstack([fut_met_profile, pad_met])
                    fb_len = N_FUTURE_TRAINED
                else:
                    fb_len = N_FUTURE_TRAINED

                last_known_surge = past_block[-1, -1]
                fut_fb = np.linspace(last_known_surge, current_pivot, fb_len).reshape(-1, 1)

                op_fut_block = np.hstack([fut_met_profile, fut_fb])

                inp = np.ascontiguousarray(np.vstack([past_block, op_fut_block]))
                scaled_inp = sc_f.transform(inp)

                # Memory-safe direct tensor invocation
                inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), dtype=tf.float32)
                raw_pred = model(inp_tensor, training=False).numpy().flatten()
                pred_delta = sc_t.inverse_transform(raw_pred.reshape(-1, 1)).flatten()

                relative_delta = pred_delta - pred_delta[0]
                relative_delta = np.clip(relative_delta, -1.5, 1.5)

                pred_surge = relative_delta + current_pivot
                pred_surge_trimmed = pred_surge[:chunk_len]
                full_pred_surge.extend(pred_surge_trimmed)

                chunk_var = np.var(pred_surge_trimmed) if len(pred_surge_trimmed) > 1 else 1e-4
                block_variances[n_past].extend([chunk_var] * chunk_len)

                current_pivot = pred_surge_trimmed[-1]

                chunk_to_append = np.hstack([fut_met_profile[:chunk_len], pred_surge_trimmed.reshape(-1, 1)])
                past_block = np.vstack([past_block, chunk_to_append])[-n_past:]

            local_horizons[n_past] = np.array(full_pred_surge)[:target_len].flatten()
            block_variances[n_past] = np.array(block_variances[n_past])[:target_len]

        # 🌟 DYNAMIC INVERSE-VARIANCE BLENDING ENGINE
        eps = 1e-6
        v24 = np.maximum(np.array(block_variances[24][:target_len], dtype=np.float64), eps)
        v48 = np.maximum(np.array(block_variances[48][:target_len], dtype=np.float64), eps)

        with np.errstate(divide='ignore', invalid='ignore'):
            inv_var_24 = 1.0 / v24
            inv_var_48 = 1.0 / v48
            total_inv_var = inv_var_24 + inv_var_48

            raw_w_24h = inv_var_24 / total_inv_var

        w_24h = np.nan_to_num(raw_w_24h, nan=0.5, posinf=0.5, neginf=0.5)
        w_24h = pd.Series(w_24h).rolling(window=12, min_periods=1, center=True).mean().values
        w_24h = np.clip(w_24h, 0.0, 1.0)
        w_48h = 1.0 - w_24h

        blended_track = (local_horizons[24] * w_24h) + (local_horizons[48] * w_48h)

        return blended_track.flatten(), local_horizons[24], local_horizons[48]

    # =====================================================================
    # BUILD 102H FORECAST OPERATIONAL WEATHER MATRIX (AFTER t0)
    # =====================================================================
    current_cycle = int((t0.hour // 6) * 6)
    op_gui = df_g[(df_g['dt'] >= t0) & (df_g['cycle'] == current_cycle)].copy().set_index('dt').sort_index()

    if len(op_gui) < N_FUTURE_OP:
        op_gui = df_g_unique.loc[t0 : t0 + pd.Timedelta(hours=N_FUTURE_OP - 1)].copy()

    full_dt_range = pd.date_range(start=t0, periods=N_FUTURE_OP, freq='h')
    op_gui = op_gui.reindex(full_dt_range)

    if 'etss_tide' not in op_gui.columns or op_gui['etss_tide'].isna().any():
        op_gui['etss_tide'] = data_full.reindex(full_dt_range)['etss_tide'].ffill().bfill().values

    if 'etss_surge_guidance' not in op_gui.columns or op_gui['etss_surge_guidance'].isna().any():
        op_gui['etss_surge_guidance'] = data_full.reindex(full_dt_range)['etss_surge_guidance'].ffill().bfill().values

    if 'etss_water_level_baseline' not in op_gui.columns or op_gui['etss_water_level_baseline'].isna().any():
        op_gui['etss_water_level_baseline'] = data_full.reindex(full_dt_range)['etss_water_level_baseline'].ffill().bfill().values

    for col in GUI_FEAT:
        if col not in op_gui.columns or op_gui[col].isna().any():
            op_gui[col] = data_full.reindex(full_dt_range)[col].ffill().bfill().values

    op_gui = op_gui.ffill().bfill()
    target_len = len(op_gui)

    # 1. Forward Forecast from t0 (102 Hours)
    piv_track, member_24h_track, member_48h_track = run_inference_window(t0, op_gui, N_FUTURE_OP)

    # 🌟 Smooth AI Forecast Surge Track
    smoothed_piv_track = pd.Series(piv_track, index=op_gui.index[:target_len])\
                           .rolling(window=6, center=True, min_periods=1).mean().values
    smoothed_m24_track = pd.Series(member_24h_track, index=op_gui.index[:target_len])\
                           .rolling(window=6, center=True, min_periods=1).mean().values
    smoothed_m48_track = pd.Series(member_48h_track, index=op_gui.index[:target_len])\
                           .rolling(window=6, center=True, min_periods=1).mean().values

    forecast_df = pd.DataFrame({
        'obs_tide': op_gui['etss_tide'].values[:target_len],
        'etss_surge': op_gui['etss_surge_guidance'].values[:target_len],
        'etss_water_level': op_gui['etss_water_level_baseline'].values[:target_len],
        'pivot_surge': smoothed_piv_track,
        'pivot_wl': smoothed_piv_track + op_gui['etss_tide'].values[:target_len],
        'member_24h_surge': smoothed_m24_track,
        'member_24h_wl': smoothed_m24_track + op_gui['etss_tide'].values[:target_len],
        'member_48h_surge': smoothed_m48_track,
        'member_48h_wl': smoothed_m48_track + op_gui['etss_tide'].values[:target_len]
    }, index=op_gui.index[:target_len])

    # =====================================================================
    # 2. 12-HOUR ROLLING STITCHED HINDCAST ENGINE (PRIOR TO t0)
    # =====================================================================
    hindcast_df = data_full.loc[hindcast_start : t0].copy()
    rolling_hindcast_records = []

    hindcast_anchors = pd.date_range(start=hindcast_start, end=t0 - pd.Timedelta(hours=HINDCAST_STEP), freq=f'{HINDCAST_STEP}h')

    for h_anchor in hindcast_anchors:
        h_chunk_gui = data_full.loc[h_anchor : h_anchor + pd.Timedelta(hours=HINDCAST_STEP - 1)].copy()
        if len(h_chunk_gui) < HINDCAST_STEP: continue

        for c_obs, c_gui in zip(OBS_FEAT, GUI_FEAT):
            h_chunk_gui[c_gui] = h_chunk_gui[c_obs]

        chunk_pred_surge, _, _ = run_inference_window(h_anchor, h_chunk_gui, HINDCAST_STEP)
        rolling_hindcast_records.append(pd.Series(chunk_pred_surge.flatten(), index=h_chunk_gui.index[:len(chunk_pred_surge)]))

    if rolling_hindcast_records:
        stitched_hindcast_series = pd.concat(rolling_hindcast_records)
        stitched_hindcast_series = stitched_hindcast_series[~stitched_hindcast_series.index.duplicated(keep='last')]
        raw_ai_hindcast = stitched_hindcast_series.reindex(hindcast_df.index).astype(np.float64)
    else:
        raw_ai_hindcast = hindcast_df['surge_residual'].astype(np.float64)

    # 🌟 Smooth AI Hindcast Surge Track
    hindcast_df['ai_surge_hindcast'] = raw_ai_hindcast.ffill().bfill()\
                                                     .rolling(window=6, center=True, min_periods=1).mean()
    hindcast_df['ai_wl_hindcast'] = hindcast_df['ai_surge_hindcast'] + hindcast_df['etss_tide']

    # =====================================================================
    # 🌟 3. COMPUTE HINDCAST EVALUATION PERFORMANCE METRICS (AI vs ETSS)
    # =====================================================================
    h_eval = hindcast_df.dropna(subset=['water_level', 'ai_wl_hindcast', 'etss_water_level_baseline']).copy()

    if len(h_eval) > 0:
        h_obs_wl = h_eval['water_level'].values
        h_ai_wl = h_eval['ai_wl_hindcast'].values
        h_etss_wl = h_eval['etss_water_level_baseline'].values

        h_obs_s = h_eval['surge_residual'].values
        h_ai_s = h_eval['ai_surge_hindcast'].values
        h_etss_s = h_eval['etss_surge_guidance'].values

        rmse_ai_wl_h = np.sqrt(np.mean((h_ai_wl - h_obs_wl) ** 2))
        bias_ai_wl_h = np.mean(h_ai_wl - h_obs_wl)
        rmse_etss_wl_h = np.sqrt(np.mean((h_etss_wl - h_obs_wl) ** 2))
        bias_etss_wl_h = np.mean(h_etss_wl - h_obs_wl)

        rmse_ai_s_h = np.sqrt(np.mean((h_ai_s - h_obs_s) ** 2))
        rmse_etss_s_h = np.sqrt(np.mean((h_etss_s - h_obs_s) ** 2))

        skill_wl_h = ((rmse_etss_wl_h - rmse_ai_wl_h) / rmse_etss_wl_h) * 100.0 if rmse_etss_wl_h > 0 else 0.0
    else:
        rmse_ai_wl_h = bias_ai_wl_h = rmse_etss_wl_h = bias_etss_wl_h = 0.0
        rmse_ai_s_h = rmse_etss_s_h = skill_wl_h = 0.0

    # --- 5. Plotting & Export ---
    fig = plt.figure(figsize=(21, 12))
    gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[0.75, 0.25], hspace=0.15, wspace=0.08)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    ax_stat = fig.add_subplot(gs[:, 1])
    ax_stat.axis('off')

    # PANEL 1: Total Water Level Space
    ax1.plot(hindcast_df.index, hindcast_df['water_level'], label='Observed Total Water (Truth)', color='#2c3e50', linewidth=2.5, zorder=4)
    ax1.plot(hindcast_df.index, hindcast_df['etss_water_level_baseline'], label='ETSS Physics Hindcast WL', color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.5, zorder=1)
    ax1.plot(hindcast_df.index, hindcast_df['ai_wl_hindcast'], label=f'AI Stitched Rolling Hindcast WL ({HINDCAST_STEP}h Cadence)', color='#d35400', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)

    ax1.plot(forecast_df.index, forecast_df['pivot_wl'], label='AI Unified Blend Forecast WL', color='#e67e22', linewidth=5.0, alpha=0.35, zorder=2)
    ax1.plot(forecast_df.index, forecast_df['member_24h_wl'], label='AI Member Track (24h Lookback)', color='#3498db', linestyle='--', linewidth=1.5, alpha=1.0, zorder=3)
    ax1.plot(forecast_df.index, forecast_df['member_48h_wl'], label='AI Member Track (48h Lookback)', color='#2ecc71', linestyle='--', linewidth=1.5, alpha=1.0, zorder=3)
    ax1.plot(forecast_df.index, forecast_df['etss_water_level'], label='ETSS Physics Forecast WL', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)
    ax1.plot(pd.concat([hindcast_df['etss_tide'], forecast_df['obs_tide']]).index, pd.concat([hindcast_df['etss_tide'], forecast_df['obs_tide']]), label='Astronomical Tide Baseline', color='#7f8c8d', linestyle='-', linewidth=1.0, alpha=0.25, zorder=1)

    ax1.axvline(x=t0, color='#000000', linestyle='--', linewidth=2.0, alpha=0.85)
    ax1.text(t0, ax1.get_ylim()[1] * 0.90, ' FORECAST ORIGIN (t₀)', color='#000000', fontweight='bold', fontsize=10)

    ax1.set_title(f"Real-Time Deep Learning Storm Surge Forecast Suite (Unified CHRONO GFS+ETSS Pivot - {N_FUTURE_OP}h Rollout): {STATION_NAME} (Origin: {t0.strftime('%Y-%m-%d %H:%MZ')})", fontsize=13, fontweight='bold', pad=12)
    ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
    ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
    ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

    # PANEL 2: Cleaned Meteorological Surge Residual Space
    ax2.plot(hindcast_df.index, hindcast_df['surge_residual'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.5, zorder=4)
    ax2.plot(hindcast_df.index, hindcast_df['etss_surge_guidance'], label='ETSS Physics Hindcast Surge', color='#e74c3c', linestyle=':', linewidth=1.2, alpha=0.5, zorder=1)
    ax2.plot(hindcast_df.index, hindcast_df['ai_surge_hindcast'], label=f'AI Stitched Rolling Hindcast Surge ({HINDCAST_STEP}h Cadence)', color='#d35400', linestyle='--', linewidth=1.5, alpha=0.8, zorder=3)

    ax2.plot(forecast_df.index, forecast_df['pivot_surge'], label='AI Unified Blend Surge', color='#e67e22', linewidth=5.0, alpha=0.35, zorder=2)
    ax2.plot(forecast_df.index, forecast_df['member_24h_surge'], label='AI Member Surge (24h Lookback)', color='#3498db', linestyle='--', linewidth=1.5, alpha=1.0, zorder=3)
    ax2.plot(forecast_df.index, forecast_df['member_48h_surge'], label='AI Member Surge (48h Lookback)', color='#2ecc71', linestyle='--', linewidth=1.5, alpha=1.0, zorder=3)
    ax2.plot(forecast_df.index, forecast_df['etss_surge'], label='ETSS Physics Forecast Surge', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.8, zorder=1)

    ax2.axvline(x=t0, color='#000000', linestyle='--', linewidth=2.0, alpha=0.85)

    ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
    ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
    ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
    ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

    fig.autofmt_xdate()

    # 🌟 SIDE PANEL: OPERATIONAL STREAM CARD WITH HINDCAST SCORES
    stat_box_text = (
        f"    REAL-TIME STREAM CARD\n"
        f"=========================\n"
        f"  Forecast Origin (t₀):\n"
        f"    > {t0.strftime('%m/%d %H:%MZ')}\n"
        f"  Station Target: {STATION_ID}\n"
        f"  Surrogate Mode: GFS+ETSS\n"
        f"  Model Type: CHRONO\n\n"
        f"  [1] HINDCAST WINDOW\n"
        f"  -----------------------\n"
        f"  * Rolling History: {HINDCAST_HOURS}h\n"
        f"  * Stitch Cadence: {HINDCAST_STEP}h\n"
        f"  * Anchor Surge: {data_full.loc[data_full.index.asof(t0), 'surge_residual']:+.3f} ft\n\n"
        f"  [2] STRATEGIC FORECAST\n"
        f"  -----------------------\n"
        f"  * Step Cadence: {N_FUTURE_TRAINED}h\n"
        f"  * Rollout Horizon: {N_FUTURE_OP}h\n"
        f"  * Peak Predicted Surge:\n"
        f"    > {forecast_df['pivot_surge'].max():+.3f} ft\n"
        f"  * Min Predicted Surge:\n"
        f"    > {forecast_df['pivot_surge'].min():+.3f} ft\n\n"
        f"  [3] HINDCAST SCORES ({HINDCAST_HOURS}h)\n"
        f"  -----------------------\n"
        f"  * TOTAL WATER LEVEL (WL):\n"
        f"    > AI RMSE  : {rmse_ai_wl_h:.3f} ft\n"
        f"    > AI Bias  : {bias_ai_wl_h:+.3f} ft\n"
        f"    > ETSS RMSE: {rmse_etss_wl_h:.3f} ft\n"
        f"    > ETSS Bias: {bias_etss_wl_h:+.3f} ft\n"
        f"  * PURE SURGE RESIDUAL:\n"
        f"    > AI RMSE  : {rmse_ai_s_h:.3f} ft\n"
        f"    > ETSS RMSE: {rmse_etss_s_h:.3f} ft\n"
        f"  -----------------------\n"
        f"  AI WL Error Reduction:\n"
        f"    * {skill_wl_h:+.1f}% Improvement"
    )

    ax_stat.text(0.05, 0.95, stat_box_text, transform=ax_stat.transAxes,
                 fontsize=9.0, verticalalignment='top', fontname='monospace',
                 bbox=dict(boxstyle='round,pad=0.6', facecolor='#f8f9fa', edgecolor='#eceff1', linewidth=1.0))

    plot_out = os.path.join(OUTPUT_DIR, f"gfs_etss_unified_chrono_realtime_forecast_{N_FUTURE_OP}h_{STATION_ID}_{t0.strftime('%Y%m%d_%H%M')}.png")
    plt.savefig(plot_out, dpi=300, bbox_inches='tight')
    print(f"💾 Multi-member CHRONO real-time forecast chart exported ➔ {plot_out}")

    plt.close('all')
    fig.clear()

    K.clear_session()
    gc.collect()

print("\n🎉 Unified Multi-Cycle Real-Time Prediction Protocol Complete Across All Targets.")
