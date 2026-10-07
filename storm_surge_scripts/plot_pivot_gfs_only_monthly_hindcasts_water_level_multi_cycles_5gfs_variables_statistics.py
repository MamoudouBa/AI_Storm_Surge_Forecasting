#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
'''
Script developed with the assistance of the Gemini AI code assistant.
Validation & Monthly Plotting Engine for 5 GFS/Obs Met + Surge Track (6 Features total)
'''

import os, sys, gc
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

TARGET_SCALER = "ROBUST"
EVAL_START = "2026-01-01 01:00:00"
EVAL_END = "2026-03-01 17:00:00"

MODELS_PATH = '../storm_surge/'
CONFIG_FILE = '../storm_surge/stations_config_gulf_atlantic.csv'
OUTPUT_DIR = '/contrib/Mamoudou.Ba/storm_surge/verification_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_FUTURE_TRAINED = 12
N_FUTURE_OP = 96

def load_and_parse(file_path):
    df = pd.read_csv(file_path)
    t_col = 'timestamp_gui' if 'timestamp_gui' in df.columns else 'timestamp'
    df['dt'] = pd.to_datetime(df[t_col], errors='coerce', format='mixed')
    if df['dt'].dt.tz is not None:
        df['dt'] = df['dt'].dt.tz_localize(None)
    df['dt'] = df['dt'].dt.round('h')
    return df.dropna(subset=['dt'])

if not os.path.exists(CONFIG_FILE):
    print(f"❌ Error: Master station configuration matrix missing at: {CONFIG_FILE}")
    sys.exit(1)

print(f"📖 Parsing master CSV registry tracks from {CONFIG_FILE}...")
df_config = pd.read_csv(CONFIG_FILE, dtype={'station_id': str})
df_config['station_id'] = df_config['station_id'].str.strip()
df_config['station_name'] = df_config['station_name'].str.strip().str.replace('"', '').str.replace(',', '').str.replace(' ', '_')

for idx_stn, row in df_config.iterrows():
    STATION_ID = str(row['station_id']).strip()
    STATION_NAME = str(row['station_name']).strip()
    if STATION_ID != "8726520": continue

    df_g_path = os.path.join(MODELS_PATH, f'test_gfs_etss_data_{STATION_ID}_2026.csv')
    df_o_path = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2021_2026.csv')
    if not os.path.exists(df_o_path):
        df_o_path = os.path.join(MODELS_PATH, f'obs_{STATION_ID}_2020_2026.csv')

    if not (os.path.exists(df_g_path) and os.path.exists(df_o_path)):
        print(f"⚠️️ [Skipping] Data files absent for station: {STATION_NAME} ({STATION_ID})")
        continue

    print("\n" + "="*80)
    print(f"🎨 COMPILING BASELINE STITCHED PIVOT PLOTS FOR: {STATION_NAME} ({STATION_ID})")
    print("="*80)

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

        data_full['water_level'] = df_o['water_level'] if 'water_level' in df_o.columns else np.nan
        raw_surge = data_full['water_level'] - data_full['etss_tide']

        data_full['surge_residual'] = pd.Series(raw_surge.values, index=raw_surge.index)\
                                            .rolling(window=12, center=True, min_periods=1).mean()

        if 'etss_water_level' in df_g_unique.columns:
            data_full['etss_water_level_baseline'] = df_g_unique['etss_water_level']
        else:
            data_full['etss_water_level_baseline'] = data_full['etss_tide']

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
        print(f"❌ Plotter Data Alignment Exception on {STATION_NAME}: {e}")
        continue

    df_g_stitched = df_g[df_g['cycle'] == (df_g['dt'].dt.hour // 6) * 6].copy()
    available_cycles = sorted([pd.to_datetime(x) for x in df_g_stitched[(df_g_stitched['dt'] >= pd.to_datetime(EVAL_START)) & (df_g_stitched['dt'] <= pd.to_datetime(EVAL_END))]['dt'].unique()])
    if not available_cycles:
        print(f"⚠️ No forecast validation anchor cycles found within specified window for {STATION_NAME}.")
        continue

    cached_models = {}
    cached_scalers = {}
    lookbacks = [24, 48]
    logics = ["PIVOT"]

    load_failure = False
    for l_type in logics:
        for n_past in lookbacks:
            key = f"{l_type}_{n_past}_{TARGET_SCALER}"
            m_name = f"{STATION_NAME}_{N_FUTURE_TRAINED}_GFS_ETSS_PIVOT_SAMPLE_WEIGHTING_ROBUST_{n_past}past_CHRONO"
            s_f = f"{STATION_NAME}_scaler_gfs_etss_pivot_sample_weighting_robust_{n_past}h_CHRONO.joblib"
            s_t = f"{STATION_NAME}_scaler_gfs_etss_pivot_target_sample_weighting_robust_{n_past}h_CHRONO.joblib"

            try:
                with open(os.path.join(MODELS_PATH, f"{m_name}.json"), 'r') as f:
                    model = model_from_json(f.read())
                model.load_weights(os.path.join(MODELS_PATH, f"{m_name}.h5"))

                cached_models[key] = model
                cached_scalers[f"{key}_f"] = joblib.load(os.path.join(MODELS_PATH, s_f))
                cached_scalers[f"{key}_t"] = joblib.load(os.path.join(MODELS_PATH, s_t))
                print(f"    > Loaded Stitched Chrono Direct Configuration Matrix: {key}")
            except Exception as model_err:
                print(f"❌ Missing deep learning CHRONO PIVOT components on key {key}: {model_err}")
                load_failure = True
                break
        if load_failure: break
    if load_failure: continue

    # 🌟 EXACT MATCH: 5 METEOROLOGICAL FEATURES (+ 1 SURGE TRACK = 6 TOTAL COLUMNS)
    OBS_FEAT = ['air_temp', 'air_pressure', 'wind_speed', 'wind_direction', 'wind_gust']
    GUI_FEAT = ['air_temp_gui', 'air_pressure_gui', 'wind_speed_gui', 'wind_direction_gui', 'wind_gust_gui']
    TARGET = 'surge_residual'

    def generate_isolated_engine_blend(logic_type, fc_origin, op_gui):
        local_horizons = {}
        target_len = len(op_gui)

        for n_past in lookbacks:
            key = f"{logic_type.upper()}_{n_past}_{TARGET_SCALER}"
            model = cached_models[key]
            sc_f = cached_scalers[f"{key}_f"]
            sc_t = cached_scalers[f"{key}_t"]

            # OBS_FEAT (5) + TARGET (1) = 6 columns
            raw_past = data_full.loc[:fc_origin][OBS_FEAT + [TARGET]].copy().values
            past_block = raw_past[-n_past:] if len(raw_past) >= n_past else np.vstack([np.repeat(raw_past[:1, :], n_past - len(raw_past), axis=0), raw_past])

            full_pred_surge = []

            for step_start in range(0, target_len, N_FUTURE_TRAINED):
                step_end = min(step_start + N_FUTURE_TRAINED, target_len)
                chunk_len = step_end - step_start

                gui_chunk = op_gui.iloc[step_start:step_end]

                fut_met_profile = np.hstack([
                    gui_chunk['air_temp_gui'].values.reshape(-1, 1),
                    gui_chunk['air_pressure_gui'].values.reshape(-1, 1),
                    gui_chunk['wind_speed_gui'].values.reshape(-1, 1),
                    gui_chunk['wind_direction_gui'].values.reshape(-1, 1),
                    gui_chunk['wind_gust_gui'].values.reshape(-1, 1)
                ])

                if chunk_len < N_FUTURE_TRAINED:
                    pad_size = N_FUTURE_TRAINED - chunk_len
                    pad_met = np.repeat(fut_met_profile[-1:, :], pad_size, axis=0)
                    fut_met_profile = np.vstack([fut_met_profile, pad_met])
                    fb_len = N_FUTURE_TRAINED
                else:
                    fb_len = N_FUTURE_TRAINED

                fut_fb = np.zeros((fb_len, 1))
                # fut_met_profile (5) + fut_fb (1) = 6 columns
                op_fut_block = np.hstack([fut_met_profile, fut_fb])

                inp = np.ascontiguousarray(np.vstack([past_block, op_fut_block]))
                scaled_inp = sc_f.transform(inp)

                inp_tensor = tf.convert_to_tensor(scaled_inp.reshape(1, n_past + N_FUTURE_TRAINED, -1), dtype=tf.float32)
                raw_pred = model(inp_tensor, training=False).numpy().flatten()
                pred_final = sc_t.inverse_transform(raw_pred.reshape(-1, 1)).flatten()

                pred_surge = pred_final
                pred_surge_trimmed = pred_surge[:chunk_len]
                full_pred_surge.extend(pred_surge_trimmed)

                chunk_to_append = np.hstack([fut_met_profile[:chunk_len], pred_surge_trimmed.reshape(-1, 1)])
                past_block = np.vstack([past_block, chunk_to_append])[-n_past:]

            local_horizons[n_past] = np.array(full_pred_surge)[:target_len]

        timeline = np.arange(target_len)
        weight_blend = 1.0 / (1.0 + np.exp((timeline - (target_len / 2)) / 2))
        return (local_horizons[24] * weight_blend) + (local_horizons[48] * (1.0 - weight_blend))

    time_axis_records = []

    for idx, fc_origin in enumerate(available_cycles):
        if idx % 6 != 0: continue

        try:
            current_cycle = int((fc_origin.hour // 6) * 6)
            op_gui = df_g[(df_g['dt'] >= fc_origin) & (df_g['cycle'] == current_cycle)].copy().set_index('dt').sort_index().iloc[:N_FUTURE_OP]
            if len(op_gui) < N_FUTURE_OP: continue

            target_len = len(op_gui)

            obs_raw = df_o.reindex(op_gui.index)
            tide_vals = df_g_unique.reindex(op_gui.index)['etss_tide'].fillna(0.0).values.flatten()[:target_len]

            obs_surge_clean = (obs_raw['water_level'].values - tide_vals) if 'water_level' in obs_raw.columns else np.zeros(target_len)
            obs_surge_filtered = pd.Series(obs_surge_clean).rolling(window=12, center=True, min_periods=1).mean().values

            piv_track = generate_isolated_engine_blend("PIVOT", fc_origin, op_gui).flatten()[:target_len]

            time_axis_records.append(pd.DataFrame({
                'obs_tide': tide_vals,
                'obs_water_level': obs_raw['water_level'].values.flatten()[:target_len] if 'water_level' in obs_raw.columns else tide_vals,
                'obs_surge': obs_surge_filtered,
                'etss_surge': (data_full.loc[op_gui.index, 'etss_water_level_baseline'] - tide_vals).values.flatten()[:target_len],
                'etss_water_level': data_full.loc[op_gui.index, 'etss_water_level_baseline'].values.flatten()[:target_len],
                'pivot_surge': piv_track,
                'direct_surge': piv_track,
                'consensus_surge': piv_track
            }, index=op_gui.index[:target_len]))
        except Exception as loop_err:
            print(f"❌ Core processing loop skip on item {fc_origin}: {loop_err}")
            continue

    if not time_axis_records:
        print(f"⚠️ Warning: Verification vector stack completely empty for {STATION_NAME}. No matching times.")
        continue

    master_plot_df = pd.concat(time_axis_records).groupby(level=0).mean()
    master_plot_df['year_month'] = master_plot_df.index.to_period('M')

    master_plot_df['pivot_wl'] = master_plot_df['pivot_surge'] + master_plot_df['obs_tide']
    master_plot_df['direct_wl'] = master_plot_df['direct_surge'] + master_plot_df['obs_tide']
    master_plot_df['consensus_wl'] = master_plot_df['consensus_surge'] + master_plot_df['obs_tide']

    print(f"    📊 Data Validation Audit Matrix for {STATION_NAME}:")
    print(f"      > Total Extracted Rows    : {master_plot_df.shape[0]}")
    print(f"      > Timestamp Coverage Span : {master_plot_df.index.min()} to {master_plot_df.index.max()}")
    sys.stdout.flush()

    for current_month in master_plot_df['year_month'].unique():
        m_slice = master_plot_df[master_plot_df['year_month'] == current_month].dropna()

        if len(m_slice) < 1:
            print(f"    ⚠️ [Skipping Month {current_month}] Zero valid records available after dropping NaN rows.")
            continue

        print(f"    🎨 Rendering chronological chart for tracking window: {current_month} ({len(m_slice)} valid hours)")
        sys.stdout.flush()

        month_str = current_month.strftime('%B %Y')

        obs_surge = m_slice['obs_surge'].values
        piv_surge = m_slice['pivot_surge'].values
        etss_surge = m_slice['etss_surge'].values

        rmse_piv_s = np.sqrt(np.mean((piv_surge - obs_surge) ** 2))
        rmse_etss_s = np.sqrt(np.mean((etss_surge - obs_surge) ** 2))

        obs_wl = m_slice['obs_water_level'].values
        piv_wl = m_slice['pivot_wl'].values
        etss_wl = m_slice['etss_water_level'].values

        rmse_piv_w = np.sqrt(np.mean((piv_wl - obs_wl) ** 2))
        rmse_etss_w = np.sqrt(np.mean((etss_wl - obs_wl) ** 2))

        bias_piv_w = np.mean(piv_wl - obs_wl)
        bias_etss_w = np.mean(etss_wl - obs_wl)

        skill_wl = ((rmse_etss_w - rmse_piv_w) / rmse_etss_w) * 100.0 if rmse_etss_w > 0 else 0.0

        fig = plt.figure(figsize=(21, 12))
        gs = matplotlib.gridspec.GridSpec(2, 2, width_ratios=[0.75, 0.25], hspace=0.15, wspace=0.08)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
        ax_stat = fig.add_subplot(gs[:, 1])
        ax_stat.axis('off')

        ax1.plot(m_slice.index, m_slice['obs_water_level'], label='Observed Total Water (Truth)', color='#2c3e50', linewidth=2.5)
        ax1.plot(m_slice.index, m_slice['pivot_wl'], label='AI Stitched Chrono Rolling Direct Forecast', color='#8e44ad', linewidth=2.0)
        ax1.plot(m_slice.index, m_slice['etss_water_level'], label='ETSS Physics Baseline WL', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)
        ax1.plot(m_slice.index, m_slice['obs_tide'], label='Astronomical Tide Baseline', color='#7f8c8d', linestyle='-', linewidth=1.0, alpha=0.3)

        ax1.set_title(f"Deep Learning Storm Surge Forecast Suite (Pure GFS Weather Surrogate - {N_FUTURE_TRAINED}h Block Autoregressive Direct Rollout): {STATION_NAME} ({month_str})", fontsize=13, fontweight='bold', pad=12)
        ax1.set_ylabel("Total Water Level Scale (ft)", fontsize=10, labelpad=10)
        ax1.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
        ax1.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

        ax2.plot(m_slice.index, m_slice['obs_surge'], label='Pristine Observed Surge (Filtered)', color='#2c3e50', linewidth=2.5)
        ax2.plot(m_slice.index, m_slice['pivot_surge'], label='AI Stitched Chrono Rolling Direct Surge', color='#8e44ad', linewidth=2.0)
        ax2.plot(m_slice.index, m_slice['etss_surge'], label='ETSS Physics Baseline Model', color='#e74c3c', linestyle='--', linewidth=1.2, alpha=0.7)

        ax2.set_xlabel("Timeline Partition (Hourly Increments)", fontsize=10, labelpad=10)
        ax2.set_ylabel("Surge Elevation Scale (ft)", fontsize=10, labelpad=10)
        ax2.grid(True, linestyle=':', alpha=0.5, color='#bdc3c7')
        ax2.legend(loc='upper right', frameon=True, facecolor='#ffffff', edgecolor='#eceff1', fontsize=9, ncol=2)

        fig.autofmt_xdate()

        stat_box_text = (
            f"    PIVOT ROLLOUT CARD\n"
            f"=========================\n"
            f"  Month: {current_month.strftime('%m/%Y')}\n"
            f"  Station Target: {STATION_ID}\n"
            f"  Surrogate Mode: GFS-ONLY\n"
            f"  Forced Scaler: {TARGET_SCALER}\n\n"
            f"  [1] TOTAL WATER LEVEL (WL)\n"
            f"  -----------------------\n"
            f"  * AI {N_FUTURE_TRAINED}H PIVOT ROLLOUT:\n"
            f"    > RMSE : {rmse_piv_w:.3f} ft\n"
            f"    > Bias : {bias_piv_w:+.3f} ft\n"
            f"  * ETSS PHYSICS BASELINE:\n"
            f"    > RMSE : {rmse_etss_w:.3f} ft\n"
            f"    > Bias : {bias_etss_w:+.3f} ft\n\n"
            f"  [2] PURE SURGE RESIDUAL\n"
            f"  -----------------------\n"
            f"  * AI {N_FUTURE_TRAINED}H PIVOT ROLLOUT:\n"
            f"    > RMSE : {rmse_piv_s:.3f} ft\n"
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

        plot_out = os.path.join(OUTPUT_DIR, f"gfs_only_chrono_direct_rolling_forecast_{N_FUTURE_TRAINED}h_{STATION_ID}_{current_month.strftime('%Y_%m')}.png")
        plt.savefig(plot_out, dpi=300, bbox_inches='tight')
        print(f"    💾 Operational Direct rolling chart successfully exported ➔ {plot_out}")

        plt.close(fig)
        fig.clear()

    K.clear_session()
    gc.collect()

print("\n🎉 Isolated Multi-Cycle Direct Baseline Model Plot Engine Complete.")
