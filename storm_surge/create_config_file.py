#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import subprocess
import pandas as pd

# Use the RAW GFS file (the one your colleague used)
gfs_raw = 'gfs.t00z.sfluxgrbf014.grib2'
config_in = 'stations_config.csv'
config_out = 'stations_config_with_elev.csv'

df = pd.read_csv(config_in)
elev_values = []

print(f"--- Extracting elevations from RAW GFS: {gfs_raw} ---")

for index, row in df.iterrows():
    # Convert West to 0-360
    lon_360 = row['longitude'] + 360 if row['longitude'] < 0 else row['longitude']
    lat = row['latitude']
    
    # Query ONLY the HGT message (usually message 1 or 2 in sflux)
    # We use -match to ensure we only get the elevation height
    cmd = ["wgrib2", gfs_raw, "-match", ":HGT:surface:", "-lon", str(lon_360), str(lat)]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if "val=" in result.stdout:
        val = float(result.stdout.split("val=")[-1].strip())
        print(f"✅ {row['station_name']}: {val}m")
        elev_values.append(val)
    else:
        print(f"❌ {row['station_name']}: Lookup failed.")
        elev_values.append(0.0)

df['model_elevation_m'] = elev_values
df.to_csv(config_out, index=False)
print(f"\nSaved updated config to {config_out}")
