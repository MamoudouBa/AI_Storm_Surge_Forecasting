#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import subprocess
import pandas as pd
import os

# --- Configuration ---
# Use the NEW full US grid you just created
gfs_reference = 'gfs_reference_full_us.grib2'
config_in = 'stations_config.csv'
config_out = 'stations_config_with_elev.csv'

if not os.path.exists(gfs_reference):
    print(f"Error: {gfs_reference} not found. Please check the filename.")
    exit()

# Load original config
df = pd.read_csv(config_in)
elev_values = []

print(f"--- Extracting Elevations from {gfs_reference} ---")

for index, row in df.iterrows():
    name = row['station_name']
    lat = row['latitude']
    lon = row['longitude']
    
    # Convert negative longitude to 0-360 for GFS
    lon_360 = lon + 360 if lon < 0 else lon
    
    # Run wgrib2 for THIS SPECIFIC station
    cmd = [
        "wgrib2", gfs_reference, 
        "-match", ":HGT:surface:",
        "-lon", str(lon_360), str(lat)
    ]
    
    # Capture the output
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    
    if "val=" in output:
        # Extract numerical value after 'val='
        val_str = output.split("val=")[-1]
        val = float(val_str)
        
        # Check for '9.999e+20' (Missing/Out of Bounds)
        if val > 1e10:
            print(f"⚠️  {name}: Out of bounds (set to 0.0)")
            elev_values.append(0.0)
        else:
            print(f"✅ {name}: {val}m")
            elev_values.append(val)
    else:
        print(f"❌ {name}: No data found at {lat}, {lon_360}")
        elev_values.append(0.0)
