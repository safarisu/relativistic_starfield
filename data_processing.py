import pandas as pd
import numpy as np

# Read the file line by line and split on whitespace
stars = []
with open("bsc5.txt", "r") as file:
    for line in file:
        # if line.strip() == '' or not line[0].isdigit():
        #     continue  # Skip header or empty lines

        parts = line.strip().split()
        if len(parts) < 8:
            continue  # Malformed line

        try:
            stars.append({
                "HR": parts[0],
                "RA": parts[1],
                "DEC": parts[2],
                "MAG": float(parts[6]),
                "SpType": parts[7]
            })
        except ValueError:
            continue  # Skip lines with invalid numbers

# Convert to DataFrame
df = pd.DataFrame(stars)

# Sort by brightness (lower magnitude is brighter) and keep top 1000
df = df.sort_values("MAG")

# Convert RA (HH:MM:SS) to degrees
def ra_to_deg(ra_str):
    h, m, s = [float(part) for part in ra_str.split(":")]
    return 15.0 * (h + m / 60.0 + s / 3600.0)

# Convert DEC (±DD:MM:SS) to degrees
def dec_to_deg(dec_str):
    sign = -1 if dec_str.strip().startswith("-") else 1
    d, m, s = [float(part) for part in dec_str.replace("+", "").replace("-", "").split(":")]
    return sign * (d + m / 60.0 + s / 3600.0)

df["RA_deg"] = df["RA"].apply(ra_to_deg)
df["DEC_deg"] = df["DEC"].apply(dec_to_deg)

# Convert spherical coordinates to unit Cartesian vector
ra_rad = np.radians(df["RA_deg"])
dec_rad = np.radians(df["DEC_deg"])

df["x"] = np.cos(dec_rad) * np.cos(ra_rad)
df["y"] = np.cos(dec_rad) * np.sin(ra_rad)
df["z"] = np.sin(dec_rad)

# Load spectral type temperature mapping
spec_map = pd.read_csv("spec_type.csv")
spec_temp_dict = {row["Spectral Type"]: row["Temperature (K)"] for _, row in spec_map.iterrows()}


# Find best match from the mapping
def match_spectral_type(sptype):
    if not isinstance(sptype, str):
        return 5500  # fallback
    sptype = sptype.strip().upper()

    for key in spec_temp_dict.keys():
        if sptype.startswith(key):
            return spec_temp_dict[key]

    # fallback to rough classes if no match found
    if sptype.startswith('O'): return 40000
    if sptype.startswith('B'): return 20000
    if sptype.startswith('A'): return 8500
    if sptype.startswith('F'): return 6500
    if sptype.startswith('G'): return 5500
    if sptype.startswith('K'): return 4000
    if sptype.startswith('M'): return 3000
    return 5000  # unknown


# Apply the refined function
df["Temp_K"] = df["SpType"].apply(match_spectral_type)

# You now have: df[["x", "y", "z", "Temp_K", "Mag"]] for simulation
df.to_csv("bright_stars2.csv", index=False)
