import json
import sys

import numpy as np
import pandas as pd

# Builds constellation_lines.csv (constellation, HR1, HR2) from d3-celestial's
# constellations.lines.json (BSD-3, see THIRD_PARTY_NOTICES.md):
# https://raw.githubusercontent.com/ofrohn/d3-celestial/master/data/constellations.lines.json
# Usage: python constellation_processing.py path/to/constellations.lines.json

# Line endpoints are given as RA/Dec, so each one is matched to the nearest star in bright_stars.csv
MAX_MATCH_DEG = 0.2

json_path = sys.argv[1] if len(sys.argv) > 1 else "constellations.lines.json"
with open(json_path, "r", encoding="utf-8") as file:
    features = json.load(file)["features"]

stars = pd.read_csv("bright_stars.csv")
star_vectors = stars[["x", "y", "z"]].values
star_hr = stars["HR"].values


def radec_to_vector(ra_deg, dec_deg):
    # d3-celestial stores RA in [-180, 180]; cos/sin handle that range directly
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])


def nearest_star(ra_deg, dec_deg):
    cos_sep = star_vectors @ radec_to_vector(ra_deg, dec_deg)
    i = np.argmax(cos_sep)
    return star_hr[i], np.degrees(np.arccos(np.clip(cos_sep[i], -1.0, 1.0)))


segments = set()
unmatched = []
match_distances = []
for feature in features:
    constellation = feature["id"]
    for line in feature["geometry"]["coordinates"]:
        hrs = []
        for ra, dec in line:
            hr, dist = nearest_star(ra, dec)
            if dist > MAX_MATCH_DEG:
                unmatched.append((constellation, ra, dec, dist))
                hr = None
            else:
                match_distances.append(dist)
            hrs.append(hr)

        for hr1, hr2 in zip(hrs[:-1], hrs[1:]):
            if hr1 is None or hr2 is None or hr1 == hr2:
                continue
            # Store each segment once, regardless of direction
            segments.add((constellation, min(hr1, hr2), max(hr1, hr2)))

print(f"Matched {len(match_distances)} endpoints, max distance {max(match_distances):.4f} deg")
print(f"Unmatched endpoints: {len(unmatched)}")
for constellation, ra, dec, dist in unmatched:
    print(f"  {constellation}: RA {ra:.4f} DEC {dec:.4f} (nearest star {dist:.3f} deg away)")

df = pd.DataFrame(sorted(segments), columns=["constellation", "HR1", "HR2"])
df.to_csv("constellation_lines.csv", index=False)
print(f"Wrote {len(df)} segments across {df['constellation'].nunique()} constellations")
