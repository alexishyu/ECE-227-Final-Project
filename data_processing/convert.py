import pandas as pd

# read space‑delimited file (no header)
df = pd.read_csv('data/facebook_combined.txt',
                 sep='\s+',      # one or more spaces/tabs
                 header=None,    # no existing header row
                 names=['Source','Target'])

# write out as CSV (with header)
df.to_csv('data/facebook_combined.csv', index=False)
