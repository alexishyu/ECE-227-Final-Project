import pandas as pd

df = pd.read_csv('data/facebook_features.csv')
# df = pd.read_csv('data/gplus_features.csv')
# 1. Count non-null (i.e. not None/NaN) per column
non_null_counts = df.count()

# 2. Count unique non-null values per column
unique_counts = df.nunique(dropna=True)


# 3. Assemble into one table
summary = pd.DataFrame({
    'non_null_count': non_null_counts,
    'unique_count':   unique_counts
})

# 4. Sort by non_null_count descending (or swap to sort by unique_count)
summary_sorted = summary.sort_values(by='non_null_count', ascending=False)

print(summary_sorted)