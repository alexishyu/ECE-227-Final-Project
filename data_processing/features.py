import os
import pandas as pd

def list_ego_ids(directory):
    """
    Scans `directory` for files named like <ego_id>.* and returns
    a sorted list of unique ego_ids.
    """
    ego_ids = set()
    for fname in os.listdir(directory):
        if '.' in fname:
            ego_id, _ = fname.split('.', 1)
            ego_ids.add(ego_id)
    return sorted(ego_ids)

def parse_featnames(path):
    """
    Reads `path` line‐by‐line, expecting lines like:
        9 education;classes;id;anonymized feature 9
    Returns a list of tuples:
        [(9, "education;classes;id", "anonymized feature 9"), …]
    """
    parsed = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # split off the column number
            col_str, rest = line.split(None, 1)
            col_num = int(col_str)
            # split at the last semicolon
            try: # for facebook data
                feature_name, value = rest.rsplit(';anonymized feature ', 1)
                parsed.append((col_num, feature_name, value))
            except: # for gplus data
                feature_name, value = rest.split(':', 1)
                parsed.append((col_num, feature_name, value))
    return parsed


def map_features(features_list):
    """
    Takes a list of tuples (col_num, feature_name, feat_val) and
    Returns a dict: col_num → (feature_name, feat_val)
    """
    mapping = {}
    for entry in features_list:
        col_str, feature_name, feat_val = entry
        mapping[int(col_str)] = (feature_name, feat_val)
    return mapping

def parse_feat_file(feat_path, feature_mapping):
    """
    Parses one .feat or .egofeat file.
    
    - If feat_path ends with '.egofeat', we treat the filename prefix as the single node_id,
      and expect each line to be just the bit-vector (no node_id column).
    - Otherwise ('.feat'), we expect each line as:
          node_id b1 b2 b3 ... bN

    Returns a list of dicts:
      [{ 'node_id': …, feat_name1: val or None, … }, …]
    """
    rows = []
    # sorted list of column indices from the mapping
    cols = sorted(feature_mapping.keys())
    feature_names = [feature_mapping[c][0] for c in cols]

    is_ego_file = feat_path.endswith('.egofeat')
    if is_ego_file:
        # extract node_id from filename, e.g. '0.egofeat' → '0'
        node_id = os.path.basename(feat_path).split('.', 1)[0]

    with open(feat_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if is_ego_file:
                bits = parts
            else:
                # regular .feat: first token is node_id
                node_id, bits = parts[0], parts[1:]

            if len(bits) < len(cols):
                raise ValueError(f"Too few bits ({len(bits)}) for expected {len(cols)} cols in {feat_path!r}")

            # build the row
            row = {'node_id': int(node_id)}
            for idx, col in enumerate(cols):
                bit = bits[idx]
                feat_name, feat_val = feature_mapping[col]
                row[feat_name] = feat_val if bit == '1' else None

            rows.append(row)

    return rows

def combine_to_csv(results, output_csv_path):
    """
    - results: list of dicts like [{'node_id': '698', 'birthday': None, …}, …]
    - output_csv_path: where to write the combined CSV
    - fill_with_none: if True, replaces pandas NaN with Python None before saving
    """
    # Create DataFrame (missing keys → NaN)
    df = pd.DataFrame(results)

    df = df.where(pd.notnull(df), None)

    # Sort by node_id    
    df.sort_values(by='node_id', inplace=True)

    # suppose df is your DataFrame with duplicates
    df = df.groupby('node_id', as_index=False).first()

    # Write to CSV; index=False so node_id is a column, not the index
    df.to_csv(output_csv_path, index=False)

    return df


def extract_features_to_csv(data_dir, output_csv):
    """
    Extracts features for all ego_ids.
    """
    feature_list = []
    e_ids = list_ego_ids(data_dir)
    for id in e_ids:
        features = parse_featnames(os.path.join(data_dir, f"{id}.featnames"))
        map = map_features(features)
        feature_list = feature_list + parse_feat_file(os.path.join(data_dir, f"{id}.egofeat"), map)
        feature_list = feature_list + parse_feat_file(os.path.join(data_dir, f"{id}.feat"), map)
    
    return combine_to_csv(feature_list, output_csv)
    

if __name__ == "__main__":
    data_dir = "data/facebook"
    output_csv = "data/facebook_features.csv"
    df = extract_features_to_csv(data_dir, output_csv)
    
    # dup_ids = df['node_id'][df['node_id'].duplicated()].unique()
    # print("Duplicate node_id values:", dup_ids)

   # 1. Count non-null (i.e. not None/NaN) per column
    non_null_counts = df.count()

    # 2. Count unique non-null values per column
    unique_counts   = df.nunique(dropna=True)

    # 3. Assemble into one table
    summary = pd.DataFrame({
        'non_null_count': non_null_counts,
        'unique_count':   unique_counts
    })

    # 4. Sort by non_null_count descending (or swap to sort by unique_count)
    summary_sorted = summary.sort_values(by='non_null_count', ascending=False)

    print(summary_sorted)