def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X_copy = X.copy()
    X_copy["txn_count_last_24h"] = X_copy["txn_count_last_24h"].clip(upper=9)
    return X_copy


def convert_cat_features(X:pd.DataFrame, cat_features:list) -> pd.DataFrame:
    X_copy = X.copy()
    X_copy[cat_features] = X_copy[cat_features].astype("category")
    return X_copy


def process_features(X:pd.DataFrame, cat_features:list) -> pd.DataFrame:
    X_copy = engineer_features(X)
    X_copy = convert_cat_features(X_copy, cat_features)
    return X_copy
