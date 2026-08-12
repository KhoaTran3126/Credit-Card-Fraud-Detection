def make_train_test_sets(path:str, target_variable="is_fraud", test_size=0.3, random_state=3126):
    """
    Parameters:
        path (str): Path to the CSV file containing the dataset.
        target_variable (str): Name of the target column to predict. Default "is_fraud".
        test_size (float): Proportion of data to allocate to the test set. Default 0.3.
        random_state (int): Seed for reproducible splitting. Default 3126.
    Return:
        X_train, X_test, y_train, y_test: Stratified train/test split of features and target.
    """
    ## Reads data and splits into X and y 
    df = pd.read_csv(path)
    X = df.copy()
    y = X.pop(target_variable)

    ## Splits intro train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=test_size, random_state=random_state, shuffle=True, stratify=y
    )

    return X_train, X_test, y_train, y_test
