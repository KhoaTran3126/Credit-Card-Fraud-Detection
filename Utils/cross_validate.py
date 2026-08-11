def cross_validate(model, model_name, X, y, save_model_path, scorer, n_splits=5, random_state=3126):
  """
    Params:
        model: unfitted sklearn-compatible estimator to clone and fit each fold.
        model_name (str): one of "lgbm", "xgb", "catboost", or other (plain .fit()).
        X (pd.DataFrame): feature matrix.
        y (array-like): target vector, index-aligned with X.
        save_model_path (str or None): directory to dump fold models to; skipped if None.
        scorer (callable): metric function scorer(y_true, y_pred) -> float.
        n_splits (int): number of StratifiedKFold splits. Default 5.
        random_state (int): seed for fold shuffling. Default 3126.
    Return:
        dict: {"scores": array of per-fold metric values, "OOF": out-of-fold predictions, "fold_number": fold id per sample}.
    """
  skf = StratifiedKFold(n_splits=n_splits, random_state=random_state, shuffle=True)
  history = {
    "scores": np.zeros(n_splits),
    "OOF": np.zeros(len(y)),
    "fold_number": np.zeros(len(y))
  }

  for k,(train_idx,val_idx) in enumerate(skf.split(X, y)):
    X_t, y_t = X.iloc[train_idx,:], y[train_idx]
    X_v, y_v = X.iloc[val_idx,:],   y[val_idx]

    ## Fits appropriate cloned model
    cloned_model = clone(model)
    if model_name=="lgbm":
      cloned_model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)],
        callbacks=[lightgbm.early_stopping(100, verbose=False)])  
    elif model_name=="xgb":
      cloned_model.fit(
        X_t, y_t,
        eval_set=[(X_v, y_v)], verbose=False)
    elif model_name=="catboost":
      cloned_model.fit(
        X_t, y_t,
        eval_set=(X_v, y_v),
        early_stopping_rounds=100, verbose=False)
    else:
      cloned_model.fit(X_t, y_t)

    ## Make predictions and store results
    y_preds = cloned_model.predict_proba(X_v)[:, -1]
    history["OOF"][val_idx] = y_preds
    history["fold_number"][val_idx] = k+1
    history["scores"][k] = scorer(y_v, y_preds)
    ## Saves model if needed 
    if save_model_path is not None:
      joblib.dump(cloned_model, f"{save_model_path}/{model_name}_{k+1}.joblib")
  return history
