def test_model(model, model_name, X_test, y_test, scorer):
  """
    Evaluate a trained classifier on a test set using probability-based scoring
    and print the result.

    Params:
        model (estimator): Trained classifier with a `predict_proba` method.
        model_name (str): Name of the model, used for the printed output.
        X_test: Test set features.
        y_test: True labels for the test set.
        scorer (callable): Scoring function with signature `scorer(y_true, y_score)`
    Return:
        None. Prints the computed score to stdout.
    """
  y_preds = model.predict_proba(X_test)[:, -1]
  score   = scorer(y_test, y_preds)
  print(f"Score of {model_name} on Test Set: {score:.3f}")
