def test_model(model, model_name, X_test, y_test, scorer):
  y_preds = model.predict_proba(X_test)[:, -1]
  score = scorer(y_test, y_preds)
  print(f"Score of {model_name} on Test Set: {score:.3f}")
