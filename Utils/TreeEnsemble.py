class TreeEnsemble():
    def __init__(self, weights, cat_features=CAT_FEATURES, 
                 lgbm_params:dict=LGBM_PARAMS, xgb_params:dict=XGB_PARAMS, cat_params:dict=CAT_PARAMS):
        self.weights = np.asarray(weights)
        self.cat_features = cat_features
        self.lgbm_params  = lgbm_params
        self.xgb_params   = xgb_params
        self.cat_params   = cat_params
        self.lgbm_mod = LGBMClassifier(**lgbm_params)
        self.xgb_mod  = XGBClassifier(**xgb_params)
        self.cat_mod  = CatBoostClassifier(**cat_params)
        
    def fit(self, X, y):
        self.lgbm_mod.fit(X, y)
        self.xgb_mod.fit(X, y)
        self.cat_mod.fit(X, y, cat_features=self.cat_features, verbose=False)
        
    def predict_proba(self, X):
        lgbm_preds = self.lgbm_mod.predict_proba(X)[:,1]
        xgb_preds  = self.xgb_mod.predict_proba(X)[:,1]
        cat_preds  = self.cat_mod.predict_proba(X)[:,1]

        final_preds = (
            self.weights[0]*lgbm_preds + 
            self.weights[1]*xgb_preds + 
            self.weights[2]*cat_preds
        )
        return final_preds 
