SEED = 3126

CAT_FEATURES = [
    'merchant_category','card_type', 'auth_method',
    'channel', 'device_type', 'day_of_week'
]


LGBM_PARAMS = {
    'max_depth': 2,
    'learning_rate': 0.07461718009719123,
    'min_child_samples': 108,
    'colsample_bytree': 0.5369984053520471,
    'subsample_freq': 4,
    'subsample': 0.7511542308555871,
    'reg_alpha': 0.013074071755364779,
    'reg_lambda': 0.0008606603803609922,
    'scale_pos_weight': 1.1103974516053718,
    'num_leaves': 4,
    ## Constants
    'n_estimators':130,
    'random_state':SEED,
    'metric':'average_precision',
    'verbose':-1,
    'n_jobs':-1
}


XGB_PARAMS = {
    'learning_rate': 0.09337094415803539,
    'colsample_bytree': 0.6475628016664312,
    'subsample': 0.6975963364412459,
    'reg_alpha': 1.1303419470280986e-05,
    'reg_lambda': 0.0059375831020790195,
    'gamma': 0.0009892073499964695,
    'min_child_weight': 3.5939245218014833,
    'grow_policy': 'lossguide',
    'scale_pos_weight': 2.3635951892927065,
    'max_leaves': 4,
    ## Constants
    'n_estimators':112,
    'enable_categorical':True,
    'eval_metric':"aucpr",
    'random_state':SEED,
    'verbosity':0,
    'n_jobs':-1
}


CAT_PARAMS = {
    'learning_rate': 0.11422046426502673, 
    'l2_leaf_reg': 2.931964012979458, 
    'depth': 2, 
    'random_strength': 0.650825384661738, 
    'bootstrap_type': 'Bernoulli', 
    'subsample': 0.5998462283815383, 
    'rsm': 0.8092477080727568, 
    'min_data_in_leaf': 25,
    'class_weights': [1.0, 3.513505010540427],
    ## Constants
    'n_estimators':142,
    'random_state':SEED,
    'eval_metric':"PRAUC",
    'thread_count':-1
}
