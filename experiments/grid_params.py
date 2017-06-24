GRID_PARAMS = {
    "rdf": {
        "class_weight": [ "balanced_subsample", "balanced" ],
        "max_depth": [ 5, 20, None ],
        "max_features": [ 25, 100, None, "auto" ],
        "min_samples_leaf": [ 1, 3 ],
        "n_estimators": [ 10, 30, 100 ]
    },
    "svc": {
        "C": [ 0.01, 0.1, 1 ],
        "class_weight": [ "balanced", None ],
        "gamma": [ 0.1, 1, 10 ],
        "kernel": [ "rbf", "poly" ]
    }
}