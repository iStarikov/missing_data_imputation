from imputation_models import DummyImputation, SVDImputation, MFImputation, KNNImputation, IterativeImputation, \
    AEImputation, FastTextEmbeddingsImputation

MODEL_ABBREVIATION_MAPPING = {
    'median': (DummyImputation, dict(strategy_name='median')),
    'mean': (DummyImputation, dict(strategy_name='mean')),
    'mode': (DummyImputation, dict(strategy_name='mode')),
    'constant': (DummyImputation, dict(strategy_name='constant')),
    'mice': (IterativeImputation, dict(n_iter=10)),
    'svd': (SVDImputation, dict(rank=10)),
    'mf': (MFImputation, dict(rank=10)),
    'knn': (KNNImputation, dict(k=10)),
    'ae': (AEImputation, dict(strategy_name='ae')),
    'emb': (FastTextEmbeddingsImputation, dict(strategy_name='emb'))
}


def model_factory(model_abbr: str, **kwargs):
    """
    Returns model instance given abbreviation with mapping in MODEL_ABBREVIATION_MAPPING and hyperparameters as kwargs
    :param model_abbr: abbreviation from MODEL_ABBREVIATION_MAPPING keys
    :param kwargs: passed directly to model __init__() method
    :return: model instance
    """
    if model_abbr in MODEL_ABBREVIATION_MAPPING:
        model, params = MODEL_ABBREVIATION_MAPPING[model_abbr]
        if kwargs:
            params.update(kwargs)
        return model(**params)
    raise NotImplementedError(f"Unknown model for abbreviation {model_abbr}")


def get_model(abbr: str, **kwargs) -> callable:
    """
    Returns factory method without parameters for specified model.
    Convenient for constructing multiple instances of the same model with the same hyperparameters.
    :param abbr: abbreviation from MODEL_ABBREVIATION_MAPPING keys
    :param kwargs: passed directly to model __init__() method
    :return:
    """

    def factory():
        return model_factory(abbr, **kwargs)

    return factory
