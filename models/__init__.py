from models import naiveregressor

def select(model_name):
    if model_name=='naive_regressor':
        return (1024, 1024), naiveregressor.NaiveRegressor()
