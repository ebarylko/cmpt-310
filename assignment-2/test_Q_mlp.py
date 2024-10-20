import toolz as tz
import functools as ft


def modify_config(config, key, val):
    return tz.assoc(config, key, val)


def gen_model_configs(default_configuration, key, values):
    """
    @param default_configuration: the default set of values to use
    for a model
    @param key: one of the values used to configure the model
    @param values: the new values to use for key
    @return: a collection of maps new model configurations
     for each value in values containing the new value for key
    """
    new_config = ft.partial(modify_config, default_configuration, key)
    return list(map(new_config, values))


expected_configs = [{"input_size": 28**2,
                     "hidden_layers": (128, ),
                     "output_size": 10,
                     'learning_rate': 0.01,
                     "epochs": 10,
                     "batch_size": 64},
                    {"input_size": 28**2,
                     "hidden_layers": (128, ),
                     "output_size": 10,
                     'learning_rate': 0.01,
                     "epochs": 20,
                     "batch_size": 64}]

default_config = {"input_size": 28**2,
                  "hidden_layers": (128, ),
                  "output_size": 10,
                  'learning_rate': 0.01,
                  "epochs": 5,
                  "batch_size": 64}


def test_gen_model_configs():
    assert expected_configs == gen_model_configs(default_config, "epochs", [10, 20])