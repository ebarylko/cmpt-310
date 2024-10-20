import toolz as tz
import functools as ft
import operator as op


def modify_config(config, key, val):
    return tz.assoc(config, key, val)


@tz.curry
def gen_model_configs(default_configuration, key, values):
    """
    @param default_configuration: the default set of values to use
    for a model
    @param key: one of the values used to configure the model
    @param values: the new values to use for key
    @return: generates a new model configuration for each
     value in values, overwriting the old value for key with the
     new value
    """
    new_config = ft.partial(modify_config, default_configuration, key)
    return list(map(new_config, values))


def gen_model_configs_for_many_parameters(default_configuration, params_and_new_values):
    """
    @param default_configuration:  the default set of values to use
    for a model
    @param params_and_new_values: a collection of pairs, where each pair contains
    a model setting and the new values to use for those settings
    @return: a collection of model configurations using the new values for the
    params given in params_and_new_values
    """
    gen_config = gen_model_configs(default_configuration)

    return tz.thread_last(params_and_new_values,
                          (map, lambda coll: gen_config(*coll)),
                          (ft.reduce, op.add))


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


expected = [{"input_size": 28**2,
             "hidden_layers": (128, ),
             "output_size": 10,
             'learning_rate': 0.01,
             "epochs": 20,
             "batch_size": 64},
            {"input_size": 28**2,
             "hidden_layers": (128, ),
             "output_size": 10,
             'learning_rate': 0.1,
             "epochs": 5,
             "batch_size": 64},
            {"input_size": 28**2,
             "hidden_layers": (128, ),
             "output_size": 10,
             'learning_rate': 0.2,
             "epochs": 5,
             "batch_size": 64}]


def test_gen_model_configs_for_many_parameters():
    assert expected == gen_model_configs_for_many_parameters(default_config,
                                                             (["epochs", [20]],
                                                              ["learning_rate", [0.1, 0.2]]))
