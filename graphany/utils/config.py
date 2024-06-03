from omegaconf import DictConfig, OmegaConf


def save_config(cfg: DictConfig, path, as_global=True):
    OmegaConf.save(config=DictConfig(cfg), f=path)
    if as_global:
        with open(path, "r") as input_file:
            original_content = input_file.read()

        with open(path, "w") as output_file:
            output_file.write("# @package _global_\n" + original_content)
    return cfg


# ! Custom OmegaConf Resolvers


def rename_alias(alias: str):
    replace_dict = {"true": "T", "false": "F", "True": "T", "False": "F"}
    for key, value in replace_dict.items():
        alias = alias.replace(key, value)
    return alias


def _eval(*args, **kwargs):  # Eval wrapper for debugging
    return eval(*args, **kwargs)


# Register resolvers
OmegaConf.register_new_resolver("eval", _eval)
OmegaConf.register_new_resolver("rename_alias", rename_alias)
