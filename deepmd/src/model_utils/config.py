"""Parse arguments"""
import os
import ast
import argparse
from pprint import pprint, pformat
import yaml


_config_path = '../../default_config.yaml'


class Config:
    """
    Configuration namespace. Convert dictionary to members
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()


def parse_cli_to_yaml(parser, cfg, helper=None, choices=None, cfg_path='default_config.yaml'):
    """
    Parse command line arguments to the configuration according to the default yaml

    Args:
        parser: Parent parser
        cfg: Base configuration
        helper: Helper description
        cfg_path: Path to the default yaml config
    """
    parser = argparse.ArgumentParser(description='[REPLACE THIS at config.py]',
                                     parents=[parser])
    helper = {} if helper is None else helper
    choices = {} if choices is None else choices
    for item in cfg:
        if not isinstance(cfg[item], list) and not isinstance(cfg[item], dict):
            help_description = helper[item] if item in helper else 'Please reference to {}'.format(cfg_path)
            choice = choices[item] if item in choices else None
            if isinstance(cfg[item], bool):
                parser.add_argument('--' + item, type=ast.literal_eval, default=cfg[item], choices=choice,
                                    help=help_description)
            else:
                parser.add_argument('--' + item, type=type(cfg[item]), default=cfg[item], choices=choice,
                                    help=help_description)
    args = parser.parse_args()
    return args


def parse_yaml(yaml_path):
    """
    Parse the yaml config file

    Args:
        yaml_path: Path to the yaml config
    """
    with open(yaml_path, 'r') as fin:
        try:
            cfgs = yaml.load_all(fin.read(), Loader=yaml.FullLoader)
            cfgs = [x for x in cfgs]
            if len(cfgs) == 1:
                cfg_helper = {}
                cfg = cfgs[0]
                cfg_choices = {}
            elif len(cfgs) == 2:
                cfg, cfg_helper = cfgs
                cfg_choices = {}
            elif len(cfgs) == 3:
                cfg, cfg_helper, cfg_choices = cfgs
            else:
                raise ValueError('At most 3 docs (config description for help, choices) are supported in config yaml')
            print(cfg_helper)
        except:
            raise ValueError('Failed to parse yaml')
    return cfg, cfg_helper, cfg_choices


def merge(args, cfg):
    """
    Merge the base config from yaml file and command line arguments

    Args:
        args: command line arguments
        cfg: Base configuration
    """
    args_var = vars(args)
    for item in args_var:
        cfg[item] = args_var[item]
    return cfg


def get_config():
    """
    Get Config according to the yaml file and cli arguments
    """
    parser = argparse.ArgumentParser(description='default name', add_help=False)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--config_path', type=str, default=os.path.join(current_dir, _config_path),
                        help='Config file path')
    path_args, _ = parser.parse_known_args()
    default, helper, choices = parse_yaml(path_args.config_path)
    args = parse_cli_to_yaml(parser=parser, cfg=default, helper=helper, choices=choices, cfg_path=path_args.config_path)
    final_config = merge(args, default)
    pprint(final_config)
    print("Please check the above information for the configurations", flush=True)
    return Config(final_config)

config = get_config()
