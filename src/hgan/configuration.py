import os
import sys
import shutil
import configparser


class ConfigSection(object):
    """
    A thin wrapper over a ConfigParser's SectionProxy object,
    that tries to infer the types of values, and makes them available as attributes
    Currently int/float/str are supported.
    """

    def __init__(self, config, section_proxy):
        self.config = config
        self.name = section_proxy.name
        self.d = {}  # key value dict where the value is typecast to int/float/str

        for k, v in section_proxy.items():
            if v in ("True", "False"):
                self.d[k] = eval(v)
                continue

            try:
                v = int(v)
            except ValueError:
                try:
                    v = float(v)
                except ValueError:
                    # We interpret a missing value as None, and a "" as the empty string
                    if v.startswith('"') and v.endswith('"'):
                        v = v[1:-1]
                    elif v == "":
                        v = None
                    self.d[k] = v
                else:
                    self.d[k] = v
            else:
                self.d[k] = v

    def __setattr__(self, key, value):
        if key in ("config", "name", "d"):
            return super(ConfigSection, self).__setattr__(key, value)
        else:
            self.d[key] = value

    def __getattr__(self, item):
        if item not in ("config", "name", "d"):
            config_value = self.d[item]
            # If an environment variable exists with name <CONFIG_NAME>_<SECTION>_<ITEM>, use it
            env_varname = "_".join(
                [str(x).upper() for x in [self.config.name, self.name, item]]
            )
            env_var = os.getenv(env_varname)
            if env_var is not None:
                val = type(config_value)(env_var)
                if val == "":
                    return None
                else:
                    return val
            return config_value

    def items(self):
        for k in self.d:
            yield k, getattr(self, k)


class Config(object):
    def __init__(self, name, filenames):
        self.name = name
        self.config = configparser.ConfigParser(inline_comment_prefixes="#")
        self.file_paths = []
        self.init_from_files(filenames)

    def init_from_files(self, filenames):
        file_paths = self.config.read(filenames)
        self.file_paths = file_paths
        self._read_sections()

    def read(self, filename):
        self.config.read(filename)
        self._read_sections()

    def _read_sections(self):
        for section in self.config.sections():
            setattr(self, section, ConfigSection(self, self.config[section]))

    def sections(self):
        return self.config.sections()

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)
        for file_path in self.file_paths:
            dest = os.path.join(folder, os.path.basename(file_path))
            shutil.copy(file_path, dest)


def save_config(folder):
    global config

    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "configuration.ini"), mode="w") as f:
        config.config.write(f)


def load_config(file_or_folder_path):
    global config
    assert os.path.exists(file_or_folder_path)
    if os.path.isdir(file_or_folder_path):
        config_file = os.path.join(file_or_folder_path, "configuration.ini")
    else:
        config_file = file_or_folder_path

    config = Config("hgan", config_file)
    return config


def show_config():
    config.config.write(sys.stdout)


config = load_config(os.path.dirname(__file__))
