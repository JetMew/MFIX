import configparser


class DictParser(configparser.ConfigParser):
    def get_dict(self):
        d = dict(self._sections)
        for k in d:
            d[k] = dict(d[k])
        return d


def parse_args(config_file='config.ini'):
    cf = DictParser()
    cf.read(config_file, encoding='utf8')
    config_dict = cf.get_dict()

    config_base = config_dict['base']
    config_algo = dict()

    for k in config_dict:
        if k == 'base':
            continue
        config_algo[k] = config_dict[k]
    return config_base, config_algo
