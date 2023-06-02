import numpy as np


def max_min_distance(default_config, src_configs, num):
    min_dis = list()
    initial_configs = list()
    initial_configs.append(default_config)

    for config in src_configs:
        dis = np.linalg.norm(config.get_array() - default_config.get_array())
        min_dis.append(dis)
    min_dis = np.array(min_dis)

    for i in range(num):
        furthest_config = src_configs[np.argmax(min_dis)]
        initial_configs.append(furthest_config)
        min_dis[np.argmax(min_dis)] = -1

        for j in range(len(src_configs)):
            if src_configs[j] in initial_configs:
                continue
            updated_dis = np.linalg.norm(src_configs[j].get_array() - furthest_config.get_array())
            min_dis[j] = min(updated_dis, min_dis[j])

    return initial_configs

