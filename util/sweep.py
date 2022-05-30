from loguru import logger
import itertools
import os
import json
import pickle
import wandb

def common_config(configs):
    items = [ set(c.items()) for c in configs]
    common_items = items[0].intersection(*items[1:])
    return dict(common_items)

def diff_config(base_config, config):
    diff = {k : v for (k, v) in config.items() if v != base_config.get(k,None) }
    return diff

def make_name(diff):
    items = [f'{k}_{v}' for (k, v) in diff.items()]
    if not items:
        return 'base'
    else:
        return '__'.join(items)

def log_info(dict):
    for (k, v) in dict.items():
        logger.opt(colors=True).info(f'    <blue>{k}</blue>: {v}')

def run_all(train_fn, configs, name=None, output_base=None, use_wandb=False):
    sweep_base = name or "manual_run"
    if use_wandb:
        assert len(configs) == 1
        config = configs[0]
        wandb.init(project="safe_imitation_learning", config=config)
        sweep_base = f'{wandb.run.name}__{wandb.run.id}'
    if output_base and output_base != "None":
        for i in itertools.count():
            sweep_name = f'{sweep_base}_{i}' if i > 0 else sweep_base
            if not os.path.exists(os.path.join(output_base, sweep_name)):
                break
        output_path = os.path.join(output_base, sweep_name)
        logger.opt(colors=True).info(f"Using output directory <blue>{output_path}</blue>")
        os.makedirs(output_path, exist_ok=True)
        sweeps = len(configs)

    base_config = common_config(configs)
    for (i, config) in enumerate(configs):
        config_diff = diff_config(base_config, config)
        # Diff the config and log the sweep 
        if sweeps > 1:
            logger.opt(colors=True).info(f"<red>Sweep run</red> <blue>{i+1}</blue>/<blue>{sweeps}</blue>")
            log_info(config_diff)
        stats, final_params = train_fn(config)
        if use_wandb:
            for k, v in stats.items():
                wandb.run.summary[k] = v
        # Save the output as JSON
        if output_path:
            run_name = make_name(config_diff)
            output = {'name': run_name, 'config': config, 'config_diff': config_diff, 'stats': stats}
            output_json_file = os.path.join(output_path, f'{run_name}.json')
            output_weight_file = os.path.join(output_path, f'{run_name}.pk')
            logger.opt(colors=True).info(f"Saving stats to <blue>{output_json_file}</blue>")
            with open(output_json_file, 'w') as f:
                json.dump(output, f)
            logger.opt(colors=True).info(f"Saving weights to <blue>{output_weight_file}</blue>")
            with open(output_weight_file, "wb") as f:
                pickle.dump(output, f)

def read_all(output_base, **sweep_names):
    sweeps = []
    for s in sweep_names:
        sweep_path = os.path.join(output_base)
    return sweeps