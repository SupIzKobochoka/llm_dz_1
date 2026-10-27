import argparse
import gc
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model

from common import (load_data, load_model, load_tokenizer, save_result,
                    set_seed, tokenize_train, train)

# Меняем по одному параметру относительно base.
EXPERIMENTS = {
    'base': dict(r=16, alpha=32, dropout=0.05, target_modules=['q_proj', 'v_proj']),
    'r8': dict(r=8, alpha=32, dropout=0.05, target_modules=['q_proj', 'v_proj']),
    'r32': dict(r=32, alpha=32, dropout=0.05, target_modules=['q_proj', 'v_proj']),
    'alpha16': dict(r=16, alpha=16, dropout=0.05, target_modules=['q_proj', 'v_proj']),
    'alpha64': dict(r=16, alpha=64, dropout=0.05, target_modules=['q_proj', 'v_proj']),
    'dropout0': dict(r=16, alpha=32, dropout=0.0, target_modules=['q_proj', 'v_proj']),
    'dropout01': dict(r=16, alpha=32, dropout=0.1, target_modules=['q_proj', 'v_proj']),
    'all_linear': dict(r=16, alpha=32, dropout=0.05, target_modules='all-linear'),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', choices=['all', *EXPERIMENTS], default='all')
    args = parser.parse_args()
    set_seed()
    tokenizer = load_tokenizer()
    raw_train, test = load_data()
    train_data = tokenize_train(raw_train, tokenizer)

    names = list(EXPERIMENTS) if args.experiment == 'all' else [args.experiment]
    for name in names:
        set_seed()
        conf = EXPERIMENTS[name]
        model = load_model()
        model = get_peft_model(model, LoraConfig(
            r=conf['r'], lora_alpha=conf['alpha'], lora_dropout=conf['dropout'],
            target_modules=conf['target_modules'], task_type='CAUSAL_LM',
        ))
        seconds, peak = train(model, tokenizer, train_data,
                              Path(f'checkpoints/lora_{name}'), lr=2e-4)
        save_result(f'lora_{name}', {**conf, 'learning_rate': 2e-4},
                    model, tokenizer, test, seconds, peak)
        model.save_pretrained(f'models/lora_{name}')
        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
