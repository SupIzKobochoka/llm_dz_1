import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from statistics import median

from datasets import load_dataset
from transformers import AutoTokenizer
from utils import schemas, parse_reference

MODEL = 'Qwen/Qwen3.5-2B'
REVISION = '15852e8c16360a2fea060d615a32b45270f8a8fc'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-dialogs', type=int, default=2500)
    ap.add_argument('--max-prompt-tokens', type=int, default=1800)
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    output = Path('data'); output.mkdir(exist_ok=True)
    raw = load_dataset('Team-ACE/ToolACE', split='train')
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    rng = random.Random(args.seed)
    indices = list(range(len(raw))); rng.shuffle(indices)
    split_point = int(len(indices) * .9)
    train_ids = set(indices[:split_point]); validation_ids = set(indices[split_point:])
    rng.shuffle(indices)
    examples = {'train': [], 'val': []}
    dialogue_counts = Counter(); skip = Counter(); sizes = []
    for idx in indices[:args.max_dialogs]:
        item = raw[idx]
        dialog = item['conversations']
        tools = schemas(item['system'])
        if not tools:
            skip['no_schema'] += 1; continue
        history = []
        split = 'train' if idx in train_ids else 'val'
        dialogue_counts[split] += 1
        for turn in dialog:
            role, content = turn['from'], turn['value']
            if role == 'user':
                history.append({'role': 'user', 'content': content})
            elif role == 'assistant':
                target = parse_reference(content)
                if target is None:
                    target = [] if not content.lstrip().startswith('[') else None
                if target is None:
                    skip['unparseable_call'] += 1
                elif history and history[-1]['role'] == 'user':
                    try:
                        prompt = tokenizer.apply_chat_template(history, tools=tools, tokenize=False,
                                                               add_generation_prompt=True, enable_thinking=False)
                        length = len(tokenizer.encode(prompt, add_special_tokens=False))
                    except Exception:
                        skip['template_error'] += 1; continue
                    if length <= args.max_prompt_tokens:
                        examples[split].append({'prompt': prompt, 'target': json.dumps(target, ensure_ascii=False),
                                                'dialog_id': idx, 'prompt_tokens': length, 'n_tools': len(tools)})
                        sizes.append(length)
                    else:
                        skip['too_long'] += 1
                if target == []:
                    history.append({'role': 'assistant', 'content': content})
                else:
                    history = []
            elif role == 'tool':
                continue
    for split, rows in examples.items():
        with (output / f'{split}.jsonl').open('w') as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False) + '\n')
    assert {r['dialog_id'] for r in examples['train']}.isdisjoint({r['dialog_id'] for r in examples['val']})
    stats = {'dataset': 'Team-ACE/ToolACE', 'dataset_rows': len(raw), 'seed': args.seed,
             'max_dialogs': args.max_dialogs, 'train_dialogs': dialogue_counts['train'],
             'val_dialogs': dialogue_counts['val'], 'train_samples': len(examples['train']),
             'val_samples': len(examples['val']), 'no_tool_samples': sum(not json.loads(r['target']) for x in examples.values() for r in x),
             'prompt_tokens_median': median(sizes) if sizes else 0,
             'prompt_tokens_max': max(sizes) if sizes else 0,
             'tools_median': median([r['n_tools'] for x in examples.values() for r in x]) if sizes else 0,
             'skipped': dict(skip)}
    (output / 'eda.json').write_text(json.dumps(stats, indent=2, ensure_ascii=False))
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not examples['train'] or not examples['val']: raise RuntimeError('No examples: inspect ToolACE schema/template')

if __name__ == '__main__': main()