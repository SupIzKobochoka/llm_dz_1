import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct' # Взял его, чтобы помещался в 16gb, иначе вроде 1.5B не помещается :(
LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
SEED = 42
MAX_LENGTH = 192
TRAIN_SIZE = 10000
RESULTS = Path('results')


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def load_data():
    data = load_dataset('dair-ai/emotion', 'split')
    train = data['train'].shuffle(seed=SEED).select(range(TRAIN_SIZE))
    test = data['test']
    return train, test


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    tok.pad_token = tok.eos_token
    return tok


def load_model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)


def prompt(tokenizer, text):
    # Классификация: модель должна вернуть только цифру от 0 до 5.
    # Обрезаем длинные тексты одинаково во всех экспериментах.
    text_ids = tokenizer(text, add_special_tokens=False)['input_ids'][:80]
    text = tokenizer.decode(text_ids, skip_special_tokens=True)
    messages = [
        {'role': 'system', 'content': 'Classify the emotion of the text. Return ONLY one digit: '
         '0=sadness, 1=joy, 2=love, 3=anger, 4=fear, 5=surprise.'},
        {'role': 'user', 'content': text},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def tokenize_train(train, tokenizer):
    def encode(row):
        prefix = tokenizer(prompt(tokenizer, row['text']), add_special_tokens=False)['input_ids']
        answer = tokenizer(str(row['label']), add_special_tokens=False)['input_ids']
        # Сохраняем конец промпта и ответ даже для длинных примеров.
        prefix = prefix[-(MAX_LENGTH - len(answer) - 1):]
        ids = prefix + answer + [tokenizer.eos_token_id]
        labels = [-100] * len(prefix) + answer + [tokenizer.eos_token_id]
        return {'input_ids': ids, 'attention_mask': [1] * len(ids), 'labels': labels}

    return train.map(encode, remove_columns=train.column_names)


def collate(rows):
    max_len = max(len(row['input_ids']) for row in rows)
    return {
        'input_ids': torch.tensor([row['input_ids'] + [0] * (max_len - len(row['input_ids'])) for row in rows]),
        'attention_mask': torch.tensor([row['attention_mask'] + [0] * (max_len - len(row['attention_mask'])) for row in rows]),
        'labels': torch.tensor([row['labels'] + [-100] * (max_len - len(row['labels'])) for row in rows]),
    }


def train(model, tokenizer, train_data, output_dir, lr):
    model.config.use_cache = False
    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=lr,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=100,
        save_strategy='no',
        report_to='none',
        seed=SEED,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_data,
                      data_collator=collate, processing_class=tokenizer)
    torch.cuda.reset_peak_memory_stats()
    import time
    start = time.perf_counter()
    trainer.train()
    seconds = round(time.perf_counter() - start, 1)
    peak_gb = round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2)
    model.config.use_cache = True
    return seconds, peak_gb


@torch.inference_mode()
def evaluate(model, tokenizer, test, batch_size=16):
    model.eval()
    model.to('cuda')
    tokenizer.padding_side = 'left'
    # Каждая цифра должна быть представлена одним токеном.
    class_ids = [tokenizer(str(i), add_special_tokens=False)['input_ids'] for i in range(6)]
    assert all(len(ids) == 1 for ids in class_ids), class_ids
    class_ids = torch.tensor([ids[0] for ids in class_ids], device='cuda')
    predictions = []
    for start in range(0, len(test), batch_size):
        batch = test.select(range(start, min(start + batch_size, len(test))))
        inputs = tokenizer([prompt(tokenizer, x) for x in batch['text']],
                           padding=True, truncation=True, max_length=MAX_LENGTH,
                           return_tensors='pt').to('cuda')
        logits = model(**inputs).logits[:, -1, :]
        predictions.extend(logits[:, class_ids].argmax(dim=1).cpu().tolist())
    y_true = list(test['label'])
    return {
        'accuracy': round(accuracy_score(y_true, predictions), 4),
        'macro_f1': round(f1_score(y_true, predictions, average='macro'), 4),
    }, predictions


def save_result(name, config, model, tokenizer, test, seconds=0, peak_gb=0):
    RESULTS.mkdir(exist_ok=True)
    metrics, predictions = evaluate(model, tokenizer, test)
    metrics.update({
        'experiment': name,
        'trainable_params': sum(p.numel() for p in model.parameters() if p.requires_grad),
        'train_seconds': seconds,
        'peak_gpu_gb': peak_gb,
        **config,
    })
    (RESULTS / f'{name}.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    (RESULTS / f'{name}_predictions.json').write_text(json.dumps(predictions), encoding='utf-8')
    print(name, metrics)


def compare():
    RESULTS.mkdir(exist_ok=True)
    rows = [json.loads(path.read_text(encoding='utf-8')) for path in RESULTS.glob('*.json')
            if not path.name.endswith('_predictions.json')]
    if not rows:
        print('Нет результатов. Сначала запусти эксперименты.')
        return
    columns = ['experiment', 'accuracy', 'macro_f1', 'trainable_params', 'train_seconds',
               'peak_gpu_gb', 'r', 'alpha', 'dropout', 'target_modules', 'learning_rate']
    with (RESULTS / 'summary.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x['experiment']))

    _, test = load_data()
    selected = [0, 1, 2, 3, 4, 5, 6, 7]
    preds = {}
    for name in ['baseline', 'full_ft', 'lora_base']:
        path = RESULTS / f'{name}_predictions.json'
        if path.exists():
            preds[name] = json.loads(path.read_text(encoding='utf-8'))
    with (RESULTS / 'manual_examples.csv').open('w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['text', 'true_label', *preds.keys()])
        for i in selected:
            writer.writerow([test[i]['text'], LABELS[test[i]['label']],
                             *[LABELS[pred[i]] for pred in preds.values()]])
    print('Готово: results/summary.csv и results/manual_examples.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['baseline', 'compare'])
    cmd = parser.parse_args().command
    if cmd == 'compare':
        compare()
    else:
        set_seed()
        tokenizer = load_tokenizer()
        model = load_model()
        _, test = load_data()
        save_result('baseline', {}, model, tokenizer, test)
