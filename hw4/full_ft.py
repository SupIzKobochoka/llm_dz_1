from pathlib import Path

from common import (load_data, load_model, load_tokenizer, save_result,
                    set_seed, tokenize_train, train)


def main():
    set_seed()
    tokenizer = load_tokenizer()
    raw_train, test = load_data()
    train_data = tokenize_train(raw_train, tokenizer)
    model = load_model()

    # Full FT: все веса модели обучаются.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    seconds, peak = train(model, tokenizer, train_data, Path('checkpoints/full_ft'), lr=2e-5)

    save_result('full_ft', {'learning_rate': 2e-5}, model, tokenizer, test, seconds, peak)
    model.save_pretrained('models/full_ft')
    tokenizer.save_pretrained('models/full_ft')


if __name__ == '__main__':
    main()
