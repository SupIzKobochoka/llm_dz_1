"""Минимальный запуск ДЗ 2. Каждый эксперимент -> results/<имя>/{train.log,summary.json}."""
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Параметры модели, данных, оптимизатора, scheduler, seed, BF16 и длины 512 не меняем.
# Liger и torch.compile: отключены, поскольку Triton JIT не работает с драйвером gpu (почему-то, обновлять не стал, потому что страшна).

EXPERIMENTS = {
    "baseline":            (300, "sdpa",              32, 2, False, False, False),
    "batch_8":             (90,  "sdpa",               8, 8, False, False, False),
    "checkpoint":          (90,  "sdpa",              32, 2, True,  False, False),
    "batch_checkpoint":    (90,  "sdpa",              16, 4, True,  False, False),
    "flash":               (90,  "flash_attention_2", 32, 2, False, False, False),
    "flash_padding_free":  (90,  "flash_attention_2",  8, 8, False, False, True),
    "flash_packing":       (90,  "flash_attention_2",  8, 8, False, True,  False),
    "flash_packing_check": (90,  "flash_attention_2",  8, 8, True,  True,  False),
}


def run(name, config, duration=None):
    seconds, attention, batch, accumulation, checkpoint, packing, padding_free = config
    seconds = seconds if duration is None else duration
    output = HERE / "results" / name
    summary_file = output / "summary.json"
    if summary_file.exists():
        print(f"Пропуск {name}: summary.json уже существует", flush=True)
        return True
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update({
        "TRAIN_SECONDS": str(seconds),
        "ATTENTION_IMPLEMENTATION": attention,
        "HW2_OUTPUT_DIR": str(output),
        "HW2_PER_DEVICE_TRAIN_BATCH_SIZE": str(batch),
        "HW2_GRADIENT_ACCUMULATION_STEPS": str(accumulation),
        "HW2_GRADIENT_CHECKPOINTING": str(checkpoint).lower(),
        "HW2_PACKING": str(packing).lower(),
        "HW2_PADDING_FREE": str(padding_free).lower(),
    })
    print(f"Запуск {name}: {seconds} секунд", flush=True)
    # output_dir в исходном train.py создаётся с exist_ok=False.
    # Поэтому вызываем train.py с вложенной новой папкой, а logs держим снаружи.
    env["HW2_OUTPUT_DIR"] = str(output / "training")
    training = output / "training"
    if training.exists() and not (training / "summary.json").exists():
        print(f"Незавершённый запуск {name}: удаляю только {training}", flush=True)
        import shutil
        shutil.rmtree(training)
    with (output / "train.log").open("w") as log:
        process = subprocess.Popen([sys.executable, "train.py"], cwd=HERE, env=env,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        code = process.wait()
    if code == 0 and (training / "summary.json").exists():
        # Файл рядом с train.log для удобства просмотра.
        import shutil
        shutil.copy2(training / "summary.json", summary_file)
        return True
    print(f"Ошибка в {name}; подробности: {output / 'train.log'}", flush=True)
    return False


def report():
    output = HERE / "REPORT.md"
    rows = []
    baseline = HERE / "results/baseline/summary.json"
    baseline_speed = json.loads(baseline.read_text())["tokens_per_second"] if baseline.exists() else None
    for path in sorted((HERE / "results").glob("*/summary.json")):
        result = json.loads(path.read_text())
        speed = result.get("tokens_per_second")
        accel = f"{speed / baseline_speed:.3f}" if speed and baseline_speed else "—"
        rows.append(f"| {path.parent.name} | {result.get('train_seconds', 0):.1f} | "
                    f"{speed:.1f} | {result.get('peak_allocated_gib', 0):.2f} | "
                    f"{result.get('validation_loss', 0):.3f} | "
                    f"{result.get('optimizer_steps', 0)} | {accel} |")
    output.write_text("# Результаты ДЗ 2\n\n"
                      "Среда: A100 80 GB, исходные данные и метрики train.py преподавателя.\n\n"
                      "Liger и torch.compile не проверялись: Triton JIT несовместим с текущим окружением сервера.\n\n"
                      "| Конфигурация | Секунд | Токенов/сек | GPU, GiB | Val loss | Шагов | S |\n"
                      "|---|---:|---:|---:|---:|---:|---:|\n"
                      + "\n".join(rows) + "\n\n"
                      "## Выводы\n\n"
                      "Заполнить минимум три наблюдения по реальным результатам, включая сочетания оптимизаций.\n")
    print(f"Отчёт: {output}", flush=True)


def main():
    names = sys.argv[1:] or [*EXPERIMENTS, "best", "report"]
    for name in names:
        if name == "report":
            report()
        elif name == "best":
            candidates = []
            for key in EXPERIMENTS:
                if key == "baseline":
                    continue
                file = HERE / "results" / key / "summary.json"
                if file.exists():
                    speed = json.loads(file.read_text()).get("tokens_per_second")
                    if speed is not None:
                        candidates.append((speed, key))
            if candidates:
                best_name = max(candidates)[1]
                print(f"Лучший короткий запуск: {best_name}")
                run("final_" + best_name, EXPERIMENTS[best_name], 300)
            else:
                print("Нет успешных коротких экспериментов")
        elif name in EXPERIMENTS:
            run(name, EXPERIMENTS[name])
        else:
            raise SystemExit(f"Неизвестный эксперимент: {name}")


if __name__ == "__main__":
    main()
