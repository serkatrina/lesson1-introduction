"""Домашнее задание 1, задание 3: сравнение производительности CPU и GPU.

Запуск:
    python homework_performance.py            # размеры из задания (нужно много памяти)
    python homework_performance.py --quick    # уменьшенные размеры для слабых компьютеров

Скрипт сам определяет, есть ли GPU: CUDA (видеокарта NVIDIA) или MPS (GPU в Mac с
чипами Apple). Если GPU нет, замеры идут только на CPU, и это не ошибка.
Результаты и анализ сохраняются в performance_results.md.
"""
import argparse
import time

import torch

# Размеры из задания: (количество матриц, строки, столбцы)
SIZES = [(64, 1024, 1024), (128, 512, 512), (256, 256, 256)]
# Уменьшенные размеры для быстрого запуска
QUICK_SIZES = [(8, 512, 512), (16, 256, 256), (32, 128, 128)]

BYTES_PER_FLOAT32 = 4

# Операции для сравнения: название -> функция от двух тензоров
OPERATIONS = {
    "Матричное умножение": lambda a, b: torch.matmul(a, b),
    "Сложение": lambda a, b: a + b,
    "Умножение (поэлемент.)": lambda a, b: a * b,
    "Транспонирование": lambda a, b: a.transpose(-1, -2).contiguous(),
    "Сумма всех элементов": lambda a, b: a.sum(),
}


# 3.1 Подготовка данных
def make_data(shape):
    """Создаёт две матрицы заданной формы со случайными числами (float32)."""
    if len(shape) != 3 or shape[1] != shape[2]:
        raise ValueError(f"Нужна форма (batch, n, n), а получена {shape}")
    return torch.rand(shape), torch.rand(shape)


def size_in_mb(shape):
    """Размер одного тензора float32 в мегабайтах."""
    elements = shape[0] * shape[1] * shape[2]
    return elements * BYTES_PER_FLOAT32 / (1024 ** 2)


# Определение устройства
def get_gpu_device():
    """Возвращает (device, название) или (None, None), если GPU нет."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "CUDA"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps"), "MPS"
    return None, None


def synchronize(device):
    """Ждёт, пока GPU закончит все операции (они выполняются асинхронно)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


# 3.2 Измерение времени
def measure_cpu_ms(function, repeats=3, warmup=1):
    """Среднее время выполнения function на CPU в миллисекундах (time.time)."""
    for _ in range(warmup):                       # прогрев: первый вызов обычно медленнее
        function()
    times = []
    for _ in range(repeats):
        start = time.time()
        function()
        times.append((time.time() - start) * 1000)
    return sum(times) / len(times)


def measure_gpu_ms(function, device, repeats=3, warmup=1):
    """Среднее время выполнения function на GPU в миллисекундах.

    Для CUDA используются torch.cuda.Event (точное измерение на самом GPU).
    Для MPS таких событий нет, поэтому берётся time.time() с синхронизацией.
    """
    for _ in range(warmup):
        function()
    synchronize(device)
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            function()
            end_event.record()
            torch.cuda.synchronize()
            times.append(start_event.elapsed_time(end_event))
        else:
            synchronize(device)
            start = time.time()
            function()
            synchronize(device)
            times.append((time.time() - start) * 1000)
    return sum(times) / len(times)


def measure_transfer_ms(tensor, device, repeats=3):
    """Среднее время копирования тензора с CPU на GPU в миллисекундах."""
    tensor.to(device)                             # прогрев (инициализация GPU)
    synchronize(device)
    times = []
    for _ in range(repeats):
        start = time.time()
        tensor.to(device)
        synchronize(device)
        times.append((time.time() - start) * 1000)
    return sum(times) / len(times)


# 3.3 Сравнение операций
def benchmark_shape(shape, device, repeats=3, warmup=1):
    """Замеряет все операции для одной формы.

    Возвращает (список результатов по операциям, время передачи на GPU или None).
    """
    a_cpu, b_cpu = make_data(shape)
    if device is not None:
        a_gpu, b_gpu = a_cpu.to(device), b_cpu.to(device)

    rows = []
    for name, operation in OPERATIONS.items():
        cpu_ms = measure_cpu_ms(lambda: operation(a_cpu, b_cpu), repeats, warmup)
        gpu_ms = None
        speedup = None
        if device is not None:
            gpu_ms = measure_gpu_ms(lambda: operation(a_gpu, b_gpu), device, repeats, warmup)
            speedup = cpu_ms / gpu_ms if gpu_ms > 0 else None
        rows.append({"shape": shape, "operation": name,
                     "cpu_ms": cpu_ms, "gpu_ms": gpu_ms, "speedup": speedup})

    transfer_ms = measure_transfer_ms(a_cpu, device, repeats) if device is not None else None
    return rows, transfer_ms


def format_table(rows, gpu_name):
    """Форматирует результаты для одной формы в виде текстовой таблицы."""
    gpu_header = f"GPU {gpu_name} (мс)" if gpu_name else "GPU (мс)"
    lines = [f"{'Операция':<24}| {'CPU (мс)':>10} | {gpu_header:>16} | {'Ускорение':>10}"]
    lines.append("-" * len(lines[0]))
    for row in rows:
        gpu_text = f"{row['gpu_ms']:.1f}" if row["gpu_ms"] is not None else "н/д"
        speedup_text = f"{row['speedup']:.1f}x" if row["speedup"] is not None else "н/д"
        lines.append(f"{row['operation']:<24}| {row['cpu_ms']:>10.1f} | {gpu_text:>16} | {speedup_text:>10}")
    return "\n".join(lines)


# 3.4 Анализ результатов
def count_elements(shape):
    """Общее число элементов в тензоре формы (batch, n, n)."""
    return shape[0] * shape[1] * shape[2]


def size_trend_lines(rows):
    """Строки анализа: как ускорение меняется с размером тензора (по реальным данным)."""
    by_operation = {}
    for row in rows:
        if row["speedup"] is not None:
            by_operation.setdefault(row["operation"], []).append(row)

    details, growing, not_growing = [], [], []
    for name, items in by_operation.items():
        if len(items) < 2:
            continue
        items = sorted(items, key=lambda r: count_elements(r["shape"]))   # от малого к большому
        speedups = [r["speedup"] for r in items]
        details.append(f"{name}: " + " -> ".join(f"{value:.1f}x" for value in speedups))
        # «растёт», если каждое следующее значение не меньше предыдущего (с запасом 5 % на шум)
        if all(b >= a * 0.95 for a, b in zip(speedups, speedups[1:])):
            growing.append(name)
        else:
            not_growing.append(name)

    lines = []
    if details:
        lines.append("Ускорение от самого маленького тензора к самому большому: "
                     + "; ".join(details) + ".")
    if growing:
        lines.append("Ускорение растёт вместе с размером тензора для: " + ", ".join(growing) + ".")
    if not_growing:
        lines.append("Ускорение НЕ растёт стабильно с размером тензора для: "
                     + ", ".join(not_growing) + ".")
    return lines


def build_analysis(all_rows, transfers, gpu_name):
    """Короткий анализ, посчитанный по реальным результатам замеров."""
    lines = []
    if gpu_name is None:
        slowest = max(all_rows, key=lambda r: r["cpu_ms"])
        lines.append("GPU на этом компьютере недоступен (нет CUDA и нет MPS), поэтому "
                     "сравнить CPU и GPU нельзя. Замеры выполнены только на CPU.")
        lines.append(f"Самая долгая операция на CPU: «{slowest['operation']}» "
                     f"на форме {slowest['shape']} ({slowest['cpu_ms']:.1f} мс).")
        return lines

    with_speedup = [r for r in all_rows if r["speedup"] is not None]
    best = max(with_speedup, key=lambda r: r["speedup"])
    worst = min(with_speedup, key=lambda r: r["speedup"])
    if best["speedup"] < 1:
        lines.append(f"Ни одна операция не ускорилась на GPU. Лучший результат: «{best['operation']}» "
                     f"на форме {best['shape']} ({best['speedup']:.1f}x, то есть GPU медленнее CPU).")
    else:
        lines.append(f"Наибольшее ускорение: «{best['operation']}» на форме {best['shape']} "
                     f"({best['speedup']:.1f}x).")
    lines.append(f"Наименьшее ускорение: «{worst['operation']}» на форме {worst['shape']} "
                 f"({worst['speedup']:.1f}x)" + (" (GPU медленнее CPU)." if worst["speedup"] < 1 else "."))

    slower = [r for r in with_speedup if r["speedup"] < 1]
    if slower:
        names = sorted({r["operation"] for r in slower})
        lines.append("На GPU медленнее CPU оказались: " + ", ".join(names) + ".")
    else:
        lines.append("Ни одна операция не оказалась медленнее на GPU.")

    lines.extend(size_trend_lines(with_speedup))
    lines.append("Замечание: в заданных размерах одновременно меняются и число матриц, и размер "
                 "каждой матрицы, поэтому влияние «размера» нельзя полностью отделить от других факторов.")

    for shape, ms in transfers.items():
        matching = [r for r in with_speedup if r["shape"] == shape and r["operation"] == "Сложение"]
        if matching:
            lines.append(f"Передача одного тензора {shape} на GPU занимает {ms:.1f} мс, "
                         f"а сложение на GPU {matching[0]['gpu_ms']:.1f} мс.")
    return lines


THEORY = """\
## Объяснения (теория)

Конкретные числа и выводы смотрите в таблицах и в разделе «Анализ по полученным
данным» выше: они посчитаны по замерам. Ниже общие принципы, которые помогают их
объяснить.

**Какие операции выигрывают от GPU.** GPU выигрывает там, где работу можно разбить
на много независимых частей и выполнять одновременно: матричное умножение,
поэлементные операции и копирование больших массивов. Насколько именно выиграет
каждая операция, зависит от компьютера. Например, у процессоров Apple есть
оптимизированные библиотеки для матричных вычислений, поэтому CPU там очень силён
в матричном умножении, и выигрыш GPU может быть скромнее, чем у видеокарт NVIDIA.

**Почему операция может быть медленнее на GPU.** Запуск операции на GPU имеет
постоянные накладные расходы (запуск ядра, синхронизация). Если данных мало или
операция очень простая, вычисление занимает меньше времени, чем эти накладные
расходы. Плохо подходят операции, которые трудно распараллелить: например, сумма
всех элементов требует объединить результаты многих потоков в одно число.

**Как размер влияет на ускорение.** Чем больше данных, тем меньше доля накладных
расходов, поэтому ускорение обычно растёт с размером. Но так бывает не всегда:
на ускорение влияют и сама операция, и устройство, и то, как работает библиотека
на CPU. Проверяйте по строке «Ускорение растёт…» в анализе выше.

**Передача данных между CPU и GPU.** Копирование тензора на GPU занимает время,
сравнимое со временем самой простой операции (см. замеры выше). Если тензор
отправили на GPU ради одной быстрой операции, копирование может съесть весь выигрыш.
Поэтому данные переносят на GPU один раз и выполняют на нём много операций подряд.
У Mac с чипами Apple память общая для CPU и GPU, но копирование всё равно занимает
измеримое время.

**Точность замеров.** Каждое значение это среднее по нескольким запускам. Отдельные
значения меньше нескольких миллисекунд могут быть шумными, поэтому одиночные
выбросы стоит перепроверять повторным запуском с большим числом повторов.
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Сравнение CPU и GPU в PyTorch")
    parser.add_argument("--quick", action="store_true", help="уменьшенные размеры матриц")
    parser.add_argument("--repeats", type=int, default=5, help="число повторов замера")
    parser.add_argument("--warmup", type=int, default=1, help="число прогревочных запусков")
    parser.add_argument("--output", default="performance_results.md", help="файл для результатов")
    args = parser.parse_args(argv)

    if args.repeats < 1 or args.warmup < 0:
        raise ValueError("repeats должен быть не меньше 1, warmup не меньше 0")

    torch.manual_seed(0)
    sizes = QUICK_SIZES if args.quick else SIZES
    device, gpu_name = get_gpu_device()

    print(f"PyTorch {torch.__version__}")
    print(f"GPU: {gpu_name if gpu_name else 'не найден, замеры только на CPU'}")
    memory_factor = 2 if device is None else 4        # две матрицы на CPU (и две копии на GPU)
    for shape in sizes:
        print(f"  размер {shape}: один тензор {size_in_mb(shape):.0f} МБ, "
              f"нужно примерно {size_in_mb(shape) * memory_factor:.0f} МБ памяти")

    all_rows = []
    transfers = {}
    report = ["# Результаты сравнения производительности", ""]
    report.append(f"PyTorch {torch.__version__}, GPU: {gpu_name if gpu_name else 'нет'}")
    report.append(f"Повторов на каждый замер: {args.repeats}, прогревочных запусков: {args.warmup}")

    for shape in sizes:
        print(f"\nРазмер {shape}")
        rows, transfer_ms = benchmark_shape(shape, device, args.repeats, args.warmup)
        all_rows.extend(rows)
        table = format_table(rows, gpu_name)
        print(table)
        report += ["", f"## Размер {shape}", "", "```", table, "```"]
        if transfer_ms is not None:
            transfers[shape] = transfer_ms
            line = f"Передача одного тензора CPU -> GPU: {transfer_ms:.1f} мс"
            print(line)
            report += ["", line]

    analysis = build_analysis(all_rows, transfers, gpu_name)
    print("\nАнализ по полученным данным:")
    for line in analysis:
        print(" -", line)
    report += ["", "## Анализ по полученным данным", ""] + [f"- {line}" for line in analysis]
    report += ["", THEORY]

    with open(args.output, "w", encoding="utf-8") as file:
        file.write("\n".join(report))
    print(f"\nРезультаты сохранены в {args.output}")


if __name__ == "__main__":
    main()