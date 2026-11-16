"""Домашнее задание 1, задание 1: создание и манипуляции с тензорами."""
import torch


def _check_tensor(value, name, ndim=None):
    """Проверяет, что value это тензор (и, если нужно, нужной размерности)."""
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} должен быть torch.Tensor, а получен {type(value).__name__}")
    if ndim is not None and value.dim() != ndim:
        raise ValueError(f"{name} должен иметь {ndim} измерения(й), а имеет {value.dim()}")


# 1.1 Создание тензоров
def create_tensors():
    """Создаёт четыре тензора из задания 1.1 и возвращает их в словаре."""
    random_tensor = torch.rand(3, 4)                  # 3x4, случайные числа от 0 до 1
    zeros_tensor = torch.zeros(2, 3, 4)               # 2x3x4 из нулей
    ones_tensor = torch.ones(5, 5)                    # 5x5 из единиц
    range_tensor = torch.arange(16).reshape(4, 4)     # 4x4, числа от 0 до 15
    return {
        "random_3x4": random_tensor,
        "zeros_2x3x4": zeros_tensor,
        "ones_5x5": ones_tensor,
        "range_4x4": range_tensor,
    }


# 1.2 Операции с тензорами
def tensor_operations(a, b):
    """Операции задания 1.2 для A размера (3, 4) и B размера (4, 3).

    Возвращает словарь: транспонированное A, матричное произведение A и B,
    поэлементное произведение A и B^T, сумма всех элементов A.
    """
    _check_tensor(a, "A", ndim=2)
    _check_tensor(b, "B", ndim=2)
    if a.shape[1] != b.shape[0]:
        raise ValueError(
            f"Матричное умножение невозможно: A имеет форму {tuple(a.shape)}, B {tuple(b.shape)}"
        )
    if a.shape != b.T.shape:
        raise ValueError(
            f"Поэлементное умножение A и B^T невозможно: формы {tuple(a.shape)} и {tuple(b.T.shape)}"
        )
    return {
        "A_transposed": a.T,                  # транспонирование A
        "A_matmul_B": a @ b,                  # матричное умножение (3x4) @ (4x3) = (3x3)
        "A_times_B_transposed": a * b.T,      # поэлементно: (3x4) * (3x4)
        "A_sum": a.sum(),                     # сумма всех элементов A
    }


# 1.3 Индексация и срезы
def indexing_examples(x):
    """Срезы задания 1.3 для тензора 5x5x5.

    Тензор считаем набором из 5 матриц 5x5. «Строка», «столбец» и «подматрица»
    берутся из первой матрицы x[0], а чётные индексы берутся по всем трём осям.
    """
    _check_tensor(x, "x", ndim=3)
    if min(x.shape) < 4:
        raise ValueError("Каждое измерение тензора должно быть не меньше 4")
    first_matrix = x[0]                                    # первая матрица (5x5)
    rows, cols = first_matrix.shape
    r, c = rows // 2, cols // 2                            # индекс центра
    return {
        "first_row": first_matrix[0, :],                   # первая строка
        "last_column": first_matrix[:, -1],                # последний столбец
        "center_2x2": first_matrix[r - 1:r + 1, c - 1:c + 1],   # блок 2x2 около центра
        "even_indices": x[::2, ::2, ::2],                  # индексы 0, 2, 4 по каждой оси
    }


# 1.4 Работа с формами
RESHAPE_TARGETS = [(2, 12), (3, 8), (4, 6), (2, 3, 4), (2, 2, 2, 3)]


def reshape_examples(x):
    """Преобразует тензор из 24 элементов во все формы задания 1.4."""
    _check_tensor(x, "x", ndim=1)
    if x.numel() != 24:
        raise ValueError(f"Нужен тензор из 24 элементов, а в нём {x.numel()}")
    return {shape: x.reshape(shape) for shape in RESHAPE_TARGETS}


def main():
    torch.manual_seed(0)

    print("=== 1.1 Создание тензоров ===")
    for name, tensor in create_tensors().items():
        print(f"{name}: shape={tuple(tensor.shape)}")
        print(tensor)

    print("\n=== 1.2 Операции с тензорами ===")
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    results = tensor_operations(a, b)
    print("A =\n", a)
    print("B =\n", b)
    for name, value in results.items():
        print(f"{name}:\n{value}")

    print("\n=== 1.3 Индексация и срезы ===")
    x = torch.arange(125).reshape(5, 5, 5)
    for name, value in indexing_examples(x).items():
        print(f"{name}: shape={tuple(value.shape)}")
        print(value)

    print("\n=== 1.4 Работа с формами ===")
    for shape, value in reshape_examples(torch.arange(24)).items():
        print(f"{shape}: shape={tuple(value.shape)}")


if __name__ == "__main__":
    main()