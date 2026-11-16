"""Тесты к домашнему заданию 1 (запуск: python -m pytest -v)."""
import pytest
import torch

import homework_autograd as ag
import homework_performance as perf
import homework_tensors as ht


# Задание 1: тензоры
def test_create_tensors_shapes_and_values():
    tensors = ht.create_tensors()
    assert tensors["random_3x4"].shape == (3, 4)
    assert tensors["random_3x4"].min() >= 0 and tensors["random_3x4"].max() < 1
    assert tensors["zeros_2x3x4"].shape == (2, 3, 4)
    assert torch.all(tensors["zeros_2x3x4"] == 0)
    assert tensors["ones_5x5"].shape == (5, 5)
    assert torch.all(tensors["ones_5x5"] == 1)
    assert tensors["range_4x4"].shape == (4, 4)
    assert torch.equal(tensors["range_4x4"].flatten(), torch.arange(16))


def make_a_b():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    return a, b


def test_tensor_operations_values():
    a, b = make_a_b()
    result = ht.tensor_operations(a, b)
    assert result["A_transposed"].shape == (4, 3)
    expected_matmul = torch.tensor([[42., 48., 54.], [114., 136., 158.], [186., 224., 262.]])
    assert torch.equal(result["A_matmul_B"], expected_matmul)
    assert result["A_times_B_transposed"].shape == (3, 4)
    assert result["A_times_B_transposed"][0].tolist() == [0.0, 3.0, 12.0, 27.0]
    assert result["A_sum"].item() == 66.0


def test_tensor_operations_errors():
    a, _ = make_a_b()
    with pytest.raises(TypeError):
        ht.tensor_operations([[1, 2]], a)
    with pytest.raises(ValueError):
        ht.tensor_operations(a, torch.zeros(3, 3))       # матрицы нельзя перемножить
    with pytest.raises(ValueError):
        ht.tensor_operations(torch.zeros(3), a)          # не двумерный тензор


def test_indexing_examples():
    x = torch.arange(125).reshape(5, 5, 5)
    result = ht.indexing_examples(x)
    assert result["first_row"].tolist() == [0, 1, 2, 3, 4]
    assert result["last_column"].tolist() == [4, 9, 14, 19, 24]
    assert result["center_2x2"].tolist() == [[6, 7], [11, 12]]
    assert result["even_indices"].shape == (3, 3, 3)
    assert result["even_indices"][1, 1, 1].item() == x[2, 2, 2].item()


def test_indexing_errors():
    with pytest.raises(ValueError):
        ht.indexing_examples(torch.zeros(5, 5))          # не трёхмерный


def test_reshape_examples():
    x = torch.arange(24)
    result = ht.reshape_examples(x)
    assert set(result.keys()) == set(ht.RESHAPE_TARGETS)
    for shape, tensor in result.items():
        assert tensor.shape == shape
        assert torch.equal(tensor.flatten(), x)          # порядок элементов не изменился


def test_reshape_errors():
    with pytest.raises(ValueError):
        ht.reshape_examples(torch.arange(25))
    with pytest.raises(ValueError):
        ht.reshape_examples(torch.arange(24).reshape(4, 6))


# Задание 2: autograd
def test_task_2_1_gradients():
    value, autograd_grads, analytic_grads = ag.task_2_1(1.0, 2.0, 3.0)
    assert value == pytest.approx(26.0)
    assert autograd_grads == pytest.approx((14.0, 10.0, 10.0))
    assert autograd_grads == pytest.approx(analytic_grads)


def test_task_2_1_other_point():
    _, autograd_grads, analytic_grads = ag.task_2_1(-0.5, 1.5, 2.0)
    assert autograd_grads == pytest.approx(analytic_grads)


def test_mse_loss_value_and_errors():
    assert ag.mse_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 4.0])).item() == pytest.approx(2.0)
    with pytest.raises(ValueError):
        ag.mse_loss(torch.zeros(3), torch.zeros(4))
    with pytest.raises(TypeError):
        ag.mse_loss([1.0], torch.zeros(1))


def test_task_2_2_gradients():
    loss, autograd_grads, analytic_grads = ag.task_2_2()
    assert loss == pytest.approx(25.375)
    assert autograd_grads == pytest.approx((-27.5, -9.5))
    assert autograd_grads == pytest.approx(analytic_grads)


def test_task_2_2_perfect_fit_has_zero_gradient():
    # при w=2, b=1 прогноз точно совпадает с данными, значит ошибка и градиенты нулевые
    loss, autograd_grads, _ = ag.task_2_2((1.0, 2.0, 3.0), (3.0, 5.0, 7.0), 2.0, 1.0)
    assert loss == pytest.approx(0.0)
    assert autograd_grads == pytest.approx((0.0, 0.0))


def test_task_2_3_chain_rule():
    value, grad_autograd, grad_backward, analytic = ag.task_2_3(2.0)
    assert value == pytest.approx(-0.9589242746631385)
    assert grad_autograd == pytest.approx(analytic)
    assert grad_backward == pytest.approx(analytic)
    assert analytic == pytest.approx(1.134648741852905)


# Задание 3: производительность
def test_make_data():
    a, b = perf.make_data((2, 8, 8))
    assert a.shape == (2, 8, 8) and b.shape == (2, 8, 8)
    with pytest.raises(ValueError):
        perf.make_data((2, 8, 4))                        # матрицы не квадратные


def test_size_in_mb():
    assert perf.size_in_mb((64, 1024, 1024)) == pytest.approx(256.0)


def test_measure_cpu_ms_returns_positive_number():
    value = perf.measure_cpu_ms(lambda: sum(range(1000)), repeats=2, warmup=1)
    assert value >= 0


def test_benchmark_shape_cpu_only():
    rows, transfer = perf.benchmark_shape((2, 16, 16), None, repeats=1, warmup=0)
    assert transfer is None
    assert [r["operation"] for r in rows] == list(perf.OPERATIONS.keys())
    assert all(r["gpu_ms"] is None and r["speedup"] is None for r in rows)
    assert all(r["cpu_ms"] >= 0 for r in rows)


def test_format_table_and_analysis_without_gpu():
    rows, _ = perf.benchmark_shape((2, 16, 16), None, repeats=1, warmup=0)
    table = perf.format_table(rows, None)
    assert "Матричное умножение" in table and "н/д" in table
    analysis = perf.build_analysis(rows, {}, None)
    assert "недоступен" in analysis[0]


@pytest.mark.skipif(perf.get_gpu_device()[0] is None, reason="GPU недоступен")
def test_benchmark_shape_with_gpu():
    device, _ = perf.get_gpu_device()
    rows, transfer = perf.benchmark_shape((2, 16, 16), device, repeats=1, warmup=0)
    assert transfer is not None and transfer >= 0
    assert all(r["gpu_ms"] is not None for r in rows)


def make_row(shape, operation, cpu_ms, gpu_ms):
    return {"shape": shape, "operation": operation, "cpu_ms": cpu_ms,
            "gpu_ms": gpu_ms, "speedup": cpu_ms / gpu_ms}


def test_build_analysis_when_gpu_is_never_faster():
    rows = [make_row((2, 8, 8), "Матричное умножение", 1.0, 2.0),
            make_row((2, 8, 8), "Сложение", 1.0, 4.0)]
    text = " ".join(perf.build_analysis(rows, {(2, 8, 8): 0.5}, "MPS"))
    assert "Ни одна операция не ускорилась" in text
    assert "Наибольшее ускорение" not in text


def test_build_analysis_when_gpu_is_faster():
    rows = [make_row((2, 8, 8), "Матричное умножение", 10.0, 1.0),
            make_row((2, 8, 8), "Сложение", 1.0, 2.0)]
    text = " ".join(perf.build_analysis(rows, {(2, 8, 8): 0.5}, "CUDA"))
    assert "Наибольшее ускорение: «Матричное умножение»" in text
    assert "10.0x" in text


def test_size_trend_on_real_measurements():
    # реальные замеры (в мс): порядок форм от большего тензора к меньшему, как в таблицах
    data = {
        (64, 1024, 1024): {"Матричное умножение": (239.2, 134.9), "Транспонирование": (67.4, 10.8)},
        (128, 512, 512): {"Матричное умножение": (63.2, 12.8), "Транспонирование": (17.2, 5.5)},
        (256, 256, 256): {"Матричное умножение": (18.4, 4.1), "Транспонирование": (6.9, 7.1)},
    }
    rows = [make_row(shape, op, cpu, gpu)
            for shape, ops in data.items() for op, (cpu, gpu) in ops.items()]
    text = " ".join(perf.size_trend_lines(rows))
    # транспонирование: 1.0x -> 3.1x -> 6.2x растёт; матричное умножение: 4.5x -> 5.0x -> 1.8x нет
    assert "растёт вместе с размером тензора для: Транспонирование" in text
    assert "НЕ растёт стабильно с размером тензора для: Матричное умножение" in text


def test_count_elements():
    assert perf.count_elements((64, 1024, 1024)) == 64 * 1024 * 1024