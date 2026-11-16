"""Домашнее задание 1, задание 2: автоматическое дифференцирование."""
import torch


# 2.1 Простые вычисления с градиентами
def task_2_1(x_value=1.0, y_value=2.0, z_value=3.0):
    """f(x, y, z) = x^2 + y^2 + z^2 + 2*x*y*z.

    Возвращает значение функции, градиенты от autograd и аналитические градиенты:
        df/dx = 2x + 2yz,  df/dy = 2y + 2xz,  df/dz = 2z + 2xy.
    """
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y_value, dtype=torch.float64, requires_grad=True)
    z = torch.tensor(z_value, dtype=torch.float64, requires_grad=True)

    f = x ** 2 + y ** 2 + z ** 2 + 2 * x * y * z
    f.backward()                                      # считаем градиенты

    autograd_grads = (x.grad.item(), y.grad.item(), z.grad.item())
    analytic_grads = (
        2 * x_value + 2 * y_value * z_value,
        2 * y_value + 2 * x_value * z_value,
        2 * z_value + 2 * x_value * y_value,
    )
    return f.item(), autograd_grads, analytic_grads


# 2.2 Градиент функции потерь
def mse_loss(y_pred, y_true):
    """MSE = (1/n) * сумма (y_pred - y_true)^2."""
    if not isinstance(y_pred, torch.Tensor) or not isinstance(y_true, torch.Tensor):
        raise TypeError("y_pred и y_true должны быть torch.Tensor")
    if y_pred.shape != y_true.shape:
        raise ValueError(
            f"Формы не совпадают: {tuple(y_pred.shape)} и {tuple(y_true.shape)}"
        )
    return torch.mean((y_pred - y_true) ** 2)


def task_2_2(x_data=(1.0, 2.0, 3.0, 4.0), y_data=(3.0, 5.0, 7.0, 9.0), w_value=0.5, b_value=0.0):
    """Градиенты MSE по w и b для линейной модели y_pred = w * x + b.

    Аналитически:
        dMSE/dw = (2/n) * сумма (y_pred - y_true) * x
        dMSE/db = (2/n) * сумма (y_pred - y_true)
    """
    x = torch.tensor(x_data, dtype=torch.float64)
    y_true = torch.tensor(y_data, dtype=torch.float64)
    w = torch.tensor(w_value, dtype=torch.float64, requires_grad=True)
    b = torch.tensor(b_value, dtype=torch.float64, requires_grad=True)

    y_pred = w * x + b
    loss = mse_loss(y_pred, y_true)
    loss.backward()

    # Аналитическая проверка (без autograd)
    difference = w.detach() * x + b.detach() - y_true
    n = x.numel()
    analytic_w = (2.0 / n) * torch.sum(difference * x)
    analytic_b = (2.0 / n) * torch.sum(difference)
    return loss.item(), (w.grad.item(), b.grad.item()), (analytic_w.item(), analytic_b.item())


# ---------------------------------------------------------------------------
# 2.3 Цепное правило
# ---------------------------------------------------------------------------
def task_2_3(x_value=2.0):
    """f(x) = sin(x^2 + 1).

    Цепное правило: пусть u = x^2 + 1, тогда df/dx = cos(u) * du/dx = cos(x^2 + 1) * 2x.
    Возвращает значение функции, градиент от torch.autograd.grad, градиент от
    backward() и аналитический градиент.
    """
    x = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    f = torch.sin(x ** 2 + 1)
    (grad_autograd,) = torch.autograd.grad(f, x)      # градиент через autograd.grad

    x_second = torch.tensor(x_value, dtype=torch.float64, requires_grad=True)
    f_second = torch.sin(x_second ** 2 + 1)
    f_second.backward()                               # то же самое через backward()

    analytic = torch.cos(torch.tensor(x_value, dtype=torch.float64) ** 2 + 1) * 2 * x_value
    return f.item(), grad_autograd.item(), x_second.grad.item(), analytic.item()


def main():
    print("=== 2.1 f = x^2 + y^2 + z^2 + 2xyz в точке (1, 2, 3) ===")
    value, autograd_grads, analytic_grads = task_2_1()
    print("f =", value)
    print("градиенты autograd (x, y, z):  ", autograd_grads)
    print("аналитические градиенты:       ", analytic_grads)

    print("\n=== 2.2 MSE и градиенты по w и b ===")
    loss, autograd_grads, analytic_grads = task_2_2()
    print("MSE =", loss)
    print("градиенты autograd (w, b):", autograd_grads)
    print("аналитические градиенты:  ", analytic_grads)

    print("\n=== 2.3 f(x) = sin(x^2 + 1) в точке x = 2 ===")
    value, grad_ag, grad_bw, analytic = task_2_3()
    print("f =", value)
    print("torch.autograd.grad:", grad_ag)
    print("backward():         ", grad_bw)
    print("аналитически:       ", analytic)


if __name__ == "__main__":
    main()