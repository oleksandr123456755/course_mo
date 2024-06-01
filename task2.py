import numpy as np
from scipy.optimize import minimize
import time
start_time = time.time()

def call_counter(func):
    def wrapper(*args, **kwargs):
        wrapper.counter += 1
        return func(*args, **kwargs)
    wrapper.counter = 0
    return wrapper

@call_counter
def rosenbrock(x):
    return sum(100.0 * (x[1:] - x[:-1] ** 2.0) ** 2.0 + (1 - x[:-1]) ** 2.0)

# Генерація випадкової початкової точки
def generate_random_point(dimension, search_range):
    return np.random.uniform(search_range[0], search_range[1], dimension)

# Функція для перевірки, чи задовольняє рішення всім обмеженням
def is_feasible(x, constraints):
    return all(constr['fun'](x) >= 0 for constr in constraints)

# Параметри методу
num_iterations = 1000  # Кількість ітерацій
sample_size = 50  # Розмір вибірки на кожній ітерації
search_range = [-5, 5]  # Діапазон пошуку
dimension = 2  # Розмірність простору

# Види допустимої області
# Випукла область
constraints_convex = [{'type': 'ineq', 'fun': lambda x: 0.5 - (x[0] + x[1])},
                      {'type': 'ineq', 'fun': lambda x: x[0] - 1.5},
                      {'type': 'ineq', 'fun': lambda x: x[1] - 1.5}]

# Невипукла область
constraints_nonconvex = [{'type': 'ineq', 'fun': lambda x: np.sin(x[0]) + np.cos(x[1]) - 0.5},
                         {'type': 'ineq', 'fun': lambda x: x[0] - 1.5},
                         {'type': 'ineq', 'fun': lambda x: x[1] - 1.5}]

# Лінійні обмеження
constraints_linear = [{'type': 'ineq', 'fun': lambda x: x[0] + x[1] - 1},
                      {'type': 'ineq', 'fun': lambda x: 1.5 - x[0]},
                      {'type': 'ineq', 'fun': lambda x: 1.5 - x[1]}]

# Метод одновимірного пошуку
method1D = 'Powell'  # 'Powell' або 'Golden'

# Точність метода одновимірного пошуку
tol1D = 1e-3

# Значення параметру в алгоритмі Свена
sven_param = 1e-6

# Функція для виконання оптимізації з різними обмеженнями
def optimize_with_constraints(constraints, description):
    print(f"\nRRS для умовної оптимізації ({description}):")
    best_solution = None
    best_value = float('inf')
    
    for iteration in range(num_iterations):
        sample = generate_random_point(dimension, search_range)
        if is_feasible(sample, constraints):
            result = minimize(rosenbrock, sample, method=method1D, tol=tol1D, options={'xtol': sven_param})
            value = result.fun
            if value < best_value:
                best_value = value
                best_solution = result.x
    
    if best_solution is not None:
        print(f"Найкраще знайдене рішення: {best_solution}")
        print(f"Значення функції у цій точці: {best_value}")
    else:
        print("Не вдалося знайти допустиме рішення.")

# Запуск оптимізації для різних видів допустимої області
optimize_with_constraints(constraints_convex, "випукла область")
optimize_with_constraints(constraints_nonconvex, "невипукла область")
optimize_with_constraints(constraints_linear, "лінійні обмеження")
print(f"Кількість обчислення функції: {rosenbrock.counter}")
end_time = time.time()

execution_time = end_time - start_time
print(f"Час виконання: {execution_time} секунд")