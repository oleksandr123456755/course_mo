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



# Обмеження для умовної оптимізації (локальний мінімум поза допустимою областю)
def constraint1(x):
    return 3 - (x[0] + x[1])  # x[0] + x[1] >= 3

def constraint2(x):
    return x[0] - 10  # x[0] <= 10

def constraint3(x):
    return x[1] - 10  # x[1] <= 10

# Параметри методу
num_iterations = 10  # Кількість ітерацій
sample_size = 3  # Розмір вибірки на кожній ітерації
search_range = [-1, 1]  # Діапазон пошуку

# Вибір початкових параметрів
dimension = 2  # Розмірність простору
best_solution = None
best_value = float('inf')

# Обмеження для умовної оптимізації
constraints = [{'type': 'ineq', 'fun': constraint1},
               {'type': 'ineq', 'fun': constraint2},
               {'type': 'ineq', 'fun': constraint3}]

# Функція для перевірки, чи задовольняє рішення всім обмеженням
def is_feasible(x, constraints):
    return all(constr['fun'](x) >= 0 for constr in constraints)

# Реалізація RRS як методу спуску для умовної оптимізації
print("\nRRS як метод спуску для умовної оптимізації:")
best_solution = None
best_value = float('inf')

for iteration in range(num_iterations):
    samples = np.random.uniform(search_range[0], search_range[1], (sample_size, dimension))
    for sample in samples:
        if is_feasible(sample, constraints):
            result = minimize(rosenbrock, sample, method='Powell')
            value = result.fun
            if value < best_value:
                best_value = value
                best_solution = result.x

if best_solution is not None:
    print(f"Найкраще знайдене рішення: {best_solution}")
    print(f"Значення функції у цій точці: {best_value}")
else:
    print("Не вдалося знайти допустиме рішення.")


print(f"Кількість обчислення функції: {rosenbrock.counter}")
end_time = time.time()

execution_time = end_time - start_time
print(f"Час виконання: {execution_time} секунд")
