import numpy as np
import time
start_time = time.time()

def call_counter(func):
    def wrapper(*args, **kwargs):
        wrapper.counter += 1
        return func(*args, **kwargs)
    wrapper.counter = 0
    return wrapper

# Функція Розенброка
@call_counter
def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)

# Параметри методу
num_iterations = 1000  # Кількість ітерацій
sample_size = 100  # Розмір вибірки на кожній ітерації
initial_search_range = [-5, 5]  # Початковий діапазон пошуку
decay_factor = 0.999  # Фактор зменшення діапазону пошуку

# Вибір початкових параметрів
dimension = 2  # Розмірність простору
best_solution = None
best_value = float('inf')
search_range = initial_search_range.copy()

# Реалізація адаптивного RRS
for iteration in range(num_iterations):
    samples = np.random.uniform(search_range[0], search_range[1], (sample_size, dimension))
    for sample in samples:
        value = rosenbrock(sample)
        if value < best_value:
            best_value = value
            best_solution = sample
            # Зменшення діапазону пошуку
            search_range = [x * decay_factor for x in search_range]

print(f"Найкраще знайдене рішення: {best_solution}")
print(f"Значення функції у цій точці: {best_value}")
print(f"Кількість обчислення функції: {rosenbrock.counter}")
end_time = time.time()
execution_time = end_time - start_time
print(f"Час виконання: {execution_time} секунд")
