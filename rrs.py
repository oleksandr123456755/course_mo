import time
import numpy as np
start_time = time.time()

def call_counter(func):
    def wrapper(*args, **kwargs):
        wrapper.counter += 1
        return func(*args, **kwargs)
    wrapper.counter = 0
    return wrapper

@call_counter
def rosenbrock(x):
    return sum(100.0*(x[1:] - x[:-1]**2.0)**2.0 + (1 - x[:-1])**2.0)


# Параметри методу
num_iterations = 10000 # Кількість ітерацій
sample_size = 100  # Розмір вибірки на кожній ітерації
search_range = [-1, 1]  # Діапазон пошуку

# Вибір початкових параметрів
dimension = 2  # Розмірність простору
best_solution = None
best_value = float('inf')

# Реалізація базового методу RRS
for iteration in range(num_iterations):
    samples = np.random.uniform(search_range[0], search_range[1], (sample_size, dimension))
    for sample in samples:
        value = rosenbrock(sample)
        if value < best_value:
            best_value = value
            best_solution = sample

print(f"\nНайкраще знайдене рішення: {best_solution}")
print(f"Значення функції у цій точці: {best_value}")
print(f"Кількість обчислення функції: {rosenbrock.counter}")
end_time = time.time()

execution_time = end_time - start_time
print(f"Час виконання: {execution_time} секунд")


