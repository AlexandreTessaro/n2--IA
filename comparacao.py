import time
import numpy as np
from sklearn.metrics import mean_squared_error
from fis_pendulo_invertido import test_fis 
from genetico_fuzzy_pendulo_invertido import initialize_membership_functions, run_genetic_algorithm
from neuro_fuzzy_pendulo_invertido import mlp, generate_fuzzy_data

best_chromosome = run_genetic_algorithm()

desired_force = 0

def evaluate_fis_system(test_cases):
    return [test_fis(angle, ang_vel, pos, vel) for angle, ang_vel, pos, vel in test_cases]

def evaluate_genetic_system(test_cases_2d):
    initialize_membership_functions(best_chromosome)
    return [test_fis(angle, ang_vel, 0, 0) for angle, ang_vel in test_cases_2d]

def evaluate_neuro_fuzzy_system(test_cases_2d):
    angle_data, angular_velocity_data = zip(*test_cases_2d)
    neuro_fuzzy_result = []
    for angle, angular_velocity in zip(angle_data, angular_velocity_data):
        result = test_fis(angle, angular_velocity, 0, 0) 
        neuro_fuzzy_result.append(result)
    return neuro_fuzzy_result

test_cases = [
    (-30, 20, 0, 0),  
    (45, -10, -5, -2),  
    (0, 0, 0, 0), 
    (15, 5, 5, 1),  
    (-45, -30, -10, -3),  
    (90, 100, 10, 5), 
    (-90, -100, -10, -5),
]

test_cases_2d = [(angle, ang_vel) for angle, ang_vel, _, _ in test_cases]


start_time = time.time()
fis_results = evaluate_fis_system(test_cases)
fis_time = time.time() - start_time

start_time = time.time()
genetic_results = evaluate_genetic_system(test_cases_2d)
genetic_time = time.time() - start_time

start_time = time.time()
neuro_fuzzy_results = evaluate_neuro_fuzzy_system(test_cases_2d)
neuro_fuzzy_time = time.time() - start_time

reference_values = [desired_force] * len(test_cases)  
fis_mse = mean_squared_error(reference_values, fis_results)
genetic_mse = mean_squared_error(reference_values, genetic_results)
neuro_fuzzy_mse = mean_squared_error(reference_values, neuro_fuzzy_results)


print("Critério               | FIS            | Genético-Fuzzy    | Neuro-Fuzzy")
print("-------------------------------------------------------------------------")
print(f"Tempo de Execução      | {fis_time:.3f} segundos | {genetic_time:.3f} segundos | {neuro_fuzzy_time:.3f} segundos")
print(f"Erro Médio (MSE)       | {fis_mse:.2f}        | {genetic_mse:.2f}          | {neuro_fuzzy_mse:.2f}")

