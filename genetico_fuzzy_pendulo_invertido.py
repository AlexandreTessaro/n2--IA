import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from deap import base, creator, tools, algorithms
import random

angle = ctrl.Antecedent(np.arange(-90, 91, 1), 'angle')
angular_velocity = ctrl.Antecedent(np.arange(-100, 101, 1), 'angular_velocity')
force = ctrl.Consequent(np.arange(-100, 101, 1), 'force')

def initialize_membership_functions(chromosome):
    angle['left'] = fuzz.trapmf(angle.universe, sorted(chromosome[0:4]))
    angle['center'] = fuzz.trimf(angle.universe, sorted(chromosome[4:7]))
    angle['right'] = fuzz.trapmf(angle.universe, sorted(chromosome[7:11]))
    angular_velocity['left'] = fuzz.trapmf(angular_velocity.universe, sorted(chromosome[11:15]))
    angular_velocity['zero'] = fuzz.trimf(angular_velocity.universe, sorted(chromosome[15:18]))
    angular_velocity['right'] = fuzz.trapmf(angular_velocity.universe, sorted(chromosome[18:22]))
    force['strong_left'] = fuzz.trapmf(force.universe, sorted(chromosome[22:26]))
    force['light_left'] = fuzz.trimf(force.universe, sorted(chromosome[26:29]))
    force['neutral'] = fuzz.trimf(force.universe, sorted(chromosome[29:32]))
    force['light_right'] = fuzz.trimf(force.universe, sorted(chromosome[32:35]))
    force['strong_right'] = fuzz.trapmf(force.universe, sorted(chromosome[35:39]))

def evaluate_fis(chromosome):
    initialize_membership_functions(chromosome)
    
    rules = [
        ctrl.Rule(angle['left'] & angular_velocity['left'], force['strong_left']),
        ctrl.Rule(angle['left'] & angular_velocity['zero'], force['light_left']),
        ctrl.Rule(angle['left'] & angular_velocity['right'], force['neutral']),
        ctrl.Rule(angle['center'] & angular_velocity['left'], force['light_left']),
        ctrl.Rule(angle['center'] & angular_velocity['zero'], force['neutral']),
        ctrl.Rule(angle['center'] & angular_velocity['right'], force['light_right']),
        ctrl.Rule(angle['right'] & angular_velocity['left'], force['neutral']),
        ctrl.Rule(angle['right'] & angular_velocity['zero'], force['light_right']),
        ctrl.Rule(angle['right'] & angular_velocity['right'], force['strong_right'])
    ]
    
    try:
        control_system = ctrl.ControlSystem(rules)
        fis_simulation = ctrl.ControlSystemSimulation(control_system)
    except Exception as e:
        print(f"Erro ao configurar o sistema de controle: {e}")
        return (float('inf'),)  
    
    test_cases = [
        (-30, 20), (45, -10), (0, 0), (15, 5), (-45, -30)
    ]
    
    errors = []
    for angle_val, ang_vel_val in test_cases:
        fis_simulation.input['angle'] = angle_val
        fis_simulation.input['angular_velocity'] = ang_vel_val
        
        try:
            fis_simulation.compute()
            output_force = fis_simulation.output.get('force', None)
            
            if output_force is None:
                raise ValueError("A variável de saída 'force' não foi gerada corretamente.")
            
            errors.append(abs(output_force)) 
        except Exception as e:
            print(f"Erro ao calcular a saída para ângulo {angle_val} e velocidade angular {ang_vel_val}: {e}")
            return (float('inf'),)  

    return sum(errors),  

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -100, 100)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, 39)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evaluate_fis)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def run_genetic_algorithm():
    population = toolbox.population(n=50)
    NGEN = 20
    for gen in range(NGEN):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.5, mutpb=0.2)
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        print(f"Generation {gen+1} complete")
    best_individual = tools.selBest(population, k=1)[0]
    print("Melhor indivíduo:", best_individual)
    print("Fitness:", best_individual.fitness.values[0])
    return best_individual

best_chromosome = run_genetic_algorithm()

initialize_membership_functions(best_chromosome)
control_system = ctrl.ControlSystem([
    ctrl.Rule(angle['left'] & angular_velocity['left'], force['strong_left']),
    ctrl.Rule(angle['left'] & angular_velocity['zero'], force['light_left']),
    ctrl.Rule(angle['left'] & angular_velocity['right'], force['neutral']),
    ctrl.Rule(angle['center'] & angular_velocity['left'], force['light_left']),
    ctrl.Rule(angle['center'] & angular_velocity['zero'], force['neutral']),
    ctrl.Rule(angle['center'] & angular_velocity['right'], force['light_right']),
    ctrl.Rule(angle['right'] & angular_velocity['left'], force['neutral']),
    ctrl.Rule(angle['right'] & angular_velocity['zero'], force['light_right']),
    ctrl.Rule(angle['right'] & angular_velocity['right'], force['strong_right'])
])

fis_simulation = ctrl.ControlSystemSimulation(control_system)

def test_optimized_fis():
    test_cases = [
        (-30, 20), (45, -10), (0, 0), (15, 5), (-45, -30)
    ]
    for angle_val, ang_vel_val in test_cases:
        fis_simulation.input['angle'] = angle_val
        fis_simulation.input['angular_velocity'] = ang_vel_val
        fis_simulation.compute()
        print(f"Ângulo: {angle_val}, Velocidade Angular: {ang_vel_val} -> Força aplicada: {fis_simulation.output['force']:.2f}")

test_optimized_fis()
