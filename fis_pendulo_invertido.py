import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

angle = ctrl.Antecedent(np.arange(-90, 91, 1), 'angle')
angular_velocity = ctrl.Antecedent(np.arange(-100, 101, 1), 'angular_velocity')
position = ctrl.Antecedent(np.arange(-10, 11, 1), 'position')
velocity = ctrl.Antecedent(np.arange(-5, 6, 1), 'velocity')
force = ctrl.Consequent(np.arange(-100, 101, 1), 'force')

angle['left'] = fuzz.trapmf(angle.universe, [-90, -90, -45, 0])
angle['center'] = fuzz.trimf(angle.universe, [-45, 0, 45])
angle['right'] = fuzz.trapmf(angle.universe, [0, 45, 90, 90])

angular_velocity['left'] = fuzz.trapmf(angular_velocity.universe, [-100, -100, -50, 0])
angular_velocity['zero'] = fuzz.trimf(angular_velocity.universe, [-50, 0, 50])
angular_velocity['right'] = fuzz.trapmf(angular_velocity.universe, [0, 50, 100, 100])

position['left'] = fuzz.trapmf(position.universe, [-10, -10, -5, 0])
position['center'] = fuzz.trimf(position.universe, [-5, 0, 5])
position['right'] = fuzz.trapmf(position.universe, [0, 5, 10, 10])

velocity['left'] = fuzz.trapmf(velocity.universe, [-5, -5, -2, 0])
velocity['zero'] = fuzz.trimf(velocity.universe, [-2, 0, 2])
velocity['right'] = fuzz.trapmf(velocity.universe, [0, 2, 5, 5])

force['strong_left'] = fuzz.trapmf(force.universe, [-100, -100, -50, -25])
force['light_left'] = fuzz.trimf(force.universe, [-50, -25, 0])
force['neutral'] = fuzz.trimf(force.universe, [-25, 0, 25])
force['light_right'] = fuzz.trimf(force.universe, [0, 25, 50])
force['strong_right'] = fuzz.trapmf(force.universe, [25, 50, 100, 100])

rules = [
    ctrl.Rule(angle['left'] & angular_velocity['left'], force['strong_left']),
    ctrl.Rule(angle['left'] & angular_velocity['zero'], force['light_left']),
    ctrl.Rule(angle['left'] & angular_velocity['right'], force['neutral']),
    ctrl.Rule(angle['center'] & angular_velocity['left'], force['light_left']),
    ctrl.Rule(angle['center'] & angular_velocity['zero'], force['neutral']),
    ctrl.Rule(angle['center'] & angular_velocity['right'], force['light_right']),
    ctrl.Rule(angle['right'] & angular_velocity['left'], force['neutral']),
    ctrl.Rule(angle['right'] & angular_velocity['zero'], force['light_right']),
    ctrl.Rule(angle['right'] & angular_velocity['right'], force['strong_right']),
    ctrl.Rule(position['left'] & velocity['left'], force['strong_right']),
    ctrl.Rule(position['left'] & velocity['zero'], force['light_right']),
    ctrl.Rule(position['center'] & velocity['left'], force['light_right']),
    ctrl.Rule(position['center'] & velocity['zero'], force['neutral']),
    ctrl.Rule(position['center'] & velocity['right'], force['light_left']),
    ctrl.Rule(position['right'] & velocity['zero'], force['light_left']),
    ctrl.Rule(position['right'] & velocity['right'], force['strong_left'])
]

control_system = ctrl.ControlSystem(rules)
fis_simulation = ctrl.ControlSystemSimulation(control_system)

def test_fis(angle_value, angular_velocity_value, position_value, velocity_value):
    fis_simulation.input['angle'] = angle_value
    fis_simulation.input['angular_velocity'] = angular_velocity_value
    fis_simulation.input['position'] = position_value
    fis_simulation.input['velocity'] = velocity_value
    fis_simulation.compute()
    return fis_simulation.output['force']

def run_multiple_tests():
    test_cases = [
        (-30, 20, 0, 0),  
        (45, -10, -5, -2),  
        (0, 0, 0, 0), 
        (15, 5, 5, 1),  
        (-45, -30, -10, -3),  
        (90, 100, 10, 5), 
        (-90, -100, -10, -5),  
    ]

    for i, (angle_value, angular_velocity_value, position_value, velocity_value) in enumerate(test_cases, start=1):
        result = test_fis(angle_value, angular_velocity_value, position_value, velocity_value)
        print(f"Teste {i}: Ângulo = {angle_value}, Velocidade Angular = {angular_velocity_value}, "
              f"Posição = {position_value}, Velocidade = {velocity_value} -> Força aplicada: {result:.2f}")

run_multiple_tests()
