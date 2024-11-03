import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

angle = ctrl.Antecedent(np.arange(-90, 91, 1), 'angle')
angular_velocity = ctrl.Antecedent(np.arange(-100, 101, 1), 'angular_velocity')
force = ctrl.Consequent(np.arange(-100, 101, 1), 'force')

angle['left'] = fuzz.trapmf(angle.universe, [-90, -90, -45, 0])
angle['center'] = fuzz.trimf(angle.universe, [-45, 0, 45])
angle['right'] = fuzz.trapmf(angle.universe, [0, 45, 90, 90])

angular_velocity['left'] = fuzz.trapmf(angular_velocity.universe, [-100, -100, -50, 0])
angular_velocity['zero'] = fuzz.trimf(angular_velocity.universe, [-50, 0, 50])
angular_velocity['right'] = fuzz.trapmf(angular_velocity.universe, [0, 50, 100, 100])

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
    ctrl.Rule(angle['right'] & angular_velocity['right'], force['strong_right'])
]

control_system = ctrl.ControlSystem(rules)
fis_simulation = ctrl.ControlSystemSimulation(control_system)

def generate_fuzzy_data(num_samples=1000):
    angle_data = np.random.uniform(-90, 90, num_samples)
    angular_velocity_data = np.random.uniform(-100, 100, num_samples)
    force_data = []

    for ang, ang_vel in zip(angle_data, angular_velocity_data):
        fis_simulation.input['angle'] = ang
        fis_simulation.input['angular_velocity'] = ang_vel
        fis_simulation.compute()
        force_data.append(fis_simulation.output['force'])

    return np.array(angle_data), np.array(angular_velocity_data), np.array(force_data)

angle_data, angular_velocity_data, force_data = generate_fuzzy_data()
X = np.column_stack((angle_data, angular_velocity_data))
y = force_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

mlp = MLPRegressor(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

y_pred_train = mlp.predict(X_train)
y_pred_test = mlp.predict(X_test)
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

print("Erro quadrado médio - Treinamento:", mse_train)
print("Erro quadrado médio - Teste:", mse_test)

def test_neuro_fuzzy(angle_test, angular_velocity_test):
    fuzzy_result = []
    neuro_fuzzy_result = []

    for ang, ang_vel in zip(angle_test, angular_velocity_test):
        fis_simulation.input['angle'] = ang
        fis_simulation.input['angular_velocity'] = ang_vel
        fis_simulation.compute()
        fuzzy_result.append(fis_simulation.output['force'])

        mlp_output = mlp.predict([[ang, ang_vel]])
        neuro_fuzzy_result.append(mlp_output[0])

    return fuzzy_result, neuro_fuzzy_result

angle_test = np.random.uniform(-90, 90, 10)
angular_velocity_test = np.random.uniform(-100, 100, 10)
fuzzy_result, neuro_fuzzy_result = test_neuro_fuzzy(angle_test, angular_velocity_test)

print("\nComparação entre FIS e Neuro-Fuzzy (MLP):")
for i in range(len(angle_test)):
    print(f"Ângulo: {angle_test[i]:.2f}, Velocidade Angular: {angular_velocity_test[i]:.2f} -> "
          f"FIS: {fuzzy_result[i]:.2f}, Neuro-Fuzzy: {neuro_fuzzy_result[i]:.2f}")
