import numpy as np
import math
from .calibration_problem import CalibrationProblem

class SimulatedAnnealing:
    """
    Implements the Simulated Annealing algorithm to find optimal model parameters.
    This metaheuristic is inspired by the metallurgical process of annealing 
    and is designed for global optimization. It avoids local minima by
    sometimes accepting worse solutions, as the temprature decreases the algorithm
    becomes less likely to accept a worst solution, thus finetunes our solution.
    """
    def __init__(self, problem: CalibrationProblem, initial_temp, final_temp, alpha, scale_factor=0.1):
        self.problem = problem
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.alpha = alpha  
        self.scale_factor = scale_factor 
        
    def _is_valid(self, params):
        """
        Ensures that generated parameters are economically valid.
        """
        if self.problem.model_type == 'HW1F':
            a_x, sigma_x = params
            return sigma_x > 0 and a_x > 0
        elif self.problem.model_type == 'HW2F':
            a_x, sigma_x, a_y, sigma_y, rho = params
            return sigma_x > 0 and sigma_y > 0 and a_x > 0 and a_y > 0 and -1 <= rho <= 1
        return False

    def generate_neighbor(self, current_params, temp):
        """
        Generates a new candidate solution by adding a random Gaussian perturbation,
        with a standard deviation proportional to the temperature.
        """
        while True:
            perturbation = np.random.normal(0, self.scale_factor * temp, len(current_params))
            new_params = current_params + perturbation
            if self._is_valid(new_params):
                return new_params

    def acceptance_probability(self, old_cost, new_cost, temp):
        """
        Calculates the probability of accepting a new solution.
        Worse solutions are accepted with a probability controlled by the temperature.
        """
        if new_cost < old_cost:
            return 1.0
        else:
            # Boltzmann distribution-like acceptance criterion
            return math.exp((old_cost - new_cost) / temp)

    def calibrate(self, initial_params, max_iter_no_improvement=50, verbose=True):
            """
            Runs the main simulated annealing loop.
            The 'verbose' flag controls the printing of intermediate results.
            """
            temp = self.initial_temp
            current_params = np.array(initial_params)
            current_cost = self.problem.cost_function(current_params)
            
            best_params = current_params
            best_cost = current_cost
            
            iter_since_last_improvement = 0
            history = {'temp': [], 'cost': [], 'best_cost': []}
    
            while temp > self.final_temp:
                neighbor_params = self.generate_neighbor(current_params, temp)
                neighbor_cost = self.problem.cost_function(neighbor_params)
                
                if self.acceptance_probability(current_cost, neighbor_cost, temp) > np.random.rand():
                    current_params = neighbor_params
                    current_cost = neighbor_cost
                
                if current_cost < best_cost:
                    best_params = current_params
                    best_cost = current_cost
                    iter_since_last_improvement = 0
                    if verbose:
                        print(f"New best cost: {best_cost:.6f} at T={temp:.4f}, params={np.round(best_params, 4)}")
                else:
                    iter_since_last_improvement += 1
    
                # Store history for plotting
                history['temp'].append(temp)
                history['cost'].append(current_cost)
                history['best_cost'].append(best_cost)
                
                temp *= self.alpha
    
                if iter_since_last_improvement > max_iter_no_improvement:
                    if verbose:
                        print("Stopping early: no improvement in best solution.")
                    break
            
            if verbose:
                print("Calibration finished.")
            return best_params, best_cost, history
