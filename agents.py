import numpy as np
from abc import ABC, abstractmethod
from typing import List
from game.core import GameConfig
import random

class Agent(ABC):
    @abstractmethod
    def predict(self, state: np.ndarray) -> int:
        pass

class HumanAgent(Agent):
    def predict(self, state: np.ndarray) -> int:
        return 0 #the input is from keyboard
import numpy as np
import random

# Classe simulada caso não exista
class Agent:
    pass

# Classe da rede neural simplificada (sem treinamento por enquanto)
class SimpleNeuralNetwork:
    def __init__(self, weights):
        if weights.size != 1475:
                    raise ValueError(f"O vetor de pesos deve ter 1475 valores, mas tem {weights.size}.")
                # Separação dos pesos
        idx = 0

        self.w1 = weights[idx:idx + 27*32].reshape((27, 32))
        idx += 27*32

        self.b1 = weights[idx:idx + 32].reshape((1, 32))
        idx += 32

        self.w2 = weights[idx:idx + 32*16].reshape((32, 16))
        idx += 32*16

        self.b2 = weights[idx:idx + 16].reshape((1, 16))
        idx += 16

        self.w3 = weights[idx:idx + 16*3].reshape((16, 3))
        idx += 16*3

        self.b3 = weights[idx:idx + 3].reshape((1, 3))

    def tanh(self, x):
        return np.tanh(x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def forward(self, x):
        x = x.reshape(1, -1)
        z1 = x @ self.w1 + self.b1
        a1 = self.tanh(z1)
        z2 = a1 @ self.w2 + self.b2
        a2 = self.tanh(z2)
        z3 = a2 @ self.w3 + self.b3
        return self.softmax(z3)

    def predict(self, x):
        output = self.forward(x)
        return int(np.argmax(output, axis=1)[0])


# NeuralBasedAgent com mesma estrutura da RuleBasedAgent
class NeuralBasedAgent(Agent):
    def __init__(self, config, danger_threshold=None, lookahead_cells=None, diff_to_center_to_move=None):
        self.config = config
        self.grid_size = config.sensor_grid_size
        self.sensor_range = config.sensor_range
        self.screen_height = config.screen_height
        self.cell_size = self.sensor_range / self.grid_size

        self.danger_threshold = danger_threshold if danger_threshold is not None else 0.3
        self.lookahead_cells = int(np.rint(lookahead_cells)) if lookahead_cells is not None else 3
        self.diff_to_center_to_move = diff_to_center_to_move if diff_to_center_to_move is not None else 200

        # Inicializa a rede neural
        self.model = SimpleNeuralNetwork()

    def predict(self, state: np.ndarray) -> int:
        if state.shape[0] != self.grid_size * self.grid_size + 2:
            raise ValueError(f"Esperado vetor de {self.grid_size**2 + 2} elementos, mas recebeu {state.shape[0]}.")

        # Extração da grade (início do vetor) e variáveis internas (últimos 2 valores)
        grid_flat = state[:self.grid_size * self.grid_size]
        grid = grid_flat.reshape((self.grid_size, self.grid_size))

        player_y_normalized = state[-2] * self.screen_height
        diff_to_center = player_y_normalized - (self.screen_height / 2)

        # Aqui usamos o modelo treinado para prever a ação
        action = self.model.predict(state)

        return action


class RuleBasedAgent(Agent):
    def __init__(self, config: GameConfig,danger_threshold = None, lookahead_cells = None, diff_to_center_to_move = None):
        self.config = config
        self.grid_size = config.sensor_grid_size
        self.sensor_range = config.sensor_range
        self.cell_size = self.sensor_range / self.grid_size

        if danger_threshold == None:
            self.danger_threshold = 0.3  # How close obstacles need to be to react
        else:
            self.danger_threshold = danger_threshold

        if lookahead_cells == None:
            self.lookahead_cells = 3  # How many cells ahead to check for obstacles
        else:
            self.lookahead_cells = int(np.rint(lookahead_cells))

        if diff_to_center_to_move == None:
            self.diff_to_center_to_move = 200
        else:
            self.diff_to_center_to_move = diff_to_center_to_move
        
    def predict(self, state: np.ndarray) -> int:
        # Reshape the state into grid if it's flattened
        grid_flat = state[:self.grid_size*self.grid_size]
        grid = grid_flat.reshape((self.grid_size, self.grid_size))
        player_y_normalized = state[-2] * self.config.screen_height # Second last element
        center_row = self.grid_size // 2
        
        # Check immediate danger in front (first column)
        first_col = grid[:, 0]
        if np.any(first_col):
            # Obstacle directly in front - need to dodge
            obstacle_rows = np.where(first_col)[0]
            
            # If obstacle is above center, go down
            if np.any(obstacle_rows < center_row):
                return 2
            # If obstacle is below center or covers center, go up
            else:
                return 1
        
        # Look ahead in the next few columns for obstacles
        for col in range(1, min(self.lookahead_cells, self.grid_size)):
            if np.any(grid[:, col]):
                # Calculate distance to obstacle
                distance = col * self.cell_size
                
                # If obstacle is getting close, prepare to dodge
                if distance < self.danger_threshold * self.sensor_range:
                    obstacle_rows = np.where(grid[:, col])[0]
                    
                    # Find the gap (if any)
                    obstacle_present = np.zeros(self.grid_size, dtype=bool)
                    obstacle_present[obstacle_rows] = True
                    
                    # Check for gaps above or below
                    gap_above = not np.any(obstacle_present[:center_row])
                    gap_below = not np.any(obstacle_present[center_row+1:])
                    
                    if gap_above and not gap_below:
                        return 1  # Move up
                    elif gap_below and not gap_above:
                        return 2  # Move down
                    elif gap_above and gap_below:
                        # Both gaps available, choose randomly
                        return random.choice([1, 2])
                    else:
                        # No gap, choose randomly (will probably hit)
                        return random.choice([0, 1, 2])
        #print(player_y_normalized)
        diff_to_center = player_y_normalized - (self.config.screen_height/2)
 
        if diff_to_center < -self.diff_to_center_to_move:
            return 2  # Must move down
        elif diff_to_center > self.diff_to_center_to_move:
            return 1  # Must move up

        # Default action - no movement needed
        return 0
