import numpy as np
from agents import SimpleNeuralNetwork, NeuralBasedAgent
import random

class BatAlgorithm:
    def __init__(self, pop_size, n_iter, dim, fmin, fmax, game_env, config):
        self.pop_size = pop_size
        self.n_iter = n_iter
        self.dim = dim  # número de pesos (1475)
        self.fmin = fmin
        self.fmax = fmax
        self.game_env = game_env
        self.config = config

        # Inicialização
        self.freq = np.zeros(pop_size)
        self.vel = np.zeros((pop_size, dim))
        self.pos = np.random.uniform(-1, 1, (pop_size, dim))  # população inicial
        self.fitness = np.array([self.evaluate(ind) for ind in self.pos])

        best_idx = np.argmin(self.fitness)
        self.best = self.pos[best_idx].copy()
        self.best_fitness = self.fitness[best_idx]

        # Loudness e taxa de emissão
        self.A = np.ones(pop_size)
        self.r = np.zeros(pop_size)

    def evaluate(self, weights):
        # Cria agente com a rede neural treinada
        nn = SimpleNeuralNetwork(weights)
        agent = NeuralBasedAgent(self.config)
        agent.model = nn

        # Joga várias partidas para obter média
        scores = []
        for _ in range(3):  # mudar para 30
            score = self.game_env.play(agent)
            scores.append(score)

        return -np.mean(scores)  # queremos minimizar → fitness negativo

    def run(self):
        for t in range(self.n_iter):
            for i in range(self.pop_size):
                # Atualiza frequência, velocidade, posição
                self.freq[i] = self.fmin + (self.fmax - self.fmin) * random.random()
                self.vel[i] += (self.pos[i] - self.best) * self.freq[i]
                new_pos = self.pos[i] + self.vel[i]

                # Busca local
                if random.random() > self.r[i]:
                    eps = np.random.uniform(-1, 1, self.dim)
                    new_pos = self.best + eps * np.mean(self.A)

                # Avalia
                new_fit = self.evaluate(new_pos)

                # Aceita solução com certa probabilidade
                if (new_fit <= self.fitness[i]) and (random.random() < self.A[i]):
                    self.pos[i] = new_pos
                    self.fitness[i] = new_fit
                    self.A[i] *= 0.9
                    self.r[i] = self.r[i] * 0.9 + 0.1

                # Atualiza global
                if new_fit <= self.best_fitness:
                    self.best = new_pos.copy()
                    self.best_fitness = new_fit

            print(f"[Iter {t}] Melhor fitness: {-self.best_fitness:.2f}")

        return self.best
