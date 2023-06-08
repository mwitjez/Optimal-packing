import random
import math
import matplotlib.pyplot as plt

from tqdm import tqdm


class GeneticAlgorithm:
    """Genetic algorithm class"""
    def __init__(self, parents_number, chromosome_length, mutation_rate, bottom_left_packer):
        self.parents_number = parents_number
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.bottom_left_packer = bottom_left_packer
        self._offspring_factor = 0.5
        self.best_fitness = []
        self.max_heights = []

    def run(self, num_generations, population_size):
        """Function that implements a genetic algorithm."""
        population = self._generate_population(population_size)
        for _ in tqdm(range(num_generations)):
            fitness_values = [self._calculate_fitness(chromosome) for chromosome in population]
            self.best_fitness.append(max(fitness_values))
            best_chromosome = population[fitness_values.index(self.best_fitness[-1])]
            self.max_heights.append(self.bottom_left_packer.get_max_height(best_chromosome))
            parents = self._select_parents(population, fitness_values)
            offspring = self._crossover(parents, int(self._offspring_factor * population_size))
            newcomers = self._generate_population(population_size - len(offspring) - len(parents) - 1)
            population = self._mutate(offspring) + parents + newcomers
            population.append(best_chromosome)
        return best_chromosome

    def _generate_population(self, population_size):
        """Generates a population of chromosomes with shuffled values."""
        population = []
        for _ in range(population_size):
            chromosome = list(range(self.chromosome_length))
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _calculate_fitness(self, chromosome):
        """Calculates the fitness value of a chromosome."""
        max_height = self.bottom_left_packer.get_max_height(chromosome)
        packing_density = self.bottom_left_packer.get_packing_density(chromosome)
        fitness = 1000/(max_height)**3 + packing_density
        if math.isnan(fitness) or math.isinf(fitness):
            fitness = 0
        return fitness

    def _select_parents(self, population, fitness_values):
        """Selects the best chromosomes to be parents for the next generation using roulette wheel selection."""
        parents = []
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        for _ in range(self.parents_number):
            spin = random.uniform(0, 1)
            cumulative_probability = 0
            for i, probability in enumerate(probabilities):
                cumulative_probability += probability
                if spin <= cumulative_probability:
                    parents.append(population[i])
                    break
        return parents

    def _crossover(self, parents, offspring_size):
        """Generates offspring through crossover of the selected parents.
        Implementation of partially mapped crossover"""
        offspring = []
        for _ in range(offspring_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            new_chromosome = self._partially_mapped_crossover(parent1, parent2)
            offspring.append(new_chromosome)
        return offspring

    def _partially_mapped_crossover(self, parent1, parent2):
        """Perform Partially Mapped Crossover (PMX) on two parent strings to generate offspring."""
        point1 = random.randint(0, len(parent1) - 1)
        point2 = random.randint(0, len(parent1) - 1)

        if point2 < point1:
            point1, point2 = point2, point1

        offspring = parent1[:]

        for i in range(point1, point2 + 1):
            if offspring[i] not in parent2[point1:point2 + 1]:
                j = parent2.index(offspring[i])
                offspring[i], offspring[j] = offspring[j], offspring[i]
        for i in range(len(offspring)):
            if i < point1 or i > point2:
                while offspring[i] in offspring[point1:point2 + 1]:
                    j = parent2.index(offspring[i])
                    offspring[i] = parent2[j]

        return offspring

    def _mutate(self, offspring):
        """Implements order based mutation on the offspring."""
        for i in range(len(offspring)):
            if random.random() < self.mutation_rate:
                offspring[i] = self._order_based_mutation(offspring[i])
        return offspring

    def _order_based_mutation(self, chromosome):
        """Performs order based mutation on a chromosome."""
        start = random.randint(0, self.chromosome_length - 1)
        end = random.randint(start, self.chromosome_length - 1)
        sub_list = chromosome[start:end]
        random.shuffle(sub_list)
        chromosome[start:end] = sub_list
        return chromosome

    def plot_stats(self):
        """Plots the best fitness and max heightsvalues for each generation."""
        fig, axs = plt.subplots(2, 1,)
        axs[0].plot(self.best_fitness, label="Best fitness")
        axs[1].plot(self.max_heights, label="Max height")
        axs[0].legend()
        axs[1].legend()
        plt.xlabel("Generation")
        plt.show()
