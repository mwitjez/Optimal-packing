import random

import matplotlib.pyplot as plt


class GeneticAlgorithm():
    """Genetic algorithm class"""
    def __init__(self, population_size, parents_number, chromosome_length, mutation_rate, bottom_left_packer):
        self.population_size = population_size
        self.parents_number = parents_number
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self._offspring_size = self.population_size - self.parents_number
        self.bottom_left_packer = bottom_left_packer
        self._best_fitness = []

    def run(self, num_generations):
        """Function that implements a genetic algorithm."""
        population = self._generate_population()
        for _ in range(num_generations):
            fitness_values = [self._calculate_fitness(chromosome) for chromosome in population]
            self._best_fitness.append(min(fitness_values))
            best_chromosome = population[fitness_values.index(self._best_fitness[-1])]
            parents = self._select_parents(population, fitness_values)
            offspring = self._crossover(parents)
            population = self._mutate(offspring) + parents
        return best_chromosome

    def _generate_population(self):
        """Generates a population of chromosomes with shuffled values."""
        population = []
        for _ in range(self.population_size):
            chromosome = list(range(self.chromosome_length))
            random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _calculate_fitness(self, chromosome):
        """Calculates the fitness value of a chromosome."""
        fitness = self.bottom_left_packer.get_max_height(chromosome)
        return fitness

    def _select_parents(self, population, fitness_values):
        """Selects the best chromosomes to be parents for the next generation."""
        parents = []
        for _ in range(self.parents_number):
            best_fitness_index = fitness_values.index(min(fitness_values))
            parents.append(population[best_fitness_index])
            fitness_values[best_fitness_index] = float("Inf")
        return parents

    def _crossover(self, parents):
        """Generates offspring through crossover of the selected parents.
        Implementation of partially mapped crossover"""
        offspring = []
        for _ in range(self._offspring_size):
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
        for i in range(self._offspring_size):
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
        """Plots the best fitness values for each generation."""
        plt.plot(self._best_fitness)
        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.show()
