import random

from bottom_left_fill import BottomLeftFill


class GeneticAlgorithm():
    """Genetic algorithm class"""
    def __init__(self, population_size, parents_number, chromosome_length, mutation_rate):
        self.population_size = population_size
        self.parents_number = parents_number
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self._offspring_size = self.population_size - self.parents_number

    def run(self, num_generations):
        """Function that implements a genetic algorithm."""
        population = self.generate_population()
        for _ in range(num_generations):
            fitness_values = [
                self._calculate_fitness(chromosome) for chromosome in population
            ]
            parents = self._select_parents(population, fitness_values)
            offspring = self._crossover(parents)
            population = self._mutate(offspring) + parents
        best_chromosome = population[fitness_values.index(max(fitness_values))]
        return best_chromosome

    def _generate_population(self):
        """Generates a population of chromosomes with shuffled values."""
        population = []
        for _ in range(self.population_size):
            chromosome = list(range(self.chromosome_length))
            chromosome = random.shuffle(chromosome)
            population.append(chromosome)
        return population

    def _calculate_fitness(self, chromosome):
        """Calculates the fitness value of a chromosome."""
        fitness = BottomLeftFill.get_max_height(chromosome)
        return fitness

    def _select_parents(self, population, fitness_values):
        """Selects the best chromosomes to be parents for the next generation."""
        parents = []
        for i in range(self.num_parents):
            max_fitness_index = fitness_values.index(max(fitness_values))
            parents.append(population[max_fitness_index])
            fitness_values[max_fitness_index] = -1
        return parents

    def _crossover(self, parents):
        """Generates offspring through crossover of the selected parents.
        Implementation of partially mapped crossover"""
        offspring = []
        for _ in range(self._offspring_size):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            chromosome1, chromosome2 = self._partially_mapped_crossover(parent1, parent2)
            offspring.append(chromosome1)
            offspring.append(chromosome2)
        return offspring

    def _partially_mapped_crossover(self, parent1, parent2):
        """Performs partially mapped crossover on two parents to generate offspring."""
        chromosome1 = []
        chromosome2 = []
        start = random.randint(0, self.chromosome_length - 1)
        end = random.randint(start, self.chromosome_length - 1)
        for i in range(self.chromosome_length):
            if start <= i <= end:
                chromosome1.append(parent1[i])
                chromosome2.append(parent2[i])
            else:
                chromosome1.append(-1)
                chromosome2.append(-1)
        for i in range(self.chromosome_length):
            if chromosome1[i] == -1:
                chromosome1[i] = self._get_value(parent2[i], chromosome2, chromosome1)
            if chromosome2[i] == -1:
                chromosome2[i] = self._get_value(parent1[i], chromosome1, chromosome2)
        return chromosome1, chromosome2

    def _get_value(self, value, chromosome1, chromosome2):
        """Returns the value to be inserted in the offspring's chromosome."""
        while value in chromosome1:
            index = chromosome1.index(value)
            value = chromosome2[index]
        return value

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
