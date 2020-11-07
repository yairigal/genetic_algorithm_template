import json
import logging
import os
import random
from abc import ABCMeta
from abc import abstractmethod

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
logger.addHandler(sh)


class GeneticAlgorithm(metaclass=ABCMeta):
    """Generic genetic algorithm class.

    Implement the abstract function and call `run`.

    Attributes:
        population_size (number): the amount of individuals in each generation.
        selection_percentage (float): number between 0 - 1 represents the percentage of individuals that are kept
            for crossover.
        max_parent_amount (number): the amount of parents are involved in a single crossover.
        mutation_rate (float): number between 0 - 1 represents the population percentage that are being mutated.
        enable_graph (bool): whether to display average fitness and top fitness graph that updates each generation.
        best_model_save_path (str): path to where to save the best model each generation.
        checkpoint_folder (str): path to where to save a checkpoint of the algorithm once a while.
        save_every (number): after how many generation to save a checkpoint (e.g. save every 10 generations).
    """

    def __init__(self, population_size, selection_percentage, mutation_rate, max_parents=2, enable_graph=True,
                 best_model_save_path='./best_model', checkpoint_folder='./checkpoint', save_every=10):
        self.population_size = population_size
        self.selection_percentage = selection_percentage
        self.max_parents_amount = max_parents
        self.mutation_rate = mutation_rate
        self.enable_graph = enable_graph
        self.best_model_save_path = best_model_save_path
        self.checkpoint_folder = checkpoint_folder
        self.save_frequency = save_every

        self.population_folder = os.path.join(self.checkpoint_folder, 'population')
        self.instance_folder = os.path.join(self.checkpoint_folder, 'instance.json')

        self.current_generation = 0
        self.best_model = None

        self.population = None
        self.avg_fitnesses = []
        self.top_fitnesses = []

    @abstractmethod
    def _calculate_individual_fitness(self, results, *args, **kwargs):
        """Calculate the fitness of a single individual.

        Higher fitness results in higher survivability chance.

        Args:
            results: results from the fitness run.

        Returns:
            number. the fitness calculated for the single individual.
        """
        return NotImplemented

    @abstractmethod
    def _individual_crossover(self, *parents):
        """Merge between multiple individuals to create another one.

        Args:
            *parents: the parents of the created individual.

        Returns:
            the individual created.
        """
        return NotImplemented

    @abstractmethod
    def _individual_mutation(self, individual):
        """Mutate a single individual

        Args:
            individual: the individual to be mutated.
        """
        raise NotImplementedError

    @abstractmethod
    def _init_population(self):
        """Initialize the population.

        Returns:
            list. list of the size of `self.population_size`.
        """
        return NotImplemented

    @abstractmethod
    def _save_single_individual(self, individual, filename):
        """Save a individual to a file.

        Args:
            individual: the individual to save.
            filename: where to save the individual.
        """
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _load_single_individual(filename):
        """Load a individual from a file.

        Args:
            filename: the file where the individual is saved.

        Returns:
            the individual.
        """
        return NotImplemented

    def run_fitness_test(self):
        return NotImplemented

    def fitness_test(self):
        fitnesses = []
        for i, individual in enumerate(self.population):
            print(f'fitness test {i + 1}/{self.population_size}', end='\r')
            results = self.run_fitness_test()
            fitnesses.append(self._calculate_individual_fitness(results))

        return fitnesses

    def selection(self, fitnesses):
        # Sort population by fitness from the best to worst
        self.population, fitnesses = zip(*sorted(zip(self.population, fitnesses), key=lambda i: i[1], reverse=True))
        total_fitness = sum(fitnesses)
        selection_amount = int(self.selection_percentage * self.population_size)

        self.best_model = self.population[0]
        self.average_fitness = total_fitness / self.population_size
        self.top_fitnesses.append(fitnesses[0])
        self.avg_fitnesses.append(self.average_fitness)
        if total_fitness == 0:
            # All are the same
            return [random.choice(self.population) for _ in range(selection_amount)]

        fitnesses_normalized = [fitness / total_fitness for fitness in fitnesses]
        fitness_accumulator = 0
        fitnesses_accumulated = []
        for fitness in fitnesses_normalized:
            fitness_accumulator += fitness
            fitnesses_accumulated.append(fitness_accumulator)

        selected = []
        for _ in range(selection_amount):
            chance = random.random()
            selected_individual = next(
                individual for individual, fitness_accumulated in zip(self.population, fitnesses_accumulated)
                if chance <= fitness_accumulated)
            selected.append(selected_individual)

        assert len(selected) == selection_amount
        return selected

    def crossover(self, selected):
        population_size_left = self.population_size - len(selected)
        added_population = []
        for _ in range(population_size_left):
            parents_amount = random.randint(2, self.max_parents_amount)
            parents = [random.choice(selected) for _ in range(parents_amount)]
            added_population.append(self._individual_crossover(*parents))

        selected.extend(added_population)
        return selected

    def mutation(self):
        for individual in self.population:
            if random.random() <= self.mutation_rate:
                self._individual_mutation(individual)

    def generation(self):
        # fitness test
        logger.debug('fitness test')
        fitnesses = self.fitness_test()
        # selection
        logger.debug("selection")
        selected_individuals = self.selection(fitnesses)
        # crossover
        logger.debug("crossover")
        self.population = self.crossover(selected_individuals)
        # mutation
        logger.debug("mutation")
        self.mutation()

    def display_summary(self):
        print(f"Generation {self.current_generation}:\n"
              f"average fitness={self.average_fitness}")

        if self.enable_graph:
            plt.plot(list(range(self.current_generation)), self.avg_fitnesses, color='b', label='average fitnesses')
            plt.plot(list(range(self.current_generation)), self.top_fitnesses, color='r', label='top fitnesses')
            plt.pause(0.000001)

    def run(self, skip_init=False):
        if not skip_init:
            self.population = self._init_population()

        if self.enable_graph:
            plt.plot(list(range(self.current_generation)), self.avg_fitnesses, color='b', label='average fitnesses')
            plt.plot(list(range(self.current_generation)), self.top_fitnesses, color='r', label='top fitnesses')
            plt.legend(loc="upper left")
            plt.xlabel('generation')

        while True:
            self.current_generation += 1
            self.generation()
            self.display_summary()
            self._save_best_model()
            if self.current_generation % self.save_frequency == 0:
                self._checkpoint()

    def _save_best_model(self):
        logger.debug(f'Saving best model to {self.best_model_save_path}')
        self._save_single_individual(individual=self.best_model, filename=self.best_model_save_path)

    def _checkpoint(self):
        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)
        # save population
        if not os.path.exists(self.population_folder):
            os.makedirs(self.population_folder)

        for i, indv in enumerate(self.population):
            filename = os.path.join(self.population_folder, f'individual_{i}')
            self._save_single_individual(individual=indv, filename=filename)

        # save other stuff
        with open(self.instance_folder, 'w') as f:
            json.dump(self.__dict__, f)

    @classmethod
    def load_checkpoint(cls, checkpoint_folder='./checkpoint'):
        instance_file = os.path.join(checkpoint_folder, 'instance.json')
        with open(instance_file, 'r') as f:
            attributes = json.load(f)

        instance = cls(**attributes)

        instance.population = []
        for indv in os.listdir(os.path.join(checkpoint_folder, 'population')):
            filename = os.path.join(checkpoint_folder, 'population', indv)
            p = cls._load_single_individual(filename)
            instance.population.append(p)

        return instance
