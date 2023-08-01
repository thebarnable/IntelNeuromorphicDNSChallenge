import numpy as np
import random
from pprint import pprint

class Genetic:
  def __init__(self, initial_population, num_genes, num_parents, num_generations,num_mutations, fitness_func, on_generation, keep_parents=True):
    self.num_individuals = len(initial_population)
    self.population = initial_population
    self.num_genes = num_genes
    self.num_parents = num_parents
    self.fitnesses = []
    self.num_generations = num_generations
    self.keep_parents = keep_parents
    self.parents = None
    self.num_mutations = num_mutations
    self.generations_completed = 0
    self.fitness_func = fitness_func
    self.on_generation = on_generation
    assert num_parents < self.num_individuals, "Number of parents has to be smaller than the number of individuals"
    assert num_parents >= 2, "Number of parents has to be at least 2"

  def run(self):
    for generation in range(self.num_generations):
      print("GENERATION: ", self.generations_completed)
      new_fitnesses = []
      for index, individual in enumerate(self.population):
        my_fitness = self.fitness_func(individual, index)
        new_fitnesses.append(my_fitness)
      self.fitnesses.append(new_fitnesses)
      new_fitnesses = np.array(new_fitnesses)

      print("Generation: {}, Avg. fitness: {}, Best in population: {}".format(generation, np.mean(new_fitnesses), np.max(new_fitnesses)))

      # Next population if not already last
      if generation < self.num_generations - 1:
        # parents - select fittest
        parent_indices = np.argsort(new_fitnesses)[-self.num_parents:]
        self.parents = self.population[parent_indices]

        # new population
        new_population = []
        if self.keep_parents:
          for item in self.parents:
            new_population.append(item)
        while len(new_population) < self.num_individuals:
          # crossover
          offspring = []
          # pick two parents
          current_parent_indices = random.sample(range(self.num_parents), k=2)
          current_parents = self.parents[current_parent_indices]
          # scattered crossover
          for gene_idx in range(self.num_genes):
            offspring.append(current_parents[random.randint(0, 1)][gene_idx])

          # mutation
          mutation_indices = random.sample(range(self.num_genes), k=5)
          for item in mutation_indices:
            offspring[item] = random.random()
          new_population.append(offspring)
        self.population = np.array(new_population)
        self.on_generation(self)
        self.generations_completed += 1
    print()
    print("Run completed")
    print("Total number of generations: ", self.generations_completed)
    print("Best fitness value: ", np.max(self.fitnesses))
    pprint(self.fitnesses)

          




      