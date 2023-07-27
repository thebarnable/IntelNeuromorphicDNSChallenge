import pygad
import numpy
from populations import *
from main_network import *

from pprint import pprint

current_best = None
fitnesses = []

my_global_idx = 0

def fitness_func(ga_instance, solution, solution_idx):
  global current_best
  global fitnesses
  global my_global_idx
  print("solution_idx: ", solution_idx)
  print("genetic_{}_{}".format(ga_instance.generations_completed, my_global_idx))
  print(solution)
  input_dict = pop2dict(solution)
  pprint(input_dict)
  snr = execute(input_dict, "genetic_{}_{}".format(ga_instance.generations_completed, my_global_idx), genetic=True)
  fitnesses.append(snr)
  if snr >= max(fitnesses):
    current_best = solution
  my_global_idx += 1
  return snr

num_generations = 5
num_parents_mating = 10
sol_per_pop = 40
num_genes = 12

def callback_generation(ga_instance):
  global current_best
  global fitnesses
  print("Generation = {generation}".format(generation=ga_instance.generations_completed), flush=True)
  print("Max fitness = {fitness}".format(fitness=max(fitnesses)), flush=True)
  print("Current best: ")
  pprint(current_best)
  pprint(pop2dict(current_best))

def callback_mutation(ga_instance, offspring_mutation):
  print("on_mutation()")
  fix_pop(offspring_mutation)
  print("pop_size", len(offspring_mutation))
  print(offspring_mutation)

ga_instance = pygad.GA(
                        initial_population=get_initial_pop(),
                        num_generations=num_generations,
                        num_parents_mating=num_parents_mating, 
                        fitness_func=fitness_func,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        keep_parents=num_parents_mating,
                        parent_selection_type="rank",
                        crossover_type="single_point",
                        mutation_type="random",
                        mutation_num_genes=1,
                        on_generation=callback_generation,
                        on_mutation=callback_mutation
                      )

ga_instance.run()

print("\n\nDONE")
print("Generation = {generation}".format(generation=ga_instance.generations_completed), flush=True)
print("Max fitness = {fitness}".format(fitness=max(fitnesses)), flush=True)
print("Current best: ")
pprint(current_best)
pprint(pop2dict(current_best))
print("Last generation")
print(ga_instance.population)

# Saving the GA instance.
filename = '../genetic/genetic_final'
ga_instance.save(filename=filename)

# Loading the saved GA instance.
loaded_ga_instance = pygad.load(filename=filename)
loaded_ga_instance.plot_fitness()