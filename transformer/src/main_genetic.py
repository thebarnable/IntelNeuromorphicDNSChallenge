from genetic.genetic import *
import numpy
from populations import *
from main_network import *

from pprint import pprint



##### Parameters #####
parser = argparse.ArgumentParser(
                    prog='Conformer_lstm genetic algorithm',
                    description='This program can be used to explore different conformer-based speech enhancement methods by a genetic algorithm'
                    )

parser.add_argument('-t', '--tag', required=True, help="tag to save log")
parser.add_argument('-b', '--batch', type=int, required=True, help="upper limit for number of batches")
parser.add_argument('-i', '--init_pop', required=True, help="use automatic for initial population or select custom as provided in custom_pop.py")

args = parser.parse_args()
tag = args.tag
limit = args.batch
init_pop = args.init_pop

current_best = None
fitnesses = []

num_generations = 10
num_parents_mating = 5
num_genes = 12

def fitness_func(solution, solution_idx):
  global current_best
  global fitnesses
  print("genetic_{}_{}".format(ga_instance.generations_completed, solution_idx))
  print(solution)
  input_dict = pop2dict(solution)
  pprint(input_dict)
  snr = execute(input_dict, "genetic_{}_{}_{}".format(tag, ga_instance.generations_completed, solution_idx), genetic=True, limit=limit)
  fitnesses.append(snr)
  if snr >= max(fitnesses):
    current_best = solution
  return snr

def callback_generation(ga_instance):
  global current_best
  global fitnesses
  print("Generation = {generation}".format(generation=ga_instance.generations_completed), flush=True)
  print("Max fitness = {fitness}".format(fitness=max(fitnesses)), flush=True)
  print("Current best: ")
  pprint(current_best)
  pprint(pop2dict(current_best))
  
  fix_pop(ga_instance.population)
  print("pop_size", len(ga_instance.population))
  print(ga_instance.population)


ga_instance = Genetic(
                        initial_population=get_initial_pop(init_pop),
                        num_genes=num_genes,
                        num_parents=num_parents_mating, num_generations=num_generations,
                        num_mutations=1,
                        fitness_func=fitness_func,
                        on_generation=callback_generation,
                        keep_parents=True
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
