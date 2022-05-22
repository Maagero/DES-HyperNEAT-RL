"""
An experiment using HyperNEAT to solve the maze
Fitness threshold set in config
- by default very high to show the high possible accuracy of this library.
"""

import pickle
import sys
import os
import copy
import random
import argparse
from turtle import position
sys.path.insert(1, '/Users/maagero/src/deshyperneatrl/maze')
import maze_environment as maze
import agent
import visualize

sys.path.insert(1, '/Users/maagero/src/deshyperneatrl/')



import config
import genome
import reproduction_deshyperneat
import species
import stagnation
import feed_forward
import population_deshyperneat
import statistics
import layout
import reporting
import parallel
import stats as st
from visualize_cppn import draw_net
from substrate import Substrate
from hyperneat import create_phenotype_network
from des_hyperneat import DESNetwork

VERSION = "L"
VERSION_TEXT = "small" if VERSION == "S" else "medium" if VERSION == "M" else "large"

# Network inputs and expected outputs.
INPUT_SUBSTRATES = [[(-1, 0), (-7/9, 0), (-5/5,0), (-3/9,0), (-1/9,0), (1/9,0), (3/9, 0), (5/9, 0), (7/9,0), (1,0)]]
#INPUT_SUBSTRATES = [[(0,0)] for i in range(10)]
#INPUT_SUBSTRATES = [[(-1, 0), (-3/5, 0), (-1/5,0), (1/5,0), (3/5,0), (1,0)], [(-1, 0), (-1/3, 0), (1/3,0), (1,0)]]
OUTPUT_SUBSTRATES = [[(-1, 1.0), (1, 1.0)]]

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 1,
            "max_depth": 3,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.03,
            "max_weight": 5.0,
            "activation": "sigmoid"}

DYNAMIC_PARAMS = params(VERSION)

local_dir = os.path.dirname(__file__)
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'maze_objective')
config_path = os.path.join(local_dir, 'config.ini')




class MazeSimulationTrial:
    """
    The class to hold maze simulator execution parameters and results.
    """
    def __init__(self, maze_env, population):
        """
        Creates new instance and initialize fileds.
        Arguments:
            maze_env:   The maze environment as loaded from configuration file.
            population: The population for this trial run
        """
        # The initial maze simulation environment
        self.orig_maze_environment = maze_env
        # The record store for evaluated maze solver agents
        self.record_store = agent.AgentRecordStore()
        # The NEAT population object
        self.population = population

# The simulation results holder for a one trial.
# It must be initialized before start of each trial.
trialSim = None

def eval_fitness(genome, config, trialSim, time_steps=400):
    """
    Evaluates fitness of the provided genome.
    Arguments:
        genome_id:  The ID of genome.
        genome:     The genome to evaluate.
        config:     The NEAT configuration holder.
        time_steps: The number of time steps to execute for maze solver simulation.
    Returns:
        The phenotype fitness score in range (0, 10]
    """
    # run the simulation
    maze_env = copy.deepcopy(trialSim.orig_maze_environment)
    #cppn = feed_forward.FeedForwardNetwork.create(genome, config)
    network = DESNetwork(INPUT_SUBSTRATES, OUTPUT_SUBSTRATES, genome, DYNAMIC_PARAMS, config)
    control_net = network.create_phenotype_network()
    fitness = maze.maze_simulation_evaluate(
                                        env=maze_env, 
                                        net=control_net, 
                                        time_steps=time_steps)

    # Store simulation results into the agent record
    record = agent.AgentRecord(
        generation=trialSim.population.generation,
        agent_id=genome.key)
    record.fitness = fitness
    record.x = maze_env.agent.location.x
    record.y = maze_env.agent.location.y
    record.hit_exit = maze_env.exit_found
    record.species_id = trialSim.population.species.get_species_id(genome.key)
    record.species_age = record.generation - trialSim.population.species.get_species(genome.key).created
    # add record to the store
    trialSim.record_store.add_record(record)

    return fitness

def eval_genomes(genomes, config):
    """
    The function to evaluate the fitness of each genome in 
    the genomes list.
    Arguments:
        genomes: The list of genomes from population in the 
                 current generation
        config:  The configuration settings with algorithm
                 hyper-parameters
    """
    

    for genome_id, genome in genomes:
        genome.fitness = eval_fitness(genome_id, genome, config)

def run_experiment(config_file, maze_env, trial_out_dir, args=None, n_generations=100, silent=False):

    seed = 4224
    random.seed(seed)
    output_file = 'des_hyperneat_maze:' + args.maze + '_seed:' + str(seed) + '_generations: ' + str(args.generations) +'.txt'
    run_stats = st.Stats(output_file)
    # Config for CPPN.
    CONFIG = config.Config(genome.DefaultGenome, reproduction_deshyperneat.DefaultReproduction,
                                species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                config_file, layout_type=layout.Layout)

    pop = population_deshyperneat.Population(CONFIG)

    # Create the trial simulation
    global trialSim
    trialSim = MazeSimulationTrial(maze_env=maze_env, population=pop)

    stats = statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(reporting.StdOutReporter(True))

    para_evaluator = parallel.ParallelEvaluator(4, eval_fitness, trialSim)

    best_genome = pop.run(para_evaluator.evaluate, n_generations, stats=run_stats)
    print("es-hyperneat done")

    print('\nBest genome:\n{!s}'.format(best_genome))

    # Verify network output against training data.
    print('\nOutput:')
    network = DESNetwork(INPUT_SUBSTRATES, OUTPUT_SUBSTRATES, best_genome, DYNAMIC_PARAMS, CONFIG)
    winner_net = network.create_phenotype_network()

    # Save CPPN if wished reused and draw it to file along with the winner.
    '''with open(out_dir, 'wb') as output:
        pickle.dump(network, output, pickle.HIGHEST_PROTOCOL)'''
    draw_net(winner_net, filename=out_dir)

    maze_env = copy.deepcopy(trialSim.orig_maze_environment)

    positions = maze.maze_simulate_pathing(
                                        env=maze_env, 
                                        net=winner_net, 
                                        time_steps=400)

    visualize.show_path(maze_env, positions)

    run_stats.write_to_file()
    return best_genome, stats, CONFIG


# If run as script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="The maze experiment runner.")
    parser.add_argument('-m', '--maze', default='medium', 
                        help='The maze configuration to use.')
    parser.add_argument('-g', '--generations', default=500, type=int, 
                        help='The number of generations for the evolutionary process.')
    parser.add_argument('--width', type=int, default=400, help='The width of the records subplot')
    parser.add_argument('--height', type=int, default=400, help='The height of the records subplot')
    args = parser.parse_args()

    if not (args.maze == 'medium' or args.maze == 'hard' or args.maze == 'easy'):
        print('Unsupported maze configuration: %s' % args.maze)
        exit(1)

    maze_env_config = os.path.join('maze', '%s_maze.txt' % args.maze)
    maze_env = maze.read_environment(maze_env_config)

    config_path = os.path.join(local_dir, 'config.ini')

    trial_out_dir = os.path.join(out_dir, args.maze)

    WINNER = run_experiment(config_file=config_path, maze_env=maze_env, trial_out_dir=trial_out_dir, n_generations=args.generations, args=args)  # Only relevant to look at the winner.
    