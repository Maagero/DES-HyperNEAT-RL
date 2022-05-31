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

sys.path.insert(1, '/Users/maagero/src/deshyperneatrl/')

import config
import genome
import reproduction
import species
import stagnation
import feed_forward
import population
import statistics
import reporting
import parallel
import stats as st
from visualize_cppn import draw_net
from substrate import Substrate
from hyperneat import create_phenotype_network

#Maze env
sys.path.insert(1, '/Users/maagero/src/deshyperneatrl/maze')
import visualize
import utils
import maze_environment as maze
import agent as agent

# Network inputs and expected outputs.
INPUT_COORDINATES = [(-1,-1.2), (-1/3, -1.2), (1/3, -1.2), (1,-1.2), (-1, -1), (-1/2, -1), (0,-1), (1/2,-1), (1,-1)]
HIDDEN_COORDINATES = [[(-1.0, 0.0), (-7/9, 0.0), (-5/9, 0.0), (-3/9, 0.0), (-1/9, 0.0), (1/9, 0.0), (3/9, 0.0), (5/9, 0.0), (7/9, 0.0), (1.0, 0.0)]]
OUTPUT_COORDINATES = [(-1, 1.0), (0.0, 1.0), (1, 1.0)]
ACTIVATIONS = len(HIDDEN_COORDINATES) + 2

SUBSTRATE = Substrate(
    INPUT_COORDINATES, OUTPUT_COORDINATES, HIDDEN_COORDINATES)

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
    cppn = feed_forward.FeedForwardNetwork.create(genome, config)
    control_net = create_phenotype_network(cppn, SUBSTRATE)
    fitness = maze.maze_simulation_evaluate(
                                        env=maze_env, 
                                        net=control_net, 
                                        time_steps=time_steps, activations=2)

    return fitness, genome.get_nodes_cppn(), control_net.get_nodes_cppn()

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

    seed = 11111
    random.seed()

    # Config for CPPN.
    CONFIG = config.Config(genome.DefaultGenome, reproduction.DefaultReproduction,
                                species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                config_file)

    pop = population.Population(CONFIG)

    output_file = 'hyperneat_maze:' + args.maze + '_seed:' + str(seed) + '_generations: ' + str(args.generations) +'.txt'
    run_stats = st.Stats(output_file)

    # Create the trial simulation
    global trialSim
    trialSim = MazeSimulationTrial(maze_env=maze_env, population=pop)

    stats = statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(reporting.StdOutReporter(True))

    para_eval = parallel.ParallelEvaluator(4, eval_fitness, trialSim).evaluate

    best_genome = pop.run(para_eval, n_generations, stats=run_stats)
    print("hyperneat done")

    print('\nBest genome:\n{!s}'.format(best_genome))

    # Verify network output against training data.
    print('\nOutput:')
    cppn = feed_forward.FeedForwardNetwork.create(best_genome, CONFIG)
    winner_net = create_phenotype_network(cppn, SUBSTRATE)

    maze_env = copy.deepcopy(trialSim.orig_maze_environment)

    positions = maze.maze_simulate_pathing(
                                        env=maze_env, 
                                        net=winner_net, 
                                        time_steps=400, activations=2)

    visualize.show_path(maze_env, positions)

    run_stats.write_to_file()

    return best_genome, stats, CONFIG


# If run as script.
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="The maze experiment runner.")
    parser.add_argument('-m', '--maze', default='medium', 
                        help='The maze configuration to use.')
    parser.add_argument('-g', '--generations', default=30, type=int, 
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
    