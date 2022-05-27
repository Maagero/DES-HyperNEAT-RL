import numpy as np # linear algebra
import pandas as pd
#import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
#Dataset
iris  = pd.read_csv('iris_test/Iris.csv')
iris = iris.sample(frac=1).reset_index(drop=True)
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
X = np.array(X)

#One hot encode labels
one_hot_encoder = OneHotEncoder(sparse=False)
Y = iris.Species
Y = one_hot_encoder.fit_transform(np.array(Y).reshape(-1, 1))

#Devide in train and test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15)
X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)



import pickle
import sys
import os
import copy
import random
import argparse
from turtle import position



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
#INPUT_SUBSTRATES = [[(-1, 0), (-7/9, 0), (-5/9,0), (-3/9,0), (-1/9,0), (1/9,0), (3/9, 0), (5/9, 0), (7/9,0), (1,0)]]
INPUT_SUBSTRATES = [[(-1,0.0), (-1/3, 0.0), (1/3,0.0), (1,0.0)]]
#INPUT_SUBSTRATES = [[(-1, 0), (-3/5, 0), (-1/5,0), (1/5,0), (3/5,0), (1,0)], [(-1, 0), (-1/3, 0), (1/3,0), (1,0)]]
OUTPUT_SUBSTRATES = [[(-1.0, 0.0), (0.0,0.0), (1.0, 0.0)]]

def params(version):
    """
    ES-HyperNEAT specific parameters.
    """
    return {"initial_depth": 1,
            "max_depth": 3,
            "variance_threshold": 0.5,
            "band_threshold": 0.0,
            "iteration_level": 1,
            "division_threshold": 0.05,
            "max_weight": 5.0,
            "activation": "sigmoid"}

DYNAMIC_PARAMS = params(VERSION)

local_dir = os.path.dirname(__file__)
out_dir = os.path.join(local_dir, 'out')
out_dir = os.path.join(out_dir, 'maze_objective')
config_path = os.path.join(local_dir, 'config.ini')

def softmax(vector):
	e = np.exp(vector)
	return e / e.sum()

def eval_fitness(id, genome, config, time_steps=400):
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
    fitness = 0.0

    network = DESNetwork(INPUT_SUBSTRATES, OUTPUT_SUBSTRATES, genome, DYNAMIC_PARAMS, config)
    control_net = network.create_phenotype_network()

    labels = []
    for i, x in enumerate(X_train):
        labels.append(softmax(control_net.activate(x)))
    
    fitness = 1 - log_loss(Y_train, labels)
    if fitness<0.0:
        fitness = 0.0

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

def run_experiment(config_file, trial_out_dir, args=None, n_generations=100, silent=False):

    seed = 843249790
    random.seed(seed)
    output_file = 'iris:' + args.maze + '_seed:' + str(seed) + '_generations: ' + str(args.generations) +'.txt'
    run_stats = st.Stats(output_file)
    # Config for CPPN.
    CONFIG = config.Config(genome.DefaultGenome, reproduction_deshyperneat.DefaultReproduction,
                                species.DefaultSpeciesSet, stagnation.DefaultStagnation,
                                config_file, layout_type=layout.Layout)

    pop = population_deshyperneat.Population(CONFIG)

    # Create the trial simulation


    stats = statistics.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(reporting.StdOutReporter(True))


    best_genome = pop.run(eval_genomes, n_generations, stats=run_stats)
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


    config_path = os.path.join(local_dir, 'config.ini')

    trial_out_dir = os.path.join(out_dir, args.maze)

    WINNER = run_experiment(config_file=config_path, trial_out_dir=trial_out_dir, n_generations=args.generations, args=args)  # Only relevant to look at the winner.
    