"""Handles node and connection genes."""
from random import random
import copy

# TODO: There is probably a lot of room for simplification of these classes using metaprogramming.
# TODO: Evaluate using __slots__ for performance/memory usage improvement.


class BaseGeneDES(object):
    """
    Handles functions shared by multiple types of genes (both node and connection),
    including crossover and calling mutation methods.
    """
    def __init__(self, key):
        self.key = key

    def init_attributes(self, config):
        key = next(config.cppn_indexer)
        g = config.genome_type(key)
        g.configure_new(config.genome_config)
        self.cppn = g


    def mutate(self, config):
        self.cppn.mutate(config.genome_config)

    def __str__(self):
        s = str(self.cppn)
        return 'CPPN for ' + str(self.__class__) + ' with key' + str(self.key) + '\n CPPN:' + s



class DefaultSubstrateGene(BaseGeneDES):
    def __init__(self, key):
        super().__init__(key)
        
    def init_attributes(self, config, coordinates = None):
        super().init_attributes(config)
        self.depth = 0
        self.coordinates = coordinates


    def mutate(self, config):
        super().mutate(config)
        if random() < config.substrate_depth_mutation:
            if self.depth == 0 or random() < 0.5:
                self.depth += 1
            else:
                self.depth -= 1   
    
    
    def copy(self):
        new_gene = self.__class__(self.key)
        new_gene.cppn = copy.deepcopy(self.cppn)
        new_gene.depth = self.depth
        return new_gene

    def crossover(self, gene2, config):
        assert self.key == gene2.key
        new_gene = self.__class__(self.key)
        key = next(config.cppn_indexer)
        new_cppn = config.genome_type(key)
        new_cppn.configure_crossover(self.cppn, gene2.cppn, config)
        new_gene.cppn = new_cppn
        if random() < 0.5:
            new_gene.depth = self.depth
        else:
            new_gene.depth = gene2.depth
        return new_gene

    def distance(self, other, config):
        d = 0
        if self.depth != 0 or other.depth != 0:
            d = abs(self.depth-other.depth) / max(self.depth, other.depth)
        if self.cppn.key != other.cppn.key:
            d += 1
        return d * config.compatibility_weight_coefficient

# TODO: Do an ablation study to determine whether the enabled setting is
# important--presumably mutations that set the weight to near zero could
# provide a similar effect depending on the weight range, mutation rate,
# and aggregation function. (Most obviously, a near-zero weight for the
# `product` aggregation function is rather more important than one giving
# an output of 1 from the connection, for instance!)
class DefaultPathGene(BaseGeneDES):


    def __init__(self, key):
        assert isinstance(key, tuple), "DefaultConnectionGene key must be a tuple, not {!r}".format(key)
        super().__init__(key)
        self.enabled = True

    def copy(self):
        new_gene = self.__class__(self.key)
        new_gene.cppn = copy.deepcopy(self.cppn)
        return new_gene
    
    def mutate(self, config):
            super().mutate(config)
            if random() < config.enabled_mutate_rate:
                self.enabled = random() < 0.5


    def crossover(self, gene2, config):
        assert self.key == gene2.key
        new_gene = self.__class__(self.key)
        key = next(config.cppn_indexer)
        new_cppn = config.genome_type(key)
        new_cppn.configure_crossover(self.cppn, gene2.cppn, config)
        new_gene.cppn = new_cppn
        return new_gene

    def distance(self, other, config):
        if self.cppn.key == other.cppn.key:
            return 0
        return 1 * config.compatibility_weight_coefficient