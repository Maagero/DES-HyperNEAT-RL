from random import choice, random
from configparser import ConfigParser
from itertools import count
from config import ConfigParameter
import copy
from genes_deshyperneat import DefaultPathGene, DefaultSubstrateGene
from genome import DefaultGenome

class LayoutConfig:

    def __init__(self, params, genome_config):
        self.substrate_gene_type = params['substrate_gene_type']
        self.path_gene_type = params['path_gene_type']
        self.genome_type = params['genome_type']
        self.genome_config = genome_config
        self.substrate_indexer = count(1)
        self.cppn_bias_indexer = count(1)
        #TODO add all layout params here for
        self._params = [ConfigParameter('prob_add_substrate', float),
                        ConfigParameter('prob_add_path', float),
                        ConfigParameter('prob_delete_substrate', float),
                        ConfigParameter('prob_delete_path', float),
                        ConfigParameter('num_input_substrates', int),
                        ConfigParameter('num_output_substrates', int),
                        ConfigParameter('enabled_mutate_rate', float),
                        ConfigParameter('substrate_depth_mutation', float),
                        ConfigParameter('enable_des_bias', bool),
                        ConfigParameter('enable_cppn_bias', bool),
                        ConfigParameter('enable_leo', bool)]
                        

        self.cppn_indexer = count(1)
        for p in self._params:
            setattr(self, p.name, p.interpret(params))

        self.input_substrates = [-i -1 for i in range(self.num_input_substrates)]
        self.output_substrates = [i for i in range(self.num_output_substrates)]
    def get_new_substrate_key(self, substrate_dict):
        if self.substrate_indexer is None:
            if substrate_dict:
                self.substrate_indexer = count(max(list(substrate_dict)) + 1)
            else:
                self.substrate_indexer = count(max(list(substrate_dict)) + 1)

        new_id = next(self.substrate_indexer)

        assert new_id not in substrate_dict

        return new_id

class Layout:

    @classmethod
    def parse_config(cls, param_dict, genome_config):
        param_dict['substrate_gene_type'] = DefaultSubstrateGene
        param_dict['path_gene_type'] = DefaultPathGene
        param_dict['genome_type'] = DefaultGenome
        return LayoutConfig(param_dict, genome_config)


    def __init__(self, key):
        self.key = key
        self.input_substrates = {}
        self.output_substrates = {}
        self.substrates = {}
        self.paths = {}
        self.fitness = None



    def configure_new(self, config):
        #Create substrate genes for the output substrates
        for sub_key in config.input_substrates:
            self.input_substrates[sub_key] = self.create_substrate(config, sub_key)

        for sub_key in config.output_substrates:
            self.output_substrates[sub_key] = self.create_substrate(config, sub_key)

        for input in self.input_substrates.keys():
            for output in self.output_substrates.keys():
                self.add_path(config, input, output, True)
        
        #Create a CPPN used for bias to all nodes
        if config.enable_des_bias:
            key = next(config.cppn_bias_indexer)
            cppn = config.genome_type(key)
            cppn.configure_new(config.genome_config)
            self.bias_cppn = cppn

    
    def configure_crossover(self, layout1, layout2, config):

        for sub_key in config.input_substrates:
            self.input_substrates[sub_key] = self.create_substrate(config, sub_key)

        for sub_key in config.output_substrates:
            self.output_substrates[sub_key] = self.create_substrate(config, sub_key)

        if layout1.fitness > layout2.fitness:
            parent1, parent2 = layout1, layout2
        else:
            parent1, parent2 = layout2, layout1

        #Inherit path genes
        for key, pg1 in parent1.paths.items():
            pg2 = parent2.paths.get(key)
            if pg2 is None:
                self.paths[key] = pg1.copy()
            else:
                pg1.cppn.fitness = parent1.fitness
                pg2.cppn.fitness = parent2.fitness
                self.paths[key] = pg1.crossover(pg2, config)
            
        parent1_set = parent1.substrates
        parent2_set = parent2.substrates

        for key, sg1 in parent1_set.items():
            sg2 = parent2_set.get(key)
            assert key not in self.substrates
            if sg2 is None:
                self.substrates[key] = sg1.copy()
            else:
                sg1.cppn.fitness = parent1.fitness
                sg2.cppn.fitness = parent2.fitness
                self.substrates[key] = sg1.crossover(sg2, config)
        
        #CPPN bias
        if config.enable_des_bias:
            if parent1.bias_cppn.key == parent2.bias_cppn.key:
                parent1.bias_cppn.fitness = parent1.fitness
                parent2.bias_cppn.fitness = parent2.fitness
                cppn = config.genome_type(parent1.bias_cppn.key)
                cppn.configure_crossover(parent1.bias_cppn, parent2.bias_cppn, config.genome_config)
                self.bias_cppn = cppn
            else:
                cppn = config.genome_type(parent1.bias_cppn.key)
                cppn.configure_new(config.genome_config)
                self.bias_cppn = cppn

        


    def mutate(self, config):
        #Mutate the layout
        if random() <= config.prob_add_path:
            self.mutate_add_path(config)
        if random() <=config.prob_add_substrate:
            self.mutate_add_substrate(config)
        if random() <= config.prob_delete_path:
            self.mutate_delete_path()
        if random() <= config.prob_delete_substrate:
            self.mutate_delete_substrate(config)

        #Mutate the CPPNs
        for sg in self.substrates.values():
            sg.mutate(config)

        for pg in self.paths.values():
            pg.mutate(config)
        
        #Mutate the bias CPPN
        if config.enable_des_bias:
            self.bias_cppn.mutate(config.genome_config)

    def add_path(self, config, input_key, output_key, enabled):
        #Check if values are valid
        assert isinstance(input_key, int)
        assert isinstance(output_key, int)
        assert output_key >= 0
        assert isinstance(enabled, bool)

        #Adds path
        key = (input_key, output_key)
        path = config.path_gene_type(key)
        path.init_attributes(config)
        path.enabled = enabled
        self.paths[key] = path


    def mutate_add_path(self, config):
        possible_outputs = list(self.substrates) + list(self.output_substrates)
        out_substrate = choice(possible_outputs)
        possible_inputs = list(self.substrates) + list(self.input_substrates)
        in_substrate = choice(possible_inputs)

        #Dont allow dupes
        key = (in_substrate, out_substrate)
        if key in self.paths:
            return

        #Don't allow cycles
        if self.check_cycles(list(self.paths), key):
            return
        
        #Add path
        pg = self.create_path(config, in_substrate, out_substrate)
        self.paths[pg.key] = pg


    def mutate_add_substrate(self, config):
        if not self.paths:
            self.mutate_add_path(config)
            return

        #Where to add the paths
        path_to_split = choice(list(self.paths.values()))
        #Add new path
        new_substrate_id = config.get_new_substrate_key(self.substrates)
        sg = self.create_substrate(config, new_substrate_id)
        self.substrates[new_substrate_id] = sg

        #Disable old path
        path_to_split.enabled = False

        #Add new paths
        i, o = path_to_split.key
        self.add_path(config, i, new_substrate_id, True)
        self.add_path(config, new_substrate_id, o, True)


    def mutate_delete_substrate(self, config):
        available_nodes = list(self.substrates.keys())
        if not self.substrates:
            return -1

        del_key = choice(available_nodes)

        paths_to_del = set()

        for k, v in self.paths.items():
            if del_key in v.key:
                paths_to_del.add(v.key)
        
        for key in paths_to_del:
            del self.paths[key]

        del self.substrates[del_key]

        return del_key


    def mutate_delete_path(self):
        if self.paths:
            key = choice(list(self.paths.keys()))
            del self.paths[key]


    def distance(self, other, config):

        #substrate gene distance
        substrate_distance = 0.0
        if self.substrates or other.substrates:
            disjoint_substrates = 0
            for k2 in other.substrates:
                if k2 not in self.substrates:
                    disjoint_substrates += 1
            for k1, s1 in self.substrates.items():
                s2 = other.substrates.get(k1)
                if s2 is None:
                    disjoint_substrates += 1
                else:
                    substrate_distance += s1.distance(s1, config)
            max_substrates = max(len(self.substrates), len(other.substrates))
            substrate_distance = (substrate_distance +
                             (config.compatibility_disjoint_coefficient *
                              disjoint_substrates)) / max_substrates

        paths_distance = 0.0
        if self.paths or other.paths:
            disjoint_paths = 0
            for k2 in other.paths:
                if k2 not in self.paths:
                    disjoint_paths += 1

            for k1, p1 in self.paths.items():
                p2 = other.paths.get(k1)
                if p2 is None:
                    disjoint_paths += 1
                else:
                    # Homologous genes compute their own distance value.
                    paths_distance += p1.distance(p2, config)

            max_path = max(len(self.paths), len(other.paths))
            paths_distance = (paths_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_paths)) / max_path

        distance = paths_distance + substrate_distance
        return distance


    def size(self):
        """
        Returns genome 'complexity', taken to be
        (number of nodes, number of enabled connections)
        """
        num_enabled_paths = sum([1 for pg in self.paths.values() if pg.enabled])
        return len(self.substrates), num_enabled_paths

    def __str__(self):
        s = "Key: {0}\nFitness: {1}\nSubstrates:".format(self.key, self.fitness)
        for k, sg in self.substrates.items():
            s += "\n\t{0} {1!s}".format(k, sg)
        s += "\nPaths:"
        paths = list(self.paths.values())
        for p in paths:
            s += "\n\t" + str(p)
        return s


    @staticmethod
    def create_substrate(config, sub_id):
        substrate = config.substrate_gene_type(sub_id)
        substrate.init_attributes(config)
        return substrate

    @staticmethod
    def create_path(config, input_id, output_id):
        key = (input_id, output_id)
        path = config.path_gene_type(key)
        path.init_attributes(config)
        return path






    #Credits to python-neat
    def check_cycles(self, paths, test):
        i, o = test
        if i == o:
            return True

        visited = {o}
        while True:
            num_added = 0
            for a, b in paths:
                if a in visited and b not in visited:
                    if b == i:
                        return True

                    visited.add(b)
                    num_added += 1

            if num_added == 0:
                return False

def main():
    pass


if __name__=='__main__':
    main()