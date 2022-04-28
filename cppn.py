from itertools import count
from secrets import choice, random
import genes
from configparser import ConfigParser
from graphs import creates_cycle
    

class CPPNConfigs:
    """Holds inforamtion for the CPPN class"""
    def __inif__(self):

        self.config = self.get_config()
        self.num_input = int(self.config['CPPN']['num_inputs'])
        self.num_outputs = int(self.config['CPPN']['num_outputs'])
        self.initially_connected = bool(self.config['CPPN']['initially_connected'])
        self.node_add_prob = float(self.config['CPPN']['node_add_prob'])
        self.conn_add_prob = float(self.config['CPPN']['conn_add_prob'])

        self.compatibility_disjoint_coefficient = float(self.config['CPPN']['compatibility_disjoint_coefficient'])
        self.compatibility_weight_coefficient = float(self.config['CPPN']['compatibility_weight_coefficient'])

        # The input keys are allways negative and the output keys are allways 0 and up
        self.input_keys = [-i - 1 for i in range(self.num_inputs)]
        self.output_keys = [i for i in range(self.num_outputs)]


        self.node_indexer = None

    @staticmethod
    def get_config():
        config = ConfigParser()
        config.read('config.ini')
        return config

    def get_new_node_key(self, node_dict):
        if self.node_indexer is None:
            self.node_indexer = count(max(list(node_dict)) + 1)

        new_id = next(self.node_indexer)

        assert new_id not in node_dict

        return new_id


class CPPN:

    def __init__(self, key):

        self.key = key

        # (gene_key, key) pairs
        self.connections = {}
        self.nodes = {}

        self.fitness = None

    def configure_new(self, config):
        """Configure CPPN from the configureations"""
        #create nodes for the output nodes
        for node_key in config.output_keys:
            self.nodes[node_key] = self.create_node(node_key)
                
        #TODO: Might add default connectivity here, or not, we'll see

    def configure_crossover(self, cppn1, cppn2):
        #Configre new CPPN based on 2 parents
        if cppn1.fitness > cppn2.fitness:
            parent1, parent2 = cppn1, cppn2
        else:
            parent1, parent2 = cppn2, cppn1

        #Inherit con genes
        for key, cg1 in parent1.connections.items():
            cg2 = parent2.connections.get(key)
            if cg2 is None:
                self.connections[key] = cg1.copy()
            else:
                self.connections[key] = cg1.crossover(cg2)

    def mutate(self, config):
        """Mutates Genome"""

        # Mutate CPPN structure
        if random() < config.node_add_prob:
                self.mutate_add_node(config)

        if random() < config.conn_add_prob:
            self.mutate_add_connection(config)

        # Mutate connection genes.
        for cg in self.connections.values():
            cg.mutate(config)

        # Mutate node genes (bias, response, etc.).
        for ng in self.nodes.values():
            ng.mutate(config)

    def mutate_add_node(self, config):
        if not self.connections:
            self.mutate_add_connection(config)

            return
        conn_to_split = choice(list(self.connections.values()))
        new_node_id = config.get_new_node_key(self.nodes)
        ng = self.create_node(new_node_id)
        self.nodes[new_node_id] = ng

        conn_to_split.enabled = False

        i, o = conn_to_split.key

        self.add_connection(config, i, new_node_id, 1, True)
        self.add_connection(config, new_node_id, o, conn_to_split.weight, True)

    def add_connection(self, config, input_key, output_key, weight, enabled):
        assert output_key >= 0

        key = (input_key, output_key)
        connection = genes.ConnectionGene(key)
        connection.init_attributes(config)
        connection.weigth = weight
        connection.enabled = enabled
        self.connections[key] = connection
        
    def mutate_add_connection(self, config):
        possible_outputs = list(self.nodes)
        out_node = choice(possible_outputs)

        possible_inputs = possible_outputs + config.input_keys
        in_node = choice(possible_inputs)

        #Don't allow duplicated
        key = (in_node, out_node)
        if key in self.connections:
            self.connections[key].enabled = True
            return
        
        #Don't allow a connection between output nodes
        if in_node in config.output_keys and out_node in config.output_keys:
            return

        #Check for cycles
        if creates_cycle(list(self.connections), key):
            return

        cg = self.create_connection(in_node, out_node)
        self.connections[cg.key] = cg


    #TODO: Consider making delete mutations as well

    def distance(self, other, config):
        """Returns the distnace from this CPPn to another"""

        node_distance = 0.0
        if self.nodes or other.nodes:
            disjoint_nodes = 0
            for k2 in other.nodes:
                if k2 not in self.nodes:
                    disjoint_nodes +=1

            for k1, n1 in self.nodes.items():
                n2 = other.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    node_distance += n1.distance(n2, config)

            max_nodes = max(len(self.nodes), len(other.nodes))
            node_distance = (node_distance + config.compatibility_disjoint_coefficient * disjoint_nodes) / max_nodes

        connection_distance = 0.0
        if self.connections or other.connections:
            disjoint_connections = 0
            for k2 in other.connections:
                if k2 not in self.connections:
                    disjoint_connections += 1
            
            for k1, c1 in self.connections.items():
                c2 = other.connections.get(k1)
                if c2 is None:
                    disjoint_connections + 1
                else:
                    connection_distance += c1.distance(c2, config)
            max_conn = max(len(self.connections), len(other.connections))
            connection_distance = (connection_distance +
                                   (config.compatibility_disjoint_coefficient *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def size(self):
        num_enabled_connections = sum([1 for cg in self.connections.values() if cg.enabled])
        return len(self.nodes), num_enabled_connections

    @staticmethod
    def create_node(node_id):
        return genes.NodeGene(node_id)

    @staticmethod
    def create_connection(input_id, output_id):
        return genes.ConnectionGene((input_id, output_id))


if __name__=='__main__':
    config = CPPNConfigs()
    cppn = CPPN(1)
    