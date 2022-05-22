import copy
import numpy as np
from hyperneat import get_cppn_bias, query_cppn, query_cppn_bias
from visualize_cppn import draw_es

import activations
import feed_forward
import recurrent


class DESNetwork:

    def __init__(self,input_substrates, output_substrates, genome, params, config):
        self.input_substrates = genome.input_substrates
        self.output_substrates = genome.output_substrates
        #Add the cords to the substrates
        if config.genome_config.num_inputs == 4:
            self.cppn_bias = False
        else:
            self.cppn_bias = True

        self.ann_bias = config.layout_config.enable_cppn_bias

        self.input_coordinates = []
        for i, substrate in enumerate(self.input_substrates.values()):
            substrate.coordinates = input_substrates[i]
            self.input_coordinates += substrate.coordinates
        
        self.output_coordinates = []
        for i, substrate in enumerate(self.output_substrates.values()):
            substrate.coordinates = output_substrates[i]
            self.output_coordinates += substrate.coordinates



        self.substrates = genome.substrates
        self.paths = genome.paths

        # DES ANN bias
        self.enable_des_bias = config.layout_config.enable_des_bias
        if self.enable_des_bias:
            self.bias_cppn = feed_forward.FeedForwardNetwork.create(genome.bias_cppn, config)

        #DES LEO 
        self.enable_leo = config.layout_config.enable_leo

        self.initial_depth = params["initial_depth"]
        self.max_depth = params["max_depth"]
        self.variance_threshold = params["variance_threshold"]
        self.band_threshold = params["band_threshold"]
        self.iteration_level = params["iteration_level"]
        self.division_threshold = params["division_threshold"]
        self.max_weight = params["max_weight"]
        self.connections = set()
        self.config = config
        # Number of layers in the network.
        self.activations = 2 ** params["max_depth"] + 1 #TODO check this out maybe multiply by num of substrates
        activation_functions = activations.ActivationFunctionSet()
        self.activation = activation_functions.get(params["activation"])

    def create_phenotype_network(self, filename=None):

        input_nodes = list(range(len(self.input_coordinates)))
        output_nodes = list(range(len(input_nodes), len(
            input_nodes)+len(self.output_coordinates)))
        hidden_idx = len(self.input_coordinates)+len(self.output_coordinates)

        coordinates, indices, draw_connections, node_evals = [], [], [], []
        nodes = {}

        coordinates.extend(self.input_coordinates)
        coordinates.extend(self.output_coordinates)
        indices.extend(input_nodes)
        indices.extend(output_nodes)

        #Unpack coordinates
        

        coords_to_id = dict(zip(coordinates, indices))


        hidden_nodes, connections = self.des_hyperneat()

        # Map hidden coordinates to their IDs.
        for x, y in hidden_nodes:
            coords_to_id[x, y] = hidden_idx
            hidden_idx += 1

        # For every coordinate:
        # Check the connections and create a node with corresponding connections if appropriate.
        for (x, y), idx in coords_to_id.items():
            for c in connections:
                if c.x2 == x and c.y2 == y:
                    draw_connections.append(c)
                    if idx in nodes:
                        initial, bias = nodes[idx]
                        initial.append((coords_to_id[c.x1, c.y1], c.weight))
                        nodes[idx] = (initial, bias)
                    else:
                        nodes[idx] = ([(coords_to_id[c.x1, c.y1], c.weight)], c.bias)

        # Combine the indices with the connections/links;
        # forming node_evals used by the RecurrentNetwork.

        #With CPPN layout bias

        for idx, links_bias in nodes.items():
            node_evals.append((idx, self.activation, sum, links_bias[1], 1.0, links_bias[0]))

        # Visualize the network?
        if filename is not None:
            draw_es(coords_to_id, draw_connections, filename)

        # This is actually a feedforward network.
        return recurrent.RecurrentNetwork(input_nodes, output_nodes, node_evals)

    @staticmethod
    def get_weights(p):
        """
        Recursively collect all weights for a given QuadPoint.
        """
        temp = []

        def loop(pp):
            if pp is not None and all(child is not None for child in pp.cs):
                for i in range(0, 4):
                    loop(pp.cs[i])
            else:
                if pp is not None:
                    temp.append(pp.w)
        loop(p)
        return temp

    def variance(self, p):
        """
        Find the variance of a given QuadPoint.
        """
        if not p:
            return 0.0
        return np.var(self.get_weights(p))

    def division_initialization(self, coord, outgoing, cppn, depth):
        """
        Initialize the quadtree by dividing it in appropriate quads.
        """
        root = QuadPoint(0.0, 0.0, 1.0, 1)
        q = [root]

        cppn = feed_forward.FeedForwardNetwork.create(cppn, self.config)

        while q:
            p = q.pop(0)

            p.cs[0] = QuadPoint(p.x - p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[1] = QuadPoint(p.x - p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[2] = QuadPoint(p.x + p.width/2.0, p.y +
                                p.width/2.0, p.width/2.0, p.lvl + 1)
            p.cs[3] = QuadPoint(p.x + p.width/2.0, p.y -
                                p.width/2.0, p.width/2.0, p.lvl + 1)

            for c in p.cs:
                c.w = query_cppn(coord, (c.x, c.y), outgoing,
                                 cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo)
                if self.ann_bias:
                    c.b = query_cppn_bias((0.0, 0.0), (c.x, c.y), outgoing,
                                    cppn, bias = self.cppn_bias)
                elif self.enable_des_bias:
                    c.b = get_cppn_bias(coord, self.bias_cppn, bias=self.cppn_bias)
                else:
                    c.b = 0
            if ((p.lvl < depth and p.lvl < self.max_depth) and self.variance(p)
                                                > self.division_threshold):
                for child in p.cs:
                    q.append(child)

        return root

    def pruning_extraction(self, coord, p, outgoing, cppn):
        """
        Determines which connections to express - high variance = more connetions.
        """
        if type(cppn) != feed_forward.FeedForwardNetwork:
            cppn = feed_forward.FeedForwardNetwork.create(cppn, self.config)
        for c in p.cs:
            d_left, d_right, d_top, d_bottom = None, None, None, None

            if self.variance(c) > self.variance_threshold:
                self.pruning_extraction(coord, c, outgoing, cppn)
            else:
                d_left = abs(c.w - query_cppn(coord, (c.x - p.width,
                                                      c.y), outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))
                d_right = abs(c.w - query_cppn(coord, (c.x + p.width,
                                                       c.y), outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))
                d_top = abs(c.w - query_cppn(coord, (c.x, c.y - p.width),
                                             outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))
                d_bottom = abs(c.w - query_cppn(coord, (c.x, c.y +
                                                        p.width), outgoing, cppn, self.max_weight, bias = self.cppn_bias, leo=self.enable_leo))

                con = None
                if max(min(d_top, d_bottom), min(d_left, d_right)) > self.band_threshold:

                    if outgoing:
                        con = Connection(coord[0], coord[1], c.x, c.y, c.w, c.b)
                    else:
                        con = Connection(c.x, c.y, coord[0], coord[1], c.w, c.b)
                if con is not None:
                    # Nodes will only connect upwards.
                    # If connections to same layer is wanted, change to con.y1 <= con.y2.
                    if not c.w == 0.0 and con.y1 < con.y2 and not (con.x1 == con.x2 and con.y1 == con.y2):
                        self.connections.add(con)

    def des_hyperneat(self):
        # Hidden nodes are (substrate_key, (nodes)) same with unexplored
        hidden_nodes, unexplored_hidden_nodes = {}, {}
        connections1, connections2, connections3 = set(), set(), set()
        #Sort the substrates
        self.substrates = self.sort_substrates()
        #Start the search from the input substrates
        for key, substrate in self.input_substrates.items():
            inputs = substrate.coordinates
            for path in self.paths.values():
                if path.key[0] == key:
                    for x, y in inputs:  # Explore from inputs.
                        root = self.division_initialization((x, y), True, path.cppn, 1)
                        self.pruning_extraction((x, y), root, True, path.cppn)
                        connections1 = connections1.union(self.connections)
                        if path.key[1] not in hidden_nodes:
                            hidden_nodes[path.key[1]] = set()
                        for c in connections1:
                            hidden_nodes[path.key[1]].add((c.x2, c.y2))
                        self.connections = set()

        unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        
        #Search for the hidden substrates
        for substrate in self.substrates.values():
            #First do a local search in this substrate
            for _ in range(self.iteration_level):  # Explore from hidden.
                if substrate.key in unexplored_hidden_nodes:
                    inputs = copy.deepcopy(unexplored_hidden_nodes[substrate.key])
                    for x, y in inputs:
                        root = self.division_initialization((x, y), True, substrate.cppn, substrate.depth)
                        self.pruning_extraction((x, y), root, True, substrate.cppn)
                        connections2 = connections2.union(self.connections)
                        for c in connections2:
                            hidden_nodes[substrate.key].add((c.x2, c.y2))
                        self.connections = set()
                        
                    unexplored_hidden_nodes[substrate.key] = hidden_nodes[substrate.key] - unexplored_hidden_nodes[substrate.key]
            if substrate.key in hidden_nodes:
                inputs = copy.deepcopy(hidden_nodes[substrate.key])
                # Then search to other substrates from the current
                for path in self.paths.values():
                    if path.key[0] == substrate.key:
                        for x, y in inputs:  # Explore from inputs.
                            root = self.division_initialization((x, y), True, path.cppn, 1)
                            self.pruning_extraction((x, y), root, True, path.cppn)
                            connections2 = connections2.union(self.connections)
                            if path.key[1] not in hidden_nodes:
                                hidden_nodes[path.key[1]] = set()
                            for c in connections2:
                                hidden_nodes[path.key[1]].add((c.x2, c.y2))
                            self.connections = set()
                            
                #If hidden nodes in new substrates are found they will be added to be searched for later
                unexplored_hidden_nodes = copy.deepcopy(hidden_nodes)
        #Seartch from the outputs
        for substrate in self.output_substrates.values():
            inputs = substrate.coordinates
            for path in self.paths.values():
                if path.key[1] == substrate.key:
                    for x, y in inputs:
                        root = self.division_initialization((x, y), False, path.cppn, 1)
                        self.pruning_extraction((x, y), root, False, path.cppn)
                        connections3 = connections3.union(self.connections)
                        self.connections = set()

        connections = connections1.union(connections2.union(connections3))  
        return self.clean_net(connections)


    def sort_substrates(self):
        '''
        Sorting substrates w
        '''
        sorted_substrate = {}
        no_in_path_substrates = {}

        edge_paths = []
        
        #Remove the paths from inputsubs
        for path in self.paths.keys():
            if path[0] not in self.input_substrates.keys():
                edge_paths.append(path)
        #Find substrates without incoming paths
        for substrate in self.substrates.values():
            in_path = False
            for path_key in edge_paths:
                if substrate.key == path_key[1]:
                    in_path = True
                    break
            if not in_path:
                no_in_path_substrates[substrate.key] = substrate
        while no_in_path_substrates:
            substrate = no_in_path_substrates.pop(list(no_in_path_substrates.keys())[0])
            sorted_substrate[substrate.key] = substrate

            for path in copy.deepcopy(edge_paths):
                if path[0]==substrate.key:
                    edge_paths.remove(path)
                    if path[1] not in self.output_substrates.keys():
                        new_sub = self.substrates[path[1]]
                        in_path = False
                        for path_key in edge_paths:
                            if new_sub.key == path_key[1]:
                                in_path = True
                                break
                        if not in_path:
                            no_in_path_substrates[new_sub.key] = new_sub

        return sorted_substrate


    def clean_net(self, connections):
        """
        Clean a net for dangling connections:
        Intersects paths from input nodes with paths to output.
        """
        connected_to_inputs = set(tuple(i)
                                  for i in self.input_coordinates)
        connected_to_outputs = set(tuple(i)
                                   for i in self.output_coordinates)
        true_connections = set()

        initial_input_connections = copy.deepcopy(connections)
        initial_output_connections = copy.deepcopy(connections)

        add_happened = True
        while add_happened:  # The path from inputs.
            add_happened = False
            temp_input_connections = copy.deepcopy(initial_input_connections)
            for c in temp_input_connections:
                if (c.x1, c.y1) in connected_to_inputs:
                    connected_to_inputs.add((c.x2, c.y2))
                    initial_input_connections.remove(c)
                    add_happened = True
        add_happened = True
        while add_happened:  # The path to outputs.
            add_happened = False
            temp_output_connections = copy.deepcopy(initial_output_connections)
            for c in temp_output_connections:
                if (c.x2, c.y2) in connected_to_outputs:
                    connected_to_outputs.add((c.x1, c.y1))
                    initial_output_connections.remove(c)
                    add_happened = True
        true_nodes = connected_to_inputs.intersection(connected_to_outputs)
        for c in connections:
            # Only include connection if both source and target node resides in the real path from input to output
            if (c.x1, c.y1) in true_nodes and (c.x2, c.y2) in true_nodes:
                true_connections.add(c)

        true_nodes -= (set(self.input_coordinates)
                       .union(set(self.output_coordinates)))

        return true_nodes, true_connections


class QuadPoint:
    """
    Class representing an area in the quadtree.
    Defined by a center coordinate and the distance to the edges of the area.
    """

    def __init__(self, x, y, width, lvl):
        self.x = x
        self.y = y
        self.w = 0.0
        self.width = width
        self.cs = [None] * 4
        self.lvl = lvl


class Connection:
    """
    Class representing a connection from one point to another with a certain weight.
    """

    def __init__(self, x1, y1, x2, y2, weight, bias = 0.0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.weight = weight
        self.bias = bias

    # Below is needed for use in set.
    def __eq__(self, other):
        if not isinstance(other, Connection):
            return NotImplemented
        return (self.x1, self.y1, self.x2, self.y2) == (other.x1, other.y1, other.x2, other.y2)

    def __hash__(self):
        return hash((self.x1, self.y1, self.x2, self.y2, self.weight))


def find_pattern(cppn, coord, res=60, max_weight=5.0):
    """
    From a given point, query the cppn for weights to all other points.
    This can be visualized as a connectivity pattern.
    """
    im = np.zeros((res, res))

    for x2 in range(res):
        for y2 in range(res):

            x2_scaled = -1.0 + (x2/float(res))*2.0
            y2_scaled = -1.0 + (y2/float(res))*2.0

            i = [coord[0], coord[1], x2_scaled, y2_scaled, 1.0]
            n = cppn.activate(i)[0]

            im[x2][y2] = n * max_weight

    return im