from random import choice, random
from configparser import ConfigParser



class Substrate:
    
    key = 0

    def __init__(self, substrate_type='hidden'):
        self.depth = 0
        self.substrate_type = substrate_type

    def mutate_depth(self):
        if random() <= 0.5 and self.depth>0:
            self.depth -=1
            return
        self.depth +=1

    @staticmethod
    def get_key():
        Substrate.key += 1
        return Substrate.key-1


class  Path:
    def __init__(self, input_substrate_key, output_substrate_key):
        self.input_substrate_key = input_substrate_key
        self.output_substrate_key = output_substrate_key
        self.enabled = True

class LayoutConfig:
    @staticmethod
    def get_config():
        config = ConfigParser()
        config.read('config.ini')
        return config

class Layout:
    def __init__(self, input_substrates, output_substrates):
        #(gene_key, gene) pair for gene sets
        self.input_substrates = input_substrates
        self.output_substrates = output_substrates
        self.substrates = {}
        self.paths = {}
        self.config = LayoutConfig.get_config()
        self.prob_add_substrate = float(self.config['layout']['prob_add_substrate'])
        self.prob_add_path = float(self.config['layout']['prob_add_path'])
        self.prob_mutate_depth = float(self.config['layout']['prob_mutate_depth'])

    def mutate(self):
        if random() <= self.prob_add_path:
            self.mutate_add_path()
        if random() <=self.prob_add_substrate:
            self.mutate_add_substrate()
        if random() <= self.prob_mutate_depth:
            random_substrate = choice(list(self.substrates.values()))
            random_substrate.mutate_depth()

    def add_path(self, input_key, output_key):
        key = (input_key, output_key)
        path = Path(input_key, output_key)
        self.paths[key] = path


    def mutate_add_path(self):
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
        
        self.add_path(in_substrate, out_substrate)

    def mutate_add_substrate(self):
        if not self.paths:
            self.mutate_add_path()
            return

        #Where to add the paths
        path_to_split = choice(list(self.paths.values()))
        print(path_to_split)
        #Add new path
        new_substrate_key = Substrate.get_key()
        new_substrate = Substrate()
        self.substrates[new_substrate_key] = new_substrate

        #Disable old path
        path_to_split.enabled = False

        #Add new paths
        self.add_path(path_to_split.input_substrate_key, new_substrate_key)
        self.add_path(new_substrate_key, path_to_split.output_substrate_key)

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
    input_subs = {}
    out_subs = {}
    for i in range(4):
        sub_key = Substrate.get_key()
        input_subs[sub_key] = Substrate()
    
    out_key = Substrate.get_key()
    out_subs[out_key] = Substrate()


    layout = Layout(input_subs, out_subs)

    for i in range(100):
        layout.mutate()
    print(layout.paths)


if __name__=='__main__':
    main()