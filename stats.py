


class Stats:


    def __init__(self, file_name):
        
        self.file_name = 'data/' + file_name

        # Key: gen_num, value: fitness
        self.champion_fitness = {}

        # Key gen_num, value: genome
        self.champions = {}

        # Key: gen_num, value (cppn size, ann size)
        self.champion_connections = {}

        #Key: gen_num, value: avg fitness
        self.average_fitness = {}
        self.std_fit = {}
        self.run_time = 0

        self.last_generation = 0

        self.mean_gene_distance = {}
        self.std_gene_distance = {}


    def generation_save(self,gen_num,  champ_fit, champ_con, avg_fit,std_fit):
        self.champion_fitness[gen_num] = champ_fit
        self.champion_connections[gen_num] = champ_con
        self.average_fitness[gen_num] = avg_fit
        self.std_fit[gen_num] = std_fit
        

    def species_save(self, gen_num, mean_dis, std_dis):
        self.mean_gene_distance[gen_num] = mean_dis
        self.std_gene_distance[gen_num] = std_dis

    def cham_save(self, gen_num, champion):
        self.champions[gen_num] = champion

    def final_save(self, gen_num, time):
        self.run_time = time
        self.last_generation = gen_num

    def write_to_file(self):
        with open(self.file_name, 'w') as file:
            file.write('Number of generations: ' + str(self.last_generation))
            file.write('\n')
            file.write('Total run time: ' + str(self.run_time) + ' seconds')            
            file.write('\n')
            file.write('gen_num, champ_fit, average_fit, std_fit, champ_cppn_con, champ_ann_con, mean_genetic_dist, std_genetic_dist  \n')
            for key in self.champion_fitness.keys():
                file.write(str(key) + ',' + 
                            str(self.champion_fitness[key]) + ',' + 
                            str(self.average_fitness[key]) + ',' + 
                            str(self.std_fit[key]) + ',' + 
                            str(self.champion_connections[key][0])+ ',' + 
                            str(self.champion_connections[key][1])+ ',' + 
                            str(self.mean_gene_distance[key])+ ',' + 
                            str(self.std_gene_distance[key]))
                file.write('\n')

            



