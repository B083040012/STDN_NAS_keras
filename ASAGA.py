from re import sub
import numpy as np
from criterion import eval_rmse
import math, random


class ASAGA_Searcher():
    
    def __init__(self, config, logger, model, val_loader, val_label):
        self.logger=logger
        self.model=model
        self.val_loader=val_loader
        self.val_label=val_label
        self.config=config
        self.generation_num=config["searching"]["generation_num"]
        self.population_num=config["searching"]["population_num"]
        self.annealing_ratio=config["searching"]["annealing_ratio"]
        self.initial_tmp=config["searching"]["initial_tmp"]
        self.crossover_rate=config["searching"]["crossover_rate"]
        self.num_choice=config["model"]["num_choice"]
        self.num_layers=config["model"]["num_layers"]
        self.threshold = config["dataset"]["threshold"] / config["dataset"]["volume_train_max"]
        self.short_term_lstm_seq_num = config["dataset"]["short_term_lstm_seq_num"]
        self.att_lstm_num = config["dataset"]["att_lstm_num"]
        self.long_term_lstm_seq_num = config["dataset"]["long_term_lstm_seq_num"]
    def search(self):
        """
        Initialization
        1. initialize the population 
        2. calculate fitness for each architecture
        """
        parent_population=[]
        for p in range(self.population_num):

            # conv size choice
            short_conv_choice = list(np.random.randint(self.num_choice, size=self.num_layers*self.short_term_lstm_seq_num))
            att_conv_choice = list(np.random.randint(self.num_choice, size=self.num_layers*self.att_lstm_num*self.long_term_lstm_seq_num))
            # pooling choice: [2,3,4] pool_size for max/avg pooling and no pooling: 3*2+1 choices
            short_pooling_choice = list(np.random.randint(self.num_choice*2+1, size=self.num_layers*self.short_term_lstm_seq_num))
            att_pooling_choice = list(np.random.randint(self.num_choice*2+1, size=self.num_layers*self.att_lstm_num*self.long_term_lstm_seq_num))
            # relu choice: relu, relu6, prelu
            short_relu_choice = list(np.random.randint(self.num_choice, size=self.num_layers*self.short_term_lstm_seq_num))
            att_relu_choice = list(np.random.randint(self.num_choice, size=self.num_layers*self.att_lstm_num*self.long_term_lstm_seq_num))
            # flow gate choice: sigmoid, relu6, tanh: 3 choices
            flow_gate_choice = list(np.random.randint(self.num_choice, size=3*self.short_term_lstm_seq_num))
            att_flow_gate_choice = list(np.random.randint(self.num_choice, size=3*self.att_lstm_num*self.long_term_lstm_seq_num))
            architecture=[short_conv_choice, att_conv_choice, short_pooling_choice, att_pooling_choice, short_relu_choice, att_relu_choice, flow_gate_choice, att_flow_gate_choice]

            # no avaliable condition currently
            parent_population.append(architecture)
        parent_population=np.array(parent_population)
        self.logger.info("evaluating parent population, wait for a sec...")
        parent_fitness=self.evaluate_architecture(parent_population, self.val_loader)
        tmp_best_loss=min(parent_fitness)
        tmp_best_index=parent_fitness.index(tmp_best_loss)
        self.logger.info("[Population Initialize] tmp_best_loss: %.5f" %(tmp_best_loss))
        
        """
        Generation Start
        1. loop (n/2) times:
            (a) generate two offsprings from two randomly chosen parent by crossover
            (b) using SA to select the parent of offspring
            (c) overwrite the old architecture with selected architecture
        2. lower temperature T
        """
        self.curr_tmp=self.initial_tmp
        global_best_loss=tmp_best_loss
        global_best_architecture=parent_population[tmp_best_index]
        self.logger.info("--------------[Generation Start]--------------")
        for gen in range(self.generation_num):
            if self.curr_tmp<=self.config["searching"]["final_tmp"]:
                break
            # randomly choose parent and avoid choose repeatly
            all_index_list = [pop_index for pop_index in range(0, self.population_num)]
            random.shuffle(all_index_list)
            for loop in range(int(self.population_num/2)):
                print("loop %d in gen %d" %(loop, gen))
                # index_list=[np.random.randint(low=0, high=self.population_num),np.random.randint(low=0, high=self.population_num)]
                index_list = [all_index_list.pop(), all_index_list.pop()]
                parent_list=[parent_population[index] for index in index_list]
                parent_subfitness=[parent_fitness[index] for index in index_list]
                offspring_list=self.crossover(parent_list)
                offspring_subfitness=self.evaluate_architecture(offspring_list, self.val_loader)
                new_fitness=self.selection(parent_subfitness, offspring_subfitness, parent_population, offspring_list, index_list)
                for i in range(len(new_fitness)):
                    parent_fitness[index_list[i]]=new_fitness[i]
            # tmp_best_index=np.argmin(parent_fitness)
            # tmp_best_loss=parent_fitness[tmp_best_index]
            tmp_best_loss=min(parent_fitness)
            tmp_best_index=parent_fitness.index(tmp_best_loss)
            tmp_best_architecture=parent_population[tmp_best_index]
            if global_best_loss>tmp_best_loss:
                global_best_loss=tmp_best_loss
                global_best_architecture=tmp_best_architecture
                self.logger.info("%%%%%%%%%%%%%%%%%%%%%%%%")
                self.logger.info("[Best Loss] gen:%d, gloabl_best_loss: %.5f" %(gen, global_best_loss))
                self.logger.info("%%%%%%%%%%%%%%%%%%%%%%%%")
            self.logger.info("[Generation %3d] temperature: %.5f, tmp_best: %.5f, gloabl_best_loss: %.5f" %(gen, self.curr_tmp, tmp_best_loss, global_best_loss))
            self.curr_tmp=self.curr_tmp*self.annealing_ratio
        self.logger.info("--------------[Generation End]--------------")

        return global_best_architecture, global_best_loss
                
    def crossover(self, parent_list):
        """
        crossover on single point
        crossover in each sublist (current), or crossover in the whole list????
        """
        offspring_list = list()
        offspring_list.append([])
        offspring_list.append([])
        for sub_index in range(0, len(parent_list[0])):
            prob = np.random.uniform(0,1)
            cross_point=np.random.randint(low=0, high=len(parent_list[0][sub_index]))
            tmp_sublist = [parent_list[0][sub_index], parent_list[1][sub_index]]
            if prob > self.crossover_rate:
                tmp_sublist[0][:cross_point]=parent_list[1][sub_index][:cross_point]
                tmp_sublist[1][cross_point:]=parent_list[0][sub_index][cross_point:]

            offspring_list[0].append(tmp_sublist[0])
            offspring_list[1].append(tmp_sublist[1])
        return offspring_list

    def selection(self, parent_subfitness, offspring_subfitness, parent_population, offspring_list, index_list):
        """
        select and overwrite
        maybe have to check the architecture is valid or not--> for now, there is no need to check this problem
        """
        new_fitness=parent_subfitness
        for i in range(len(parent_subfitness)):
            prob=np.random.uniform(0,1)
            accept_prob=math.exp(-(offspring_subfitness[i]-parent_subfitness[i])/self.curr_tmp)
            if parent_subfitness[i]>offspring_subfitness[i]:
                parent_population[index_list[i]]=offspring_list[i]
                new_fitness[i]=offspring_subfitness[i]
            elif prob<accept_prob:
                parent_population[index_list[i]]=offspring_list[i]
                new_fitness[i]=offspring_subfitness[i]
        return new_fitness

    def evaluate_architecture(self, architecture_list, val_loader):
        """
        Evaluate architecture in population,
        return the loss value of each architecture
        """
        architecture_loss=[]
        for index, architecture in enumerate(architecture_list):
            self.model.set_choice(architecture)
            y_pred = self.model.predict(val_loader)
            # denormalized
            y_pred=y_pred*self.config["dataset"]["volume_train_max"]
            loss_rmse = eval_rmse(self.val_label, y_pred, self.threshold)
            architecture_loss.append(loss_rmse)
        # architecture_loss=np.array(architecture_loss)
        return architecture_loss