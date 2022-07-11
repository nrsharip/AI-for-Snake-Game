#*********************************************************************************
#trainGeneticAlgorithm.py
#Author: Craig Haber
#5/9/2020
#This program was used to train a genetic algorithm in order to create 
#the intelligent Snake Game agents that can be observed in test_trained_agents.py
#For more detailed information, check out:
#https://craighaber.github.io/AI-for-Snake-Game/
#*********************************************************************************
#Instructions: 
#Simply run the module to observe how the genetic algorithm was trained in action,
#starting from randomized chromosomes. 
#Specific information about each population is saved in Gadata.txt in the same 
#folder as this program, and the file will be created if it does not already exist
#after the first generation.
#Also, for every 10 populations, the population is saved in a file 
#in the populations directory.
#*********************************************************************************
#Dependecies: 
#
#To run this module, you must have the module pygame installed.
#Type pip install pygame in the command prompt or terminal to install it.
#If necessary, more specific instructions for installing pygame are here:
#https://www.pygame.org/wiki/GettingStarted 
#
#Also, a Python version of 3.7 or higher is required.
#*********************************************************************************
import random #nrsharip
import multiprocessing
import os
from helpers import geneticAlgorithm as ga
from helpers.snakeGameGATrainMulti import SnakeGameGATrainMulti 

def starter(games, cur_chrom, chromosome):
	#print(multiprocessing.current_process().name, cur_chrom, sep=' ')

	cur_game = None
	while cur_game is None:
		for game in games:
			if game.released == True:
				game.released = False
				cur_game = game
				break

	cur_game.resetGame()
	cur_game.initWeights(chromosome)
	
	while cur_game.play:
		cur_game.move_snake()
		cur_game.check_collisions()
		#check if snake is killed for not eating a fruit in a while
		cur_game.update_frames_since_last_fruit()
		cur_game.frames_alive += 1

	result = (cur_chrom, cur_game.score, cur_game.calc_fitness())
	cur_game.released = True

	return result

if __name__ == "__main__":
	#random.seed(123); #nrsharip

	chroms_per_gen = 200
	bits_per_weight = 8

	num_inputs = 9
	num_hidden_layer_nodes = 10
	num_outputs = 4
	
	total_bits = (
		(num_inputs+1)*num_hidden_layer_nodes 
		+ num_hidden_layer_nodes*(num_hidden_layer_nodes+1) 
		+ num_outputs*(num_hidden_layer_nodes + 1)
	) * bits_per_weight
	
	population = ga.genPopulation(chroms_per_gen, total_bits)

	fitness_scores = [None] * chroms_per_gen
	game_scores = [None] * chroms_per_gen
	
	games = []
	high_score = 0
	cur_generation = 0

	# https://realpython.com/python-concurrency/
	with multiprocessing.Pool() as pool:
		# https://stackoverflow.com/questions/20353956/get-number-of-workers-from-process-pool-in-python-multiprocessing-module
		print('Number of CPUs: ', pool._processes)

		for i in range(pool._processes):
			games.append(SnakeGameGATrainMulti(bits_per_weight, num_inputs, num_hidden_layer_nodes, num_outputs))

		while True:
			results = pool.starmap(starter, [
				(games, x, population[x]) for x in range(chroms_per_gen)
			]) # , chroms_per_gen

			for result in results:
				(chrom_number, score, fitness) = result
				game_scores[chrom_number] = score
				fitness_scores[chrom_number] = fitness

			#Move onto next generation
			cur_generation +=1
			next_generation, best_individual, best_fitness, average_fitness = ga.createNextGeneration(population, fitness_scores, cur_generation)
			
			population = next_generation

			average_game_score = sum(game_scores)/len(game_scores)

			high_score_per_cur_gen = max(game_scores)

			if high_score_per_cur_gen > high_score:
				high_score = high_score_per_cur_gen

			print(cur_generation, high_score, average_game_score, high_score_per_cur_gen, average_fitness)

			for i in range(chroms_per_gen):
				fitness_scores[i] = None
				game_scores[i] = None

			#Write data about this generation to ga_data.txt
			file = open("GAdata.txt", "a+")
			file.write("Generation " + str(cur_generation) + "\n")
			file.write("Best Individual: " + str(best_individual) + "\n")
			file.write("Best Fitness: " + str(best_fitness) + "\n")
			file.write("Average Fitness:" + str(average_fitness) + "\n")
			file.write("Average Game Score:" + str(average_game_score) + "\n\n")
			file.write("\n")
			file.close()

			#Every 10 generations save the population to a file in the populations folder
			if cur_generation % 10 == 0:
				#Get the path of the directory with all the populations
				abs_file_path = os.path.join(os.getcwd(), "populations/population_" + str(cur_generation) + ".txt")
				file = open(abs_file_path, "a+")
				file.write(str(population))
				file.write("\n")
				file.close()