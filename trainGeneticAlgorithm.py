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
import pygame
from helpers.snakeGameGATrain import SnakeGameGATrain
from helpers import geneticAlgorithm as ga 

random.seed(123); #nrsharip

def main():
	"""Function to train the genetic algorithm for creating intelligent Snake Game agents."""
	game_fps = 3000
	chroms_per_gen = 200
	num_inputs = 9
	num_hidden_layer_nodes = 10
	bits_per_weight = 8
	num_outputs = 4
	total_bits = ((num_inputs+1)*num_hidden_layer_nodes + num_hidden_layer_nodes*(num_hidden_layer_nodes+1) + num_outputs*(num_hidden_layer_nodes + 1))*bits_per_weight
	population = ga.genPopulation(chroms_per_gen, total_bits)
	game = SnakeGameGATrain(game_fps, population, chroms_per_gen, bits_per_weight, num_inputs, num_hidden_layer_nodes, num_outputs)
	
	#pygame.font.init()

	while game.play:

		#game.clock.tick() # game.clock.tick(game.fps)
		
		game.move_snake()
		game.check_collisions()
		#check if snake is killed for not eating a fruit in a while
		game.update_frames_since_last_fruit()
		game.frames_alive += 1

		if game.restart == True:
			game.restart = False
			continue
		
		# nrsharip - we finished the 200's generation
		# if game.num_generations > 200:
		# 	return

		#game.redraw_window() #nrsharip
		
		#game.event_handler() #nrsharip
		
main()