#**************************************************************************************
#snakeGameGATrain.py
#Author: Craig Haber
#5/9/2020
#Module with the SnakeGameGATrain class that is instantiated in testGenticAlgorithm.py
#to train a population of intelligent Sanke Game agents.
#*************************************************************************************

#import pygame
#import random
#import collections
from helpers.snake import Snake
from helpers.snakeGameGATest import SnakeGameGATest
from helpers import neuralNetwork as nn
from helpers import geneticAlgorithm as ga 
#import os

class SnakeGameGATrainMulti(SnakeGameGATest):
	"""Class framework to train a population of intelligent Snake Game agents through a genetic algorithm.

	This class inherets from the SnakeGameGATest class (since the two classes are quite similar), which
	inherets from the SnakeGame class.
	The actual methods if the genetic algorithm are in geneticAlgorithm.py.

	Attributes:
		self.cur_chrom: An index of the current chromosome being tested in self.population.
		self.frames_alive: The number of frames the current agent has been alive, used for the fitness function.
		self.population: A list of all the chromosomes in the population for the current generation.
		self.weights: The weights for the neural network converted from the current chromosome.
		self.fitness_scores: A list of all the fitness scores, each index corresponding to a chromosome in self.population.
		self.game_scores: A list of all the in-game scores, each index corresponding to a chromosome in self.population.
		self.num_generation: The number of generations that have passed.
	"""

	def __init__(self, bits_per_weight, num_inputs, num_hidden_layer_nodes, num_outputs):
		"""Initializes the SnakeGameGATrain class

		Arguments:
			fps: The frame rate of the game.
			population: A list of all the chromosomes in the population for the current generation.
			bits_per_weight: The number of bits per each weight in the nueral network.
			num_inputs: The number of inputs in the neural network.
			num_hidden_layer_nodes: The number of nodes per each of the 2 hidden layers in the neural network.
			num_ouputs: The number of outputs in the neural network.
		"""
		super().__init__(300000, "", bits_per_weight, num_inputs, num_hidden_layer_nodes, num_outputs)
		
		self.score = 0
		self.frames_alive = 0
		self.frames_since_last_fruit = 0
		self.play = False
		self.released = True

	def resetGame(self):
		self.snake = Snake(self.rows, self.cols)
		self.generate_fruit()

		self.score = 0
		self.frames_alive = 0
		self.frames_since_last_fruit = 0
		self.play = True

	def initWeights(self, chromosome):
		self.weights = nn.mapChrom2Weights(
			chromosome, 
			self.bits_per_weight, 
			self.num_inputs, 
			self.num_hidden_layer_nodes, 
			self.num_outputs
		)

	def game_over(self):
		"""Function that restarts the game upon game over.

		This overrides the method in the SnakeGameGATest superclass."""

		self.play = False

	def calc_fitness(self):
		"""Function to calculate the fitness score for a chromosome.
		
		Returns: A fitness score.
		"""

		frame_score = self.frames_alive
		#If the frames since the last fruit was eaten is at least 50
		if self.frames_since_last_fruit >= 50:
			#Subtract the number of frames since the last fruit was eaten from the fitness
			#This is to discourage snakes from trying to gain fitness by avoiding fruit
			frame_score = self.frames_alive - self.frames_since_last_fruit
			#Ensure we do not multiply fitness by a factor of 0
			if frame_score <= 0:
					frame_score = 1

		_1 = (self.score*2)**2 # nrsharip
		_2 = frame_score**1.5  # nrsharip
		_3 = _1 * _2           # nrsharip ((self.score*2)**2)*(frame_score**1.5)

		return (_3, frame_score)
			
