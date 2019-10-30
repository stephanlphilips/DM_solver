from dataclasses import dataclass
import numpy as np

class ME_solver():
	"""docstring for ME_solver"""
	def __init__(self, size):
		super(ME_solver, self).__init__()
		self.size = size
		