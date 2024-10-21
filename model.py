import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from jsonschema import validate

import numpy as np

from stormpy import SparseMatrixBuilder, StateLabeling, SparseModelComponents
import stormpy

with open("./cvas.schema.json", "r") as f:
	cvas_schema = json.load(f)

with open("./result.schema.json", "r") as f:
	result_schema = json.load(f)

def validate_data(data, schema):
	try:
		validate(instance=data, schema=schema)
	except Exception:
		raise ValueError("Invalid CVAS!")

@dataclass
class StateUpdate(object):
	rate : int
	vector : tuple

	def __init__(self, rate, vector) -> None:
		self.rate = rate
		self.vector = tuple(vector)

@dataclass_json
@dataclass
class CVAS(object):
	'''
A continuous time Vector Addition System. The specification for this object
is contained in the file cvas.schema.json
	'''
	dim : int
	initialState : tuple
	stateUpdates : list

	def __init__(self, data : dict) -> None:
		self.dim = data["dim"]
		self.initialState = tuple(data["initialState"])
		self.stateUpdates = [StateUpdate(d["rate"], d["vector"]) for d in data["stateUpdates"]]

	def __post_init__(self):
		'''
	Validates the class via the cvas json schema
		'''
		validate_data(self.to_dict(), cvas_schema)

@dataclass
class Frame(object):
	timeStep : int
	stateFrames : list

@dataclass
class StateFrame(object):
	state : tuple
	probability : float

@dataclass
class CVASResult(object):
	dim : int
	frames : list

	def __init__(self, dim, frames=[].copy()) -> None:
		self.dim = dim
		self.frames = frames

	def validate(self):
		validate_data(self.to_dict(), cvas_schema)

class Entry:
	def __init__(self, col : int, val : float):
		self.col : int = col
		self.val : float = val

	def __eq__(self, other):
		return other.col == self.col

	def __gt__(self, other):
		return self.col > other.col

	def __le__(self, other):
		return not self > other

	def __lt__(self, other):
		return self.col < other.col

	def __ge__(self, other):
		return not self < other

	def __str__(self):
		return f"{self.col}:{self.val}"

class RandomAccessSparseMatrixBuilder:
	'''
	A wrapper class for random entry into storm's sparse matrix builder
	'''
	def __init__(self):
		self.from_list = []
		self.exit_rates = []

	def add_next_value(self, row : int, col : int, val : float):
		while len(self.from_list) <= row:
			self.from_list.append([])
		self.from_list[row].append(Entry(col, val))

	def to_smb(self):
		'''
		Creates a stormpy.SparseMatrixBuilder
		'''
		matrix_builder = SparseMatrixBuilder()
		for row in range(len(self.from_list)):
			self.from_list[row].sort()
			if len(self.from_list[row]) == 0:
				matrix_builder.add_next_value(row, row, 1.0)
			for entry in self.from_list[row]:
				col = entry.col
				val = entry.val
				if row == col:
					assert(len(self.from_list[row]) == 1)
					matrix_builder.add_next_value(row, col, 1.0)
					break
				matrix_builder.add_next_value(row, col, val)
		return matrix_builder

	def build(self):
		'''
		Creates a stormpy SparseMatrix
		'''
		matrix_builder = self.to_smb()
		return matrix_builder.build()

	def size(self):
		return len(self.from_list)

	def add_exit_rate(self, idx : int, rate : float):
		while len(self.exit_rates) <= idx:
			self.exit_rates.append(None)
		self.exit_rates[idx] = rate

	def assert_all_entries_correct(self):
		# assert(len(self.exit_rates) == len(self.from_list))
		for i in range(len(self.from_list)):
			# print(f"{i}: {self.exit_rates[i]}, {[str(entry) for entry in self.from_list[i]]}")
			if len(self.from_list[i]) == 0:
				self.exit_rates[i] = None
				assert(self.exit_rates[i] is None)
				continue
			max_entry = max(self.from_list[i])
			max_rate = max_entry.val
			if not self.exit_rates[i] >= max_rate:
				print(f"Error: {self.exit_rates[i]} < {max_rate} (state index {i})")
			assert(self.exit_rates[i] >= max_rate or math.isclose(max_rate, self.exit_rates[i]))

class Explorer(object):
	def __init__(self, filename : str, dim_max = 100) -> None:
		with open(filename, 'r') as f:
			self.cvas = CVAS(json.loads(f.read()))
		self.__currentState = self.cvas.initialState
		self.__exploredStates = set()
		self.__stateToIndex = dict()
		self.__indexToState = dict() # could use an array. am lazy
		self.__dim_max = dim_max
		self.__matrixBuilder = None

	def build(self):
		# build with bound
		self.__exploredStates = dict()
		self.__matrixBuilder = RandomAccessSparseMatrixBuilder()
		queue = [(self.cvas.initialState, 0)]
		self.__stateToIndex[self.cvas.initialState] = 0
		self.__indexToState[0] = self.cvas.initialState
		nextIdx = 1
		while len(queue) > 0:
			# dequeue the first state
			s, idx = queue.pop()
			print(f"Exploring state {s} (idx: {idx})")
			self.__exploredStates[s] = nextIdx
			exitRate = 0.0
			# Enqueue successors
			for sNxt, rate in self.__successors(s):
				exitRate += rate
				# Choose the next index if it exists
				succIdx = None
				if sNxt in self.__stateToIndex:
					succIdx = self.__stateToIndex[sNxt]
				else:
					succIdx = nextIdx
					nextIdx += 1
					self.__stateToIndex[sNxt] = succIdx
					self.__indexToState[succIdx] = sNxt
				self.__matrixBuilder.add_next_value(idx, succIdx, rate)
				# Only enqueue states we haven't explored yet
				if not sNxt in self.__exploredStates:
					queue.append((sNxt, succIdx))
			self.__matrixBuilder.add_exit_rate(idx, exitRate)
		self.__matrixBuilder.assert_all_entries_correct()
		return self.__matrixBuilder.build()

	def __successors(self, state : tuple) -> list:
		successors = []
		for update in self.cvas.stateUpdates:
			nextCandidate = tuple(np.add(state, update.vector))
			if not np.all([d >= 0 and d <= self.__dim_max for d in nextCandidate]):
				continue
			successors.append((nextCandidate, self.__rate(state, update.vector, update.rate)))
		return successors

	def __rate(self, state : tuple, update : tuple, rConst : float):
		assert(len(state) == len(update))
		return rConst * np.prod([state[i] ** max(-update[i], 0) for i in range(len(state))])

	def stateCount(self) -> int:
		if self.__matrixBuilder is None:
			raise Exception("Must build model first!")
		return self.__matrixBuilder.size()

	def createLabels(self):
		assert(self.__matrixBuilder is not None)
		labeling = StateLabeling(self.__matrixBuilder.size())
		# Add the initial state
		labeling.add_label("init")
		labeling.add_label_to_state("init", 0)
		for i in range(self.__matrixBuilder.size()):
			label = f"state_{i}"
			labeling.add_label(label)
			labeling.add_label_to_state(label, i)
		return labeling

	def createModel(self):
		matrix = self.build()
		labels = self.createLabels()
		print(labels)
		components = SparseModelComponents(matrix, labels, {}, rate_transitions=True)
		model = stormpy.storage.SparseCtmc(components)
		return model

	def state(self, idx : int):
		return self.__indexToState[idx]
