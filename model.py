import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from jsonschema import validate

import numpy as np

from stormpy import SparseMatrixBuilder, StateLabeling, SparseModelComponents
import stormpy
import math

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
	rate: int
	vector: tuple

	def __init__(self, rate, vector, needed=None, ignore=False) -> None:
		self.rate = rate
		self.vector = tuple(vector)
		self.needed = needed
		self.ignore = ignore

@dataclass_json
@dataclass
class CVAS(object):
	'''
A continuous time Vector Addition System. The specification for this object
is contained in the file cvas.schema.json
	'''
	dim: int
	initialState: tuple
	stateUpdates: list

	def __init__(self, data: dict) -> None:
		self.dim = data["dim"]
		self.initialState = tuple(data["initialState"])
		self.stateUpdates = [StateUpdate(d["rate"], d["vector"],
                                   None if "needed" not in d else d["needed"],
                                   False if "ignore" not in d else bool(d["ignore"])) for d in data["stateUpdates"]]

	def __post_init__(self):
		'''
	Validates the class via the cvas json schema
		'''
		validate_data(self.to_dict(), cvas_schema)

@dataclass
class Frame(object):
	timeStep: int
	stateFrames: list

@dataclass
class StateFrame(object):
	state: tuple
	probability: float

@dataclass
class CVASResult(object):
	dim: int
	frames: list

	def __init__(self, dim, frames=[].copy()) -> None:
		self.dim = dim
		self.frames = frames

# def __init__(self, filename: str) -> None:
# with open(filename, 'r') as f:
# data = json.load(f)
# validate_data(data, result_schema)

	def validate(self):
		validate_data(self.to_dict(), result_schema)

class Entry:
	def __init__(self, col: int, val: float):
		self.col: int = col
		self.val: float = val

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

	def add_next_value(self, row: int, col: int, val: float):
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
					assert (len(self.from_list[row]) == 1)
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

	def add_exit_rate(self, idx: int, rate: float):
		while len(self.exit_rates) <= idx:
			self.exit_rates.append(None)
		self.exit_rates[idx] = rate

	def assert_all_entries_correct(self):
		# assert(len(self.exit_rates) == len(self.from_list))
		for i in range(len(self.from_list)):
			# print(f"{i}: {self.exit_rates[i]}, {[str(entry) for entry in self.from_list[i]]}")
			if len(self.from_list[i]) == 0:
				self.exit_rates[i] = None
				assert (self.exit_rates[i] is None)
				continue
			max_entry = max(self.from_list[i])
			max_rate = max_entry.val
			if not self.exit_rates[i] >= max_rate:
				print(f"Error: {self.exit_rates[i]} < {max_rate} (state index {i})")
			assert (self.exit_rates[i] >= max_rate or math.isclose(max_rate, self.exit_rates[i]))

	def export_transitions(self, outfile: str):
		with open(outfile, 'w') as of:
			for i in range(len(self.from_list)):
				row = self.from_list[i]
				for entry in row:
					of.write(f"{i},{entry}\n")

class Explorer(object):
	def __init__(self, filename: str, dim_max=50, use_abs: bool = True) -> None:
		with open(filename, 'r') as f:
			self.cvas = CVAS(json.loads(f.read()))
		self.__currentState = self.cvas.initialState
		self.__exploredStates = set()
		self.__stateToIndex = dict()
		self.__indexToState = dict()  # could use an array. am lazy
		self.__dim_max = dim_max
		self.__matrixBuilder = None
		self.__use_abs = use_abs

	def build(self):
		# build with bound
		ABSORBING_STATE = tuple([-1 for _ in self.cvas.initialState])
		self.__exploredStates = dict()
		if self.__use_abs:
			self.__exploredStates[ABSORBING_STATE] = 0
		self.__matrixBuilder = RandomAccessSparseMatrixBuilder()

		if self.__use_abs:
			queue = [(self.cvas.initialState, 1)]
			self.__stateToIndex[self.cvas.initialState] = 1
			self.__stateToIndex[ABSORBING_STATE] = 0
			self.__indexToState[1] = self.cvas.initialState
			self.__indexToState[0] = ABSORBING_STATE
			nextIdx = 2
		else:
			queue = [(self.cvas.initialState, 0)]
			nextIdx = 1
			self.__indexToState[0] = self.cvas.initialState
		while len(queue) > 0:
			# dequeue the first state
			s, idx = queue.pop()
			print(f"\rExploring state with ID {idx}...", end="")
			assert (np.all([d >= 0 for d in s]))
			# print(f"Exploring state {s} (idx: {idx})")
			self.__exploredStates[s] = nextIdx
			exitRate = 0.0
			successors = self.__successors(s)
			if len(successors) == 0:
				# Create a self loop
				self.__matrixBuilder.add_next_value(idx, idx, 1.0)
				self.__matrixBuilder.add_exit_rate(idx, 1.0)
				continue
			# Enqueue successors
			for sNxt, rate in successors:
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
		print("finished.")
		self.__matrixBuilder.assert_all_entries_correct()
		return self.__matrixBuilder.build()

	def __successors(self, state: tuple) -> list:
		successors = []
		for update in self.cvas.stateUpdates:
			if update.needed is not None and not np.all([state[i] <= update.needed[i] for i in range(len(state))]):
				continue
			if update.ignore and self.__use_abs:
				# redirect to the absorbing state
				successors.append((tuple([-1 for _ in state]), self.__rate(state, update.vector, update.rate)))
				continue
			nextCandidate = tuple(np.add(state, update.vector))
			if not np.all([d >= 0 and d <= self.__dim_max for d in nextCandidate]):
				continue
			successors.append((nextCandidate, self.__rate(state, update.vector, update.rate)))
		return successors

	def __rate(self, state: tuple, update: tuple, rConst: float):
		assert (len(state) == len(update))
		return rConst * np.prod([state[i] ** max(-update[i], 0) for i in range(len(state))])

	def stateCount(self) -> int:
		if self.__matrixBuilder is None:
			raise Exception("Must build model first!")
		return self.__matrixBuilder.size()

	def createLabels(self):
		assert (self.__matrixBuilder is not None)
		labeling = StateLabeling(self.__matrixBuilder.size())
		# Add the initial state
		labeling.add_label("init")
		if self.__use_abs:
			labeling.add_label_to_state("init", 1)
		else:
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

	def state(self, idx: int):
		return self.__indexToState[idx]

	def export_transitions(self, filename: str):
		assert self.__matrixBuilder is not None
		self.__matrixBuilder.export_transitions(filename)

class PrismExplorer(Explorer):
	def __init__(self, prism_filename: str, csl_prop: str) -> None:
		# No need to call super.
		self.__program = stormpy.parse_prism_program(prism_filename)
		self.__properties = stormpy.parse_properties_for_prism_program(csl_prop, self.__program, None)
		self.__matrixBuilder = None
		self.__init_ids = []

	def build(self):
		self.__init_ids = []
		prism_model = stormpy.build_model(self.__program, self.__properties)
		# We do not need to do a DFS or anything similar because we already have all of the states to iterate over
		self.__matrixBuilder = RandomAccessSparseMatrixBuilder()
		for state in prism_model.states:
			if state.id in prism_model.initial_states:
				self.__init_ids.append(state.id)
			# We only work on deterministic models. No mdps
			assert len(state.actions) == 1
			row = state.id
			for transition in state.actions[0]:
				col = transition.column
				rate = transition.value()
				self.__matrixBuilder.add_next_value(row, col, rate)
		self.__prism_model = prism_model
		return self.__matrixBuilder.build()

	def createLabels(self):
		assert (self.__matrixBuilder is not None)
		labeling = StateLabeling(self.__matrixBuilder.size())
		# Add the initial state
		labeling.add_label("init")
		for init_id in self.__init_ids:
			labeling.add_label_to_state("init", init_id)
		for state in self.__prism_model:
			label = f"state_{state.id}"
			labeling.add_label(label)
			labeling.add_label_to_state(label, state.id)
		return labeling

	def state(self, idx: int):
		assert self.__prism_model is not None
		return self.__prism_model.states[idx].valuation
