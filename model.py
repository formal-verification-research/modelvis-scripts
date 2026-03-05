import json
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from jsonschema import validate

import numpy as np
import math

# Import Stormpy (Required for PRISM mode)
from stormpy import SparseMatrixBuilder, StateLabeling, SparseModelComponents
import stormpy
import stormpy.storage

# --- Load Schemas ---
try:
    with open("./cvas.schema.json", "r") as f:
        cvas_schema = json.load(f)
    with open("./result.schema.json", "r") as f:
        result_schema = json.load(f)
except FileNotFoundError:
    print("Warning: Schema files not found. Validation will be skipped.")
    cvas_schema = {}
    result_schema = {}


def validate_data(data, schema):
    if not schema: return
    try:
        validate(instance=data, schema=schema)
    except Exception:
        raise ValueError("Invalid CVAS Data against Schema!")


# --- Data Classes ---

@dataclass
class StateUpdate(object):
    rate: float
    vector: tuple
    needed: list
    ignore: bool
    rxn_id: str

    def __init__(self, rate, vector, needed=None, ignore=False, reactions=None) -> None:
        self.rate = rate
        self.vector = tuple(vector)
        self.needed = needed
        self.ignore = ignore
        self.rxn_id = reactions.get("id") if isinstance(reactions, dict) else None


@dataclass_json
@dataclass
class CVAS(object):
    '''
    A continuous time Vector Addition System.
    '''
    dim: int
    initialState: tuple
    stateUpdates: list

    def __init__(self, data: dict) -> None:
        self.dim = data["dim"]
        self.initialState = tuple(data["initialState"])
        self.stateUpdates = [
            StateUpdate(d["rate"], d["vector"],
                        None if "needed" not in d else d["needed"],
                        False if "ignore" not in d else bool(d["ignore"]),
                        d.get("reactions")) for d in data["stateUpdates"]
        ]

    def __post_init__(self):
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

    def __init__(self, dim, frames=None) -> None:
        self.dim = dim
        self.frames = frames if frames is not None else []

    def validate(self):
        validate_data(self.to_dict(), result_schema)


# --- Matrix Builder Helpers ---

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
        matrix_builder = SparseMatrixBuilder()
        for row in range(len(self.from_list)):
            self.from_list[row].sort()
            if len(self.from_list[row]) == 0:
                matrix_builder.add_next_value(row, row, 1.0)
            for entry in self.from_list[row]:
                col = entry.col
                val = entry.val
                if row == col:
                    matrix_builder.add_next_value(row, col, 1.0)
                    break
                matrix_builder.add_next_value(row, col, val)
        return matrix_builder

    def build(self):
        matrix_builder = self.to_smb()
        return matrix_builder.build()

    def size(self):
        return len(self.from_list)

    def add_exit_rate(self, idx: int, rate: float):
        while len(self.exit_rates) <= idx:
            self.exit_rates.append(None)
        self.exit_rates[idx] = rate

    def assert_all_entries_correct(self):
        for i in range(len(self.from_list)):
            if len(self.from_list[i]) == 0:
                self.exit_rates[i] = None
                continue
            max_entry = max(self.from_list[i])
            max_rate = max_entry.val
            assert (self.exit_rates[i] >= max_rate or math.isclose(max_rate, self.exit_rates[i]))

    def export_transitions(self, outfile: str):
        # Default export (CSV style) used by base Explorer
        with open(outfile, 'w') as of:
            for i in range(len(self.from_list)):
                row = self.from_list[i]
                for entry in row:
                    of.write(f"{i},{entry}\n")


# --- Base Explorer (for Custom JSON Models) ---

class Explorer(object):
    def __init__(self, filename: str, dim_max=20, use_abs: bool = True) -> None:
        with open(filename, 'r') as f:
            self.cvas = CVAS(json.loads(f.read()))
        self.__currentState = self.cvas.initialState
        self.__exploredStates = {}
        self.__stateToIndex = dict()
        self.__indexToState = dict()
        self.__dim_max = dim_max
        self.__matrixBuilder = None
        self.__use_abs = use_abs
        self.__edge_reaction = {}  # Stores reaction IDs for transitions

    def build(self):
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
            s, idx = queue.pop()
            print(f"\rExploring state with ID {idx}...", end="")
            self.__exploredStates[s] = nextIdx
            exitRate = 0.0
            successors = self.__successors(s)

            if len(successors) == 0:
                self.__matrixBuilder.add_next_value(idx, idx, 1.0)
                self.__matrixBuilder.add_exit_rate(idx, 1.0)
                continue

            for sNxt, rate, upd in successors:
                exitRate += rate
                succIdx = None
                if sNxt in self.__stateToIndex:
                    succIdx = self.__stateToIndex[sNxt]
                else:
                    succIdx = nextIdx
                    nextIdx += 1
                    self.__stateToIndex[sNxt] = succIdx
                    self.__indexToState[succIdx] = sNxt

                self.__matrixBuilder.add_next_value(idx, succIdx, rate)

                # Store reaction label (e.g., "R1")
                if (idx, succIdx) not in self.__edge_reaction:
                    self.__edge_reaction[(idx, succIdx)] = upd.rxn_id

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
                successors.append((tuple([-1 for _ in state]), self.__rate(state, update.vector, update.rate), update))
                continue
            nextCandidate = tuple(np.add(state, update.vector))
            if not np.all([d >= 0 and d <= self.__dim_max for d in nextCandidate]):
                continue
            successors.append((nextCandidate, self.__rate(state, update.vector, update.rate), update))
        return successors

    def __rate(self, state: tuple, update: tuple, rConst: float):
        return rConst * np.prod([state[i] ** max(-update[i], 0) for i in range(len(state))])

    def stateCount(self) -> int:
        if self.__matrixBuilder is None:
            raise Exception("Must build model first!")
        return self.__matrixBuilder.size()

    def createLabels(self):
        assert (self.__matrixBuilder is not None)
        labeling = StateLabeling(self.__matrixBuilder.size())
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
        components = SparseModelComponents(matrix, labels, {}, rate_transitions=True)
        model = stormpy.storage.SparseCtmc(components)
        return model

    def state(self, idx: int):
        return self.__indexToState[idx]

    def export_transitions(self, filename: str):
        if self.__matrixBuilder is None:
            raise Exception("Call build() before exporting.")

        # Format states as simple lists of integers
        n_states = max(self.__indexToState.keys()) + 1 if self.__indexToState else 0
        states = [[int(x) for x in self.__indexToState[i]] for i in range(n_states)]

        transitions = []
        for src, row in enumerate(self.__matrixBuilder.from_list):
            for entry in row:
                tgt = int(entry.col)
                transitions.append({
                    "source": int(src),
                    "target": tgt,
                    "rate": float(entry.val),
                    "reaction": self.__edge_reaction.get((int(src), tgt), "unknown")
                })

        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"states": states, "transitions": transitions}, f, indent=4)


# --- PRISM Explorer (Enhanced for Label Inference) ---

class PrismExplorer(Explorer):
    def __init__(self, prism_filename: str, csl_prop: str) -> None:
        self.__program = stormpy.parse_prism_program(prism_filename, prism_compat=True)
        self.__properties = stormpy.parse_properties_for_prism_program(csl_prop, self.__program, None)
        self.__matrixBuilder = None
        self.__init_ids = []
        self.__valuations = []
        self.__raw_valuations = []
        self.__valuation_labels = []

    def _parse_valuation(self, vals):
        """Parses state.valuations string/object into a dictionary."""
        if isinstance(vals, str):
            if vals.startswith("[") and vals.endswith("]"):
                inner = vals[1:-1].strip()
                if not inner: return {}
                parts = inner.split("&")
                d = {}
                for p in parts:
                    if "=" in p:
                        k, v = p.split("=")
                        try:
                            d[k.strip()] = int(v.strip())
                        except ValueError:
                            try:
                                d[k.strip()] = float(v.strip())
                            except ValueError:
                                d[k.strip()] = v.strip()
                return d
            try:
                return json.loads(vals)
            except:
                return {}
        try:
            return {k: int(v) for k, v in vals.items()}
        except:
            return self._parse_valuation(str(vals))

    def build(self):
        options = stormpy.BuilderOptions()
        options.set_build_state_valuations()
        self.__init_ids = []
        self.__valuations = []
        self.__raw_valuations = []
        self.__valuation_labels = []
        prism_model = stormpy.build_sparse_model_with_options(self.__program, options)
        # We do not need to do a DFS or anything similar because we already have all of the states to iterate over
        self.__matrixBuilder = RandomAccessSparseMatrixBuilder()
        for state in prism_model.states:
            # Store String Label (for visualization/debugging)
            self.__valuation_labels.append(str(state.valuations))

            # Store Python Dict (for JSON export)
            val_dict = self._parse_valuation(state.valuations)
            self.__valuations.append(val_dict)

            # Store Object (for math evaluation)
            self.__raw_valuations.append(state.valuations)

            if state.id in prism_model.initial_states:
                self.__init_ids.append(state.id)

            # We only work on deterministic models. No mdps
            assert len(state.actions) <= 1
            row = state.id
            for action in state.actions:
                for transition in action.transitions:
                    col = transition.column
                    rate = transition.value()
                    self.__matrixBuilder.add_next_value(row, col, rate)
        self.__prism_model = prism_model
        return self.__matrixBuilder.build()

    def createLabels(self):
        assert (self.__matrixBuilder is not None)
        labeling = StateLabeling(self.__matrixBuilder.size())
        labeling.add_label("init")
        for init_id in self.__init_ids:
            labeling.add_label_to_state("init", init_id)
        for state in self.__prism_model.states:
            label = f"state_{state.id}"
            labeling.add_label(label)
            labeling.add_label_to_state(label, state.id)
        return labeling

    def state(self, idx: int):
        return self.__valuation_labels[idx]

    def _find_label(self, source_raw, source_dict, target_dict):
        """
        Scans the PRISM program to find which command explains the transition.
        Uses a robust mix of string-matching (fast) and C++ evaluation (accurate).
        """
        for module in self.__program.modules:
            for command in module.commands:
                # Optional: Check guard conditions if simple enough
                try:
                    if not command.guard.evaluate_as_bool(source_raw):
                        continue
                except:
                    pass

                for update in command.updates:
                    is_match = True

                    # 1. Check assignments
                    for assignment in update.assignments:
                        var_name = assignment.variable.name
                        expression = assignment.expression
                        target_v = float(target_dict.get(var_name, 0))

                        # Optimization: Try string match for simple constants first
                        try:
                            if math.isclose(float(str(expression)), target_v, abs_tol=1e-5):
                                continue
                        except:
                            pass

                            # Fallback: Full C++ evaluation
                        try:
                            new_val = expression.evaluate_as_double(source_raw)
                            if not math.isclose(new_val, target_v, abs_tol=1e-5):
                                is_match = False
                                break
                        except Exception:
                            is_match = False
                            break

                    # 2. Check unassigned variables to ensure they stayed the same
                    if is_match:
                        assigned_vars = set(a.variable.name for a in update.assignments)
                        for var_name, val in source_dict.items():
                            if var_name not in assigned_vars and var_name in target_dict:
                                if not math.isclose(float(target_dict[var_name]), float(val), abs_tol=1e-5):
                                    is_match = False
                                    break

                    if is_match:
                        return command.action_name if command.action_name else "unlabeled"

        return "unknown"

    def export_transitions(self, filename: str):
        if self.__matrixBuilder is None:
            raise Exception("Call build() before exporting.")

        states = [self.__valuations[i] for i in range(self.stateCount())]
        transitions = []

        for src_idx in range(len(self.__matrixBuilder.from_list)):
            row = self.__matrixBuilder.from_list[src_idx]
            source_raw = self.__raw_valuations[src_idx]
            source_dict = self.__valuations[src_idx]

            for entry in row:
                tgt_idx = entry.col
                target_dict = self.__valuations[tgt_idx]

                label = self._find_label(source_raw, source_dict, target_dict)

                transitions.append({
                    "source": int(src_idx),
                    "target": int(tgt_idx),
                    "rate": float(entry.val),
                    "reaction": label
                })

        with open(filename, "w", encoding="utf-8") as f:
            json.dump({"states": states, "transitions": transitions}, f, indent=4)

    def stateCount(self) -> int:
        return self.__matrixBuilder.size()