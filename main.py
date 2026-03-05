from model import *

import sys
from dataclasses import asdict
import json

import stormpy

# F*** my life
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NpEncoder, self).default(obj)


if __name__ == "__main__":
    assert (len(sys.argv) >= 2)
    export_transitions = "--export_trans" in sys.argv
    ignore_abs = "--ignore_abs" in sys.argv
    use_prism = "--prism" in sys.argv
    filename = sys.argv[1]
    model = None
    stCount = None
    e = None
    if not use_prism:
        print("Using custom JSON format")
        e = Explorer(filename, use_abs=not ignore_abs)
    else:
        print(f"Loading prism model {filename}")
        csl = None
        for arg in sys.argv:
            if arg.startswith("--csl="):
                csl = arg.replace("--csl=", "")
        if csl is None:
            raise Exception("Must provide a CSL property if using PRISM mode.")
        e = PrismExplorer(filename, csl)
    model = e.createModel()
    stCount = e.stateCount()

    assert model is not None and stCount is not None and e is not None

    cvr = CVASResult(3)

    env = stormpy.Environment()
    env.solver_environment.native_solver_environment.precision = stormpy.Rational(1e-100)

    timerange = 2
    for timeStep in range(1, timerange+1):
        # timeStep = timeStep / 2
        f = Frame(timeStep, [])
        print(f"Processing Time Step {timeStep} / {timerange}")
        for state in range(stCount):
            #print(f"Checking state {state} for time step {timeStep}...", end="")
            propStr = f"P=? [ F={timeStep} \"state_{state}\" ]"
            prop = stormpy.parse_properties(propStr)[0]
            # The reason we only check initial states is because this is from the
            # STARTING state, whereas we want to see what we're reaching from that
            # state at any given time step
            result = stormpy.check_model_sparse(model, prop, only_initial_states=True)
            # TODO: Add this to a CVAS result
            p = result.at(0)
            # print(f"P = {p}")
            sf = StateFrame(e.state(state), p)
            f.stateFrames.append(sf)
        cvr.frames.append(f)

    # write this shit to a file
    cvr_json = json.dumps(asdict(cvr), cls=NpEncoder, indent='\t')
    with open("model_prob2.json", 'w') as f:
        f.write(cvr_json)

    if export_transitions:
        e.export_transitions("model_transitions2.json")
