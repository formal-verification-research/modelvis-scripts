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
	assert(len(sys.argv) == 2)
	filename = sys.argv[1]
	e = Explorer(filename)
	# matrix = e.build()
	model = e.createModel()
	stCount = e.stateCount()

	cvr = CVASResult(3)

	for timeStep in range(1, 11):
		f = Frame(timeStep, [])
		for state in range(stCount):
			print(f"Checking state {state} for time step {timeStep}")
			propStr = f"P=? [ F={timeStep} \"state_{state}\" ]"
			prop = stormpy.parse_properties(propStr)[0]
			# The reason we only check initial states is because this is from the
			# STARTING state, whereas we want to see what we're reaching from that
			# state at any given time step
			result = stormpy.check_model_sparse(model, prop, only_initial_states=True)
			# TODO: Add this to a CVAS result
			p = result.at(0)
			sf = StateFrame(e.state(state), p)
			f.stateFrames.append(sf)
		cvr.frames.append(f)

	# write this shit to a file
	cvr_json = json.dumps(asdict(cvr), cls=NpEncoder, indent='\t')
	with open("output.json", 'w') as f:
		f.write(cvr_json)
