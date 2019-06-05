import argparse, os, time, pickle, json, copy, ast, sys
import numpy as np
from MahlerNet import MahlerNet
from DataProcessor import DataProcessor

parser = argparse.ArgumentParser(description='Train and / or generate music with MahlerNet. The concept of sequence base, or unit, is used throughout and refers to the size of the unit used to model music, most often bar by bar.')
parser.add_argument('mode', action = "store", type = str, choices = ['train', 'generate'], help = "start a training session with MahlerNet or generate music from a trained model")
parser.add_argument('name', action = "store", type = str, metavar='NAME', help = "unique name for this run to store data under in subdirectory 'runs'")
parser.add_argument('-r', '--root', action = "store", nargs = "?", default = "", type = str, metavar='ROOT_DIR', help = 'overall root directory to use for fetching and saving data')
parser.add_argument('-c', '--config', action = "store", nargs = "?", default = None, type = str, metavar = 'CONFIG_FILE', help = 'config file residing in \"ROOT_DIR\"')

parser.add_argument('-t', '--type', action = "store", nargs = "*", choices = ['recon', 'n_recon', 'pred', 'intpol', 'n_pred'], default = None, type = str, metavar='GEN_TYPE', help = 'the desired type(s) of generation. If given in train mode, "intpol" or "n_pred" may not be chosen and generation takes place as an end of epoch (eoe) function. If given in generation mode, all types are allowed. Must be accompanied by "file" if the units argument contains references to specific bars. Must be accompanied with the "units" argument')
parser.add_argument('-u', '--units', action = "store", nargs = "*", default = None, type = str, metavar='ROOT_DIR', help = 'Sequence base input positions in FILE to use for generation. Should be a number followed by "z" and / or "c" to indicate whether to use input or context or both. Use "-" to generate without any cintextual input at all')
parser.add_argument('-f', '--file', action = "store", nargs = "?", default = None, type = str, metavar='FILE_NAME', help = 'overall root directory to use for fetching and saving data')

parser.add_argument('-m', '--model', action = "store", nargs = "?", default = None, help = 'name of the model residing in \"ROOT_DIR/saved_models\" to load when generating')

parser.add_argument('-s', '--samples', action = "store", nargs = "?", default = 1, type = int, metavar='SAMPLES', help = 'the number of samples to generate for each setup')
parser.add_argument('-l', '--length', action = "store", nargs = "?", default = 1, type = int, metavar='LENGTH', help = 'the length of the samples to generate from type "pred", "intpol" and "n_pred"')
parser.add_argument('-M', '--meter', action = "store", nargs = "?", default = 4, type = int, metavar='METER', help = 'the meter of the time signature to enforce when generating music')
parser.add_argument('-T', '--use_triplets', action = "store_true", default = False, help = 'what default sub division to use for offset and duration when these are not modelled, can be either triplet or duplet (default) eight notes')

parser.add_argument('-U', '--no_ctx', action = "store_true", default = False, help = "whether to use context, if applicable, or not when interpolating using type 'intpol' or reconstructing using 'n_recon'")
parser.add_argument('-C', '--use_start_ctx', action = "store_true", default = False, help = 'whether to use the start context (start token set to 1) instead of an empty context (default, all zeros) whenever not context is supplied')

parser.add_argument('-S', '--steps', action = "store", nargs = "?", default = 10, type = int, metavar='INTERPOLATION_STEPS', help = 'the number of steps to use for interpolation between latent states')	
parser.add_argument('-x', '--max_limit', action = "store", nargs = "?", default = None, type = int, metavar='MAX_LIMIT_TRAINING_SEQUENCE', help = 'the maximum number of steps, if any that a training sequence may contain to participate in training')	
parser.add_argument('-F', '--use_teacher_forcing', action = "store_true", default = False, help = "whether to use teacher forcing or not when reconstructing")
parser.add_argument('-e', '--use_slerp', action = "store_true", default = False, help = "whether to use slerp for interpolations or do linear interpolations")

parser.add_argument('-o', '--cont', action = "store_true", default = False, help = "whether to continue a previously started training session, otherwise, a new unique name must be used for training")

def batch_generator(dp, batch_size, is_validation_set, data_dir, use_randomness, max_limit = None, debugging = -1):
	# gen yields tuples of size 6 with (ctxs, ctx_lengths, inps, inp_lengths, sample_nums, file_names)
	# gen yields tuples with shapes (batch_size, num_steps, num_features) fors ctxs and inps and (batch_size) for the rest
	gen = dp.sequences_to_training_data_generator(batch_size, is_validation_set, root_dir = data_dir, random_ordering = use_randomness, max_limit = max_limit)
	skipped = 0
	for no, tup in enumerate(gen): # [1] is the actual inputs (as opposed to context)
		if skipped < debugging:
			skipped += 1
		else:
			batch = {}
			ctx = tup[0][:, :, : dp.mp.act_notes_a] # context without the active instruments and pitches part (lies last in the data representation)
			batch["ctx_lengths"] = tup[1]
			batch["inp_lengths"] = tup[3]
			y = tup[2]
			x = np.concatenate((np.zeros((y.shape[0], 1, y.shape[2])), y[:, : -1, :]), axis = 1)
			
			# extract individual parts
			batch["batch_sz"] = y.shape[0]
			batch["x_o"] = x[:, :, dp.mp.time_a: (dp.mp.time_a + dp.mp.num_durations)]
			batch["x_b"] = x[:, :, dp.mp.beats_a: (dp.mp.beats_a + dp.mp.NUM_BEATS)]
			batch["x_p"] = x[:, :, dp.mp.inp_a: (dp.mp.inp_a + dp.mp.NUM_PITCHES)]
			batch["x_d"] = x[:, :, dp.mp.dur_a: (dp.mp.dur_a + dp.mp.num_durations)]
			batch["x_i"] = x[:, :, dp.mp.instr_a: (dp.mp.instr_a + dp.mp.NUM_INSTRUMENTS)]
			batch["x_ap"] = x[:, :, dp.mp.act_notes_a: (dp.mp.act_notes_a + dp.mp.NUM_PITCHES)]
			batch["x_ai"] = x[:, :, dp.mp.act_inst_a: (dp.mp.act_inst_a + dp.mp.NUM_INSTRUMENTS)]
			batch["y_o"] = y[:, :, dp.mp.time_a: (dp.mp.time_a + dp.mp.num_durations)]
			batch["y_b"] = y[:, :, dp.mp.beats_a: (dp.mp.beats_a + dp.mp.NUM_BEATS)]
			batch["y_p"] = y[:, :, dp.mp.inp_a: (dp.mp.inp_a + dp.mp.NUM_PITCHES)]
			batch["y_d"] = y[:, :, dp.mp.dur_a: (dp.mp.dur_a + dp.mp.num_durations)]
			batch["y_i"] = y[:, :, dp.mp.instr_a: (dp.mp.instr_a + dp.mp.NUM_INSTRUMENTS)]
			batch["ctx_s"] = ctx[:, :, dp.mp.final_tokens_a: (dp.mp.final_tokens_a + dp.mp.NUM_SPECIAL_TOKENS)]
			batch["ctx_o"] = ctx[:, :, dp.mp.time_a: (dp.mp.time_a + dp.mp.num_durations)]
			batch["ctx_b"] = ctx[:, :, dp.mp.beats_a: (dp.mp.beats_a + dp.mp.NUM_BEATS)]
			batch["ctx_p"] = ctx[:, :, dp.mp.inp_a: (dp.mp.inp_a + dp.mp.NUM_PITCHES)]
			batch["ctx_d"] = ctx[:, :, dp.mp.dur_a: (dp.mp.dur_a + dp.mp.num_durations)]
			batch["ctx_i"] = ctx[:, :, dp.mp.instr_a: (dp.mp.instr_a + dp.mp.NUM_INSTRUMENTS)]
			
			yield batch
		
def parse_position(pos):
	s = pos
	c = False
	z = False
	while ord(pos[-1]) > ord('9') or ord(pos[-1]) < ord('0'):
		if pos[-1] == "z":
			z = True
		elif pos[-1] == "c":
			c = True
		assert(len(pos) > 0), "[RUN]: ERROR - malformed position indicator in file, must be a number followed by 'c' and / or 'z': " + str(pos)
		pos = pos[: -1]
	assert(z or c), "[RUN]: ERROR - malformed position indicator in file, must be a number followed 'c' and / or 'z' but at least one: " + str(pos)
	return int(pos), c, z
	
def parse_units(units):
	input = []
	empty = False
	input = []
	ctxs = 0
	zs = 0
	for unit in units:
		if unit == "-":
			empty = True
		else:
			input += [parse_position(unit)]	
			ctxs = (ctxs + 1) if input[-1][1] else ctxs
			zs = (zs + 1) if input[-1][2] else zs
	return (empty, input, ctxs, zs)
	
def fetch_units(input, generator):
	i = 0
	for tup_id in range(len(input)):
		while i <= input[tup_id][0]:
			generated = next(generator, None)
			assert(generated is not None), "[RUN]: ERROR - given positions in the file are beyond the end of the file"
			ctx, inp = generated
			i += 1
		input[tup_id] = (input[tup_id][0], np.copy(ctx) if input[tup_id][1] else None, np.copy(inp) if input[tup_id][2] else None)
	return input		

def dict_values_to_str(d):
	return {k: dict_values_to_str(v) if isinstance(v, dict) else str(v) for k, v in d.items()}

def main():
	params = parser.parse_args()
	params.start_ctx = "START" if params.use_start_ctx else None
	params.type = [] if params.type is None else params.type
	params.with_ctx = not params.no_ctx	
	mn_params = {}
	orig_params_file = None
	save_dir = os.path.join(params.root, "runs", params.name)
	assert(len(params.name) > 0), "[RUN]: ERROR - please submit a legitimate unique run id for the process run to start"
	if params.mode == "train":
		train_mission = True
		gen_mission = False
		params_file = None
		if params.cont:
			assert(os.path.exists(save_dir)), "[RUN]: ERROR - can't continue training from non-existing run folder '" + str(save_dir) + "'"
			assert(params.model is not None), "[RUN]: ERROR - must supply a saved model to continue training"
			if os.path.exists(os.path.join(save_dir, "params.txt")):
				params_file = os.path.join(save_dir, "params.txt")
		else:
			assert(not os.path.exists(save_dir)), "[RUN]: ERROR - the name specified for this run already exists, please specify a new unique name or delete the folder named \'" + str(params.name) + "' in subdir \'runs\'"
			params_file = params.config
		if params_file is not None:
			with open(params_file, mode = "r") as pf:
				if params.cont:
					mn_params = json.loads(pf.read())
				else:
					mn_params = ast.literal_eval(pf.read())
				orig_params_file = copy.deepcopy(mn_params)
		os.makedirs(os.path.join(params.root, "runs"), exist_ok = True)		
	elif params.mode in ["generate"]:
		train_mission = False
		gen_mission = True
		assert(os.path.exists(save_dir)), "[RUN]: ERROR - specified path '" + str(save_dir) + "' does not name a run folder in '" + str(os.path.join(params.root, "runs")) + "'"
		params_path = os.path.join(save_dir, "params.txt")
		if os.path.exists(params_path):
			with open(params_path, mode = "r") as pf:
				mn_params = json.loads(pf.read())

	dp = DataProcessor()
	
	if train_mission:
		if len(mn_params) > 0:
			dp.setup_training(root_dir = params.root, validation_set_ratio = mn_params["validation_set_ratio"], random_ordering = mn_params["use_randomness"])
			assert(mn_params["generation"]["sequence_base"] == dp.sequence_base), "[RUN]: different sequence base in prepared training directory and in input parameters: \"" + str(mn_params["generation"]["sequence_base"]) + "\" and \"" + str(dp.sequence_base) + "\""
		else:
			dp.setup_training(root_dir = params.root)
			mn_params["generation"]["sequence_base"] = dp.sequence_base
			
	new_params = {
		"num_durations" : dp.mp.num_durations,
		"num_pitches" : dp.mp.NUM_PITCHES,
		"num_instruments" : dp.mp.NUM_INSTRUMENTS,
		"num_beats" : dp.mp.NUM_BEATS,
		"num_special_tokens": dp.mp.NUM_SPECIAL_TOKENS,
		"modelling_properties": {
			"offset": {
				"indices": (dp.mp.time_a, dp.mp.time_a + dp.mp.num_durations)
			},
			"beats": {
				"indices": (dp.mp.beats_a, dp.mp.beats_a + dp.mp.NUM_BEATS)
			},
			"duration": {
				"indices": (dp.mp.dur_a, dp.mp.dur_a + dp.mp.num_durations)
			},
			"pitch": {
				"indices": (dp.mp.inp_a, dp.mp.inp_a + dp.mp.NUM_PITCHES)
			},
			"instrument": {
				"indices": (dp.mp.instr_a, dp.mp.instr_a + dp.mp.NUM_INSTRUMENTS)
			},
			"active_pitches": {
				"indices": (dp.mp.act_notes_a, dp.mp.act_notes_a + dp.mp.NUM_PITCHES)
			},
			"active_instruments": {
				"indices": (dp.mp.act_inst_a, dp.mp.act_inst_a + dp.mp.NUM_INSTRUMENTS)
			},
			"special_tokens": {
				"indices": (dp.mp.final_tokens_a, dp.mp.final_tokens_a + dp.mp.NUM_SPECIAL_TOKENS)
			}
		},
		"generation": {
			"default_durations": np.reshape(dp.mp.default_durations, [1, 1, dp.mp.default_durations.shape[0]]),
			"default_bar": dp.mp.default_bar_unit_duration,
			"default_beats": np.reshape(dp.mp.default_beats_vector, [1, 1, dp.mp.default_beats_vector.shape[0]]),
			"default_duration_set": dp.mp.default_duration_set,
			"default_duration_sets": [0, 0, dp.mp.default_duple_duration_set, dp.mp.default_triple_duration_set, dp.mp.default_common_duration_set],
			"default_duplet_duration": "d8",
			"default_triplet_duration": "t8"
		}
	}
	if train_mission:
		new_params["generation"]["ctx_length"] = dp.ctx_length
		new_params["generation"]["inp_length"] = dp.inp_length
	def update_dict(orig, add):
		for k in add:
			if isinstance(add[k], dict):
				if k not in orig:
					orig[k] = add[k]
				else:
					assert(isinstance(orig[k], dict)), "[RUN]: ERROR - while merging parameters for MahlerNet, dict key \"" + k + "\" should be a dict but is not in config file"
					orig[k] = update_dict(orig[k], add[k])
			else:
				orig[k] = add[k]
		return orig
	mn_params = update_dict(mn_params, new_params)
	mn_params["root_dir"] = params.root
	mn_params["model_name"] = params.name
	mn_params["save_dir"] = save_dir
	mn = MahlerNet(mn_params)
	mn.build()
	input = None
	empty = False
	ctxs = 0
	zs = 0
	if params.units is not None:
		empty, input, ctxs, zs = parse_units(params.units)
		input.sort()
		if params.file is not None:
			generator = dp.data_generator(params.file + ".pickle", mn.params["root_dir"], "data", mn.params["generation"]["sequence_base"], mn.params["generation"]["ctx_length"], mn.params["generation"]["inp_length"])
			input = fetch_units(input, generator)
		else:
			assert(len(input) == 0), "[RUN]: ERROR - units are given for generation but no file was supplied, only '-' is allowed as a position without an input file"
		if empty:
			input = [("empty", None, None)] + input			
	elif params.file is not None:
		generator = dp.data_generator(params.file + ".pickle", mn.params["root_dir"], "data", mn.params["generation"]["sequence_base"], mn.params["generation"]["ctx_length"], mn.params["generation"]["inp_length"])
	
	# check conditions applying to both training and generating before proceeding
	assert("recon" not in params.type or len(input) > 0 and params.file is not None), "[RUN]: ERROR - must supply file and > 0 input units to generate by reconstruction with option 'recon'"
	assert("pred" not in params.type or empty or (input is not None and len(input) > 0 and params.file is not None)), "[RUN]: ERROR - must supply file and > 0 input units or '-' unit to generate by prediction with option 'pred'"
	
	# if units were given, input is now a list of tuples were each tuple is on the form (c, i) where c, i = None if that property should use default (if modelled) and other wise (bool, prop) if prop should be used (if modelled).
	# the bool indicates whether to process the input with latent() or context() first or if it is the processed version given. empty might be set to true indicating the use of the all empty input (None, None) in which case
	# sampling of z and initial ctx are used.
	if train_mission:
		assert("intpol" not in params.type and "n_pred" not in params.type and "n_recon" not in params.type), "[RUN]: ERROR - only 'recon' and 'pred' are available for continuous generation while training"
		print("[RUN]: training with", dp.total, "samples with a maximum of", dp.max_context_length, "steps in context and", dp.max_input_length, "in input")
		if params.model is not None and params.cont:
			mn.load_model(params.model)
		generator_fn = lambda batch: (batch_generator(dp, batch, False, params.root, mn_params["use_randomness"], params.max_limit), batch_generator(dp, batch, True, params.root, mn_params["use_randomness"],  params.max_limit))
		eoe_fns = []
		if "recon" in params.type:
			eoe_fns += [lambda model, epoch, step: reconstruction_test(model, dp, "epoch" + str(epoch) + "_step" + str(step), input, params.samples, params.meter, params.use_triplets, params.start_ctx, params.use_teacher_forcing)]
		if "pred" in params.type:
			eoe_fns += [lambda model, epoch, step: prediction_test(model, dp, "epoch" + str(epoch) + "_step" + str(step), input, params.samples, params.meter, params.length, params.use_triplets, params.start_ctx)]
		os.makedirs(os.path.join(mn.params["save_dir"]), exist_ok = True)
		if not os.path.exists(os.path.join(save_dir, "commands.txt")):
			with open(os.path.join(save_dir, "commands.txt"), 'w'): 
				pass
		with open(os.path.join(save_dir, "commands.txt"), "a") as f:
			f.write(" ".join(sys.argv))				
		with open(os.path.join(mn.params["save_dir"], 'all_params.txt'), 'w') as all_params_file: # save ALL params of the current setup before starting
			all_params_file.write(json.dumps(dict_values_to_str(mn.params), sort_keys = False, indent = 4))
		if orig_params_file is not None:
			with open(os.path.join(mn.params["save_dir"], 'params.txt'), 'w') as params_file: # save input params for reference and future runs of the current setup before starting
				params_file.write(json.dumps(orig_params_file, sort_keys=False, indent = 4))
		(epoch_losses, step_losses, epoch_prec_rec, step_prec_rec, epoch_dist, step_dist) = mn.train(generator_fn, dp.total, dp.sz_training_set, dp.sz_validation_set, init_vars = params.model is None, eoe_fns = eoe_fns)	
		if mn.params["save_stats"]:
			os.makedirs(os.path.join(mn.params["save_dir"], "records"))	
			path = os.path.join(mn.params["save_dir"], "records", "stats")
			file = open(path, mode = "xb")
			pickle.dump((epoch_losses, step_losses, epoch_prec_rec, step_prec_rec, epoch_dist, step_dist), file)
			print("[RUN]: saved training statistics to", path)
			file.close()		
	else:
		assert(params.model is not None), "[RUN]: ERROR - must specify a model name within the 'saved_models' folder of the root directory to generate something"
		assert(params.type is not None),  "[RUN]: ERROR - must specify at least one type of generation to generate from a loaded model"
		with open(os.path.join(save_dir, "commands.txt"), "a") as f:
			f.write(" ".join(sys.argv))			
		mn.load_model(params.model)
		if "recon" in params.type:
			assert(mn.params["model"]["vae"]), "[RUN]: ERROR - must have a model that includes a vae to run reconstruction tests"
			reconstruction_test(mn, dp, mn.params["model_name"], input, params.samples, params.meter, params.use_triplets, params.start_ctx, params.use_teacher_forcing)
		if "n_recon" in params.type:
			assert(mn.params["model"]["vae"]), "[RUN]: ERROR - must have a model that includes a vae to run reconstruction tests"
			if params.file is not None: # use the input file for the generator
				r_gen = dp.data_generator(params.file + ".pickle", mn.params["root_dir"], "data", mn.params["generation"]["sequence_base"], mn.params["generation"]["ctx_length"], mn.params["generation"]["inp_length"])
			else: # use the training generator, must adapt its output however
				_, _, f = dp.setup_dirs(mn.params["root_dir"], "data", "pickle", files = None)
				def generator_converter(filenames):
					for filename in filenames:
						for (ctx, inp) in dp.data_generator(filename, mn.params["root_dir"], "data", mn.params["generation"]["sequence_base"], mn.params["generation"]["ctx_length"], mn.params["generation"]["inp_length"]):
							yield (ctx, inp)					  		  
				r_gen = generator_converter(f)
			n_reconstruction_test(mn, dp, mn.params["model_name"], r_gen, params.with_ctx, params.start_ctx, params.use_teacher_forcing)
		if "pred" in params.type:
			prediction_test(mn, dp, params.file, input, params.samples, params.meter, params.length, params.use_triplets, params.start_ctx)
		if "n_pred" in params.type:
			assert(zs <= 1 and ctxs > 1), "[RUN]: ERROR - specify several context units, each followed by 'c' and at most one unit to use for input, to use the n_pred generation type"
			assert(mn.params["model"]["ctx"] and mn.params["model"]["vae"]), "[RUN]: ERROR - must use a model that uses both a vae and a context to use the \"n_pred\" generation type"
			z_units = [(n, i) for (n, c, i) in input if i is not None]
			ctx_units = [(n, c) for (n, c, i) in input if c is not None]
			n_prediction_test(mn, dp, params.file, ctx_units, z_units[0] if len(z_units) > 0 else None, params.meter, params.length, params.use_triplets)
		if "intpol" in params.type:
			assert(zs == 2 and ctxs <= 1), "[RUN]: ERROR - must supply exactly two input units ('-' not counted) and an optional context to use for interpolating between latent states"
			z_units = [(n, i) for (n, c, i) in input if i is not None]
			ctx_units = [(n, c) for (n, c, i) in input if c is not None]
			interpolation_test(mn, dp, params.file, ctx_units[0] if len(ctx_units) > 0 else params.start_ctx, z_units, params.use_slerp, params.steps, params.meter, params.use_triplets, params.start_ctx)
			
'''
	Measures accuracy between predictions and targets, at the most within the length of the target; superfluous predictions are ignored. In essence, this function returns two numbers indicating the number of times the predictions
	predicted a 1 where the target had a 1, and then the total number of 1's present in the target. This is effectively the precision and recall.
	
	Params
	------
	
	model: MahlerNet
		The model to use
		
	dp: DurationSet
		The duration set to use for indexing into targets and predictions
		
	targets: numpy array
		The target patterns
		
	preds: numpy array
		The predicted pattern, shorter, equal to or longer than the targets.
		
	Returns
	-------
	
	(int, int)
		The number of correctly predicted 1's versus the total number of 1's existing in the targets.
'''
def measure_accuracy(model, dp, targets, preds):
	chances = {"offset": 0, "duration": 0, "pitch": 0, "instrument": 0}
	correct = {"offset": 0, "duration": 0, "pitch": 0, "instrument": 0}
	effective_timesteps = min(targets.shape[0], preds.shape[0])
	for cat in ["offset", "duration", "pitch", "instrument"]:
		if model.params["modelling_properties"][cat]["include"]:
			cat_targets = targets[: , model.params["modelling_properties"][cat]["indices"][0]: model.params["modelling_properties"][cat]["indices"][1]]
			cat_preds = preds[: effective_timesteps, model.params["modelling_properties"][cat]["indices"][0]: model.params["modelling_properties"][cat]["indices"][1]]
			correct[cat] += np.sum(np.max(np.multiply(cat_targets[: effective_timesteps], cat_preds), axis = -1))
			active_timesteps = np.sum(np.max(cat_targets, axis = -1))
			chances[cat] += active_timesteps
	return (correct, chances)

'''
	Convenience function to reconstruct from a given input. Requires at least one concrete input unit to be given in a concrete file. Ignores the '-' unit since testing reconstruction by reconstructing from an unknown source 
	would be insignificant. If the model uses context, the START context is generated if no context is supplied in the input. If the model does not use context, the context is ignored altogether. None inputs are ignored as 
	well. Prints information about the reconstruction accuracy and saves the reconstructed music to the 'generated' folder in the 'root_dir'.
	
	Params
	------
	
	model: MahlerNet
		The model to use
		
	dp: DurationSet
		The duration set to use for conversion from data to midi
		
	save_name: str
		The name to use as part of the name of saved output
		
	inputs: list of (int or str, numpy array or None, numpy array or None)
		A list with tuples where each tuple is on the form (id, ctx, inp) where id is the unit number from the input file, or "empty" in the case of the '-' unit, ctx is the numpy array representing the context, or None if none is
		given, and inp is the numpy array representing the input pattern, or None, if none is given.
		
	samples: int
		The number of samples to generate from each unit to reconstruction
		
	meter: int
		The induced meter of the output which also affects which duration set is used internally while generating. This affects the range of each output unit which is limited to 2, 3 or 4 beats depending on the given meter.
		Should match what is reconstructed if this is known.
	
	triplet: bool
		Whether the default offset and / or duration should be triplet eighth note or duplet eighth note if offset or duration is not modelled by this model.
		
	default_ctx: str or None
		Default value to use for missing contexts, may either be None, which results in an empty context (all 0's) or "START" which results in the START context (start token only set to 1)		
		
'''
def reconstruction_test(model, dp, save_name, inputs, samples = 1, meter = 4, triplet = False, default_ctx = None, use_teacher_forcing = False):
	save_name = "EMPTY" if save_name is None else save_name
	tot_right = {"offset": 0, "duration": 0, "pitch": 0, "instrument": 0}
	tot_possible = {"offset": 0, "duration": 0, "pitch": 0, "instrument": 0}
	for (base_id, c, i) in inputs:
		if i is not None:
			for id in range(samples):
				if use_teacher_forcing:
					data, default_step_unit, perplexity = model.generate_sample(ctx = default_ctx if c is None else (True, c), inp = (True, i), meter = meter, length = 1, triplet = triplet, use_teacher_forcing = use_teacher_forcing)
				else:
					data, default_step_unit = model.generate_sample(ctx = default_ctx if c is None else (True, c), inp = (True, i), meter = meter, length = 1, triplet = triplet, use_teacher_forcing = use_teacher_forcing)
				right, possible = measure_accuracy(model, dp, i, data[-1][0])
				tot_right = {k: tot_right[k] + right[k] for k in tot_right}
				tot_possible = {k: tot_possible[k] + possible[k] for k in tot_possible}
				base_time = data[-1][4] - data[-1][3]
				data = [(data[-1][0], data[-1][1], data[-1][2], base_time, 2 * base_time)]
				music = [(i, 0, data[0][2], 0, base_time)] + data # music has the input and the decoded input
				files = dp.mp.data_to_midi([music], True, default_step_unit)
				os.makedirs(os.path.join(model.params["save_dir"], "generated"), exist_ok = True)
				files[0].save(os.path.join(model.params["save_dir"], "generated", "recon_" + save_name + "_unit" + str(base_id) + "_try" + str(id) + "_" + time.strftime("%Y%m%d_%H.%M.%S", time.localtime()) + ".mid"))
				acc_str = ", ".join(["(" + k[0] + ") " + ('%.2f' % (100 * (right[k] / possible[k]))) + "%" for k in possible if possible[k] > 0])
				print("[RUN]: generated reconstruction sample with " + str(meter) + "-based meter from unit " + str(base_id) + " in input file with accuracies:", acc_str)
		acc_str = ", ".join(["(" + k[0] + ") " + ('%.2f' % (100 * (tot_right[k] / tot_possible[k]))) + "%" for k in tot_possible if tot_possible[k] > 0])	
		if use_teacher_forcing:
			print("[RUN]: teacher forcing used with resulting perplexities:" + "".join([" " + k + ": " + (('%.2f' % (np.exp(-v))) if v != "N/A" else v) for k, v in perplexity.items()]))
		print("[RUN]: generated " + str(samples * len(inputs)) + " reconstruction sample(s) with " + str(meter) + "-based meter with accuracies:", acc_str)

'''
	Convenience function to measure reconstruction accuracy for all samples generated by the input generator. If a file is given as input, the generator generates all the units in the file, otherwise, the training generator
	with training data (not validation data) is used (warning, this might take long). Ignores the units parameter entirely. Ignores the '-' unit since testing reconstruction by reconstructing from un unknown source would 
	be insignificant. If the model uses context, the START context is generated if no context is supplied in the input. If the model does not use context, the context is ignored altogether. None inputs are ignored as well.
	Prints information about the reconstruction accuracy and saves the reconstructed music to the 'generated' folder in the 'root_dir'.
	
	Params
	------
	
	model: MahlerNet
		The model to use
		
	dp: DurationSet
		The duration set to use for conversion from data to midi
		
	save_name: str
		The name to use as part of the name of saved output
		
	generator: generator of (int or str, numpy array or None, numpy array or None)
		A generator with tuples where each tuple is on the form (id, ctx, inp) where id is the unit number from the input file, or "empty" in the case of the '-' unit, ctx is the numpy array representing the context, or None 
		if none is given, and inp is the numpy array representing the input pattern, or None, if none is given.
		
	with_ctx: bool
		A boolean indicating whether we want to use context or not, given that this model uses context.
		
	meter: int
		The induced meter of the output which also affects which duration set is used internally while generating. This affects the range of each output unit which is limited to 2, 3 or 4 beats depending on the given meter.
		Should match what is reconstructed if this is known.
		
	default_ctx: str or None
		Default value to use for missing contexts, may either be None, which results in an empty context (all 0's) or "START" which results in the START context (start token only set to 1)
		
'''
def n_reconstruction_test(model, dp, save_name, generator, with_ctx, default_ctx = None, use_teacher_forcing = False):
	tot_right = {"offset": 0, "duration": 0, "pitch": 0, "instrument": 0}
	tot_possible = {"offset": 0, "duration": 0, "pitch": 0, "instrument": 0}
	for num, (c, i) in enumerate(generator):
		if use_teacher_forcing:
			data, default_step_unit, _ = model.generate_sample(ctx = (True, c) if with_ctx else default_ctx, inp = (True, i), meter = 4, length = 1, triplet = True, use_teacher_forcing = use_teacher_forcing)
		else:
			data, default_step_unit = model.generate_sample(ctx = (True, c) if with_ctx else default_ctx, inp = (True, i), meter = 4, length = 1, triplet = True, use_teacher_forcing = use_teacher_forcing)
		right, possible = measure_accuracy(model, dp, i, data[-1][0])
		tot_right = {k: tot_right[k] + right[k] for k in tot_right}
		tot_possible = {k: tot_possible[k] + possible[k] for k in tot_possible}
		acc_str = ", ".join(["(" + k[0] + ") " + ('%.2f' % (100 * (tot_right[k] / tot_possible[k]))) + "%" for k in tot_possible if tot_possible[k] > 0])
		print("[RUN]: processed sample " + str(num + 1) + " with " + str(right["offset"] + right["duration"] + right["pitch"] + right["instrument"]) + " predictions correct out of " + str(possible["offset"] + possible["duration"] + possible["pitch"] + possible["instrument"]) + ", (total accuracy so far:", acc_str)
	acc_str = ", ".join(["(" + k[0] + ") " + ('%.2f' % (100 * (tot_right[k] / tot_possible[k]))) + "%" for k in tot_possible if tot_possible[k] > 0])	
	print("[RUN]: generated full run with " + str(num + 1) + " reconstruction sample(s) with accuracies", acc_str)
	
'''
	Convenience function to predict from a given context and / or input. Requires at least one unit to be given. Units are given on the form #cz where # indicates the unit number in the ordered list of units constituting the input file,
	and 'c' and or 'z' indicates that the context and / or input for the given unit should be used when predicting. The '-' unit indicates that both, when applicable, both START context and sampled latent dimension should be
	used (equivalent to totally unconditioned sampling). If units other than the '-' unit are given, a file must also be given from which additional units are fetched. If the model uses context, the START context is generated 
	if no context is supplied in the input. If the model does not use context, the context is ignored altogether. None inputs results in the latent dimension being sampled from, otherwise the input is used to create the latent 
	dimension. Prints information about the prediction and saves the predicted music to the 'generated' folder in the 'save_dir'.
	
	Params
	------
	
	model: MahlerNet
		The model to use
		
	dp: DurationSet
		The duration set to use for conversion from data to midi
		
	save_name: str
		The name to use as part of the name of saved output
		
	inputs: list of (int or str, numpy array or None, numpy array or None)
		A list with tuples where each tuple is on the form (id, ctx, inp) where id is the unit number from the input file, or "empty" in the case of the '-' unit, ctx is the numpy array representing the context, or None if none 
		is given, and inp is the numpy array representing the input pattern, or None, if none is given.
		
	samples: int
		The number of samples to generate from each unit to reconstruction
		
	meter: int
		The induced meter of the output which also affects which duration set is used internally while generating. This affects the range of each output unit which is limited to 2, 3 or 4 beats depending on the given meter.
		Should match what is reconstructed if this is known.
	
	length: int
		The number of units to generate in a row in each output sample.
		
	triplet: bool
		Whether the default offset and / or duration should be triplet eighth note or duplet eighth note if offset or duration is not modelled by this model.
		
	default_ctx: str or None
		Default value to use for missing contexts, may either be None, which results in an empty context (all 0's) or "START" which results in the START context (start token only set to 1)
		
'''
def prediction_test(model, dp, save_name, inputs, samples = 1, meter = 4, length = 1, triplet = False, default_ctx = None):
	save_name = "EMPTY" if save_name is None else save_name
	for (base_id, c, i) in inputs:
		for id in range(samples):
			music, default_step_unit = model.generate_sample(ctx = default_ctx if c is None else (True, c), inp = (True, i) if i is not None else None, meter = meter, length = length, triplet = triplet)
			files = dp.mp.data_to_midi([music], True, default_step_unit)
			os.makedirs(os.path.join(model.params["save_dir"], "generated"), exist_ok = True)
			files[0].save(os.path.join(model.params["save_dir"], "generated", "pred_" + save_name + "_unit" + str(base_id) + "_try" + str(id) + "_" + time.strftime("%Y%m%d_%H.%M.%S", time.localtime()) + ".mid"))
		print("[RUN]: generated " + str(samples) + " prediction sample(s) of length " + str(length) + " with " + str(meter) + "-based meter from unit " + str(base_id) + " in input file")
		
'''
	Convenience function to test the impact of different contexts on the same latent dimension content. Input units must contain more than one unit to use for context and an optional unit to use for input to generate the latent
	dimension content. Saves a file where every other first unit is one of the input context and every other second unit is the resulting prediction, optionally following after the initial unit with the input to use for
	the latent space.
	
	Params
	------
	
	model: MahlerNet
		The model to use
		
	dp: DurationSet
		The duration set to use for conversion from data to midi
		
	save_name: str
		The name to use as part of the name of saved output
		
	ctxs: tuple of (int, numpy array)
		The contexts to use for the predictions.
		
	z_unit: tuple like (int, numpy array) or None
		The input to convert to latent space to use throughout.
		
	meter: int
		The induced meter of the output which also affects which duration set is used internally while generating. This affects the range of each output unit which is limited to 2, 3 or 4 beats depending on the given meter.
		Should match what is reconstructed if this is known.
	
	length: int
		The number of units to generate for each prediction
		
	triplet: bool
		Whether the default offset and / or duration should be triplet eighth note or duplet eighth note if offset or duration is not modelled by this model.
		
'''
def n_prediction_test(model, dp, save_name, ctxs, z_unit, meter = 4, length = 1, triplet = False):
	save_name = "EMPTY" if save_name is None else save_name
	if z_unit is None:
		z = np.random.normal(0.0, 1.0, [1, model.params["vae"]["z_dim"]])
	else:
		if params.start_ctx is None:
			ctx = model.empty_context(False)
		else:
			ctx = model.context(model.empty_context(True))
		z = model.latent(z_unit[1], ctx) # (1, latent_dim)
	data = []
	c_data = []
	ctx_txt = []
	for (base_id, c) in ctxs:
		ctx_txt += [str(base_id)]
		c_data += [c]
		data += [model.generate_sample(ctx = (True, c), inp = (False, z), meter = meter, length = length, triplet = triplet)]
	# data is an array with (data_repr, step_unit) where data_repr is an array of 5-tuples
	default_step_unit = data[0][1]
	data = list(map(lambda a: a[0], data)) # [0] picks all arrays of 5-tuples
	base_time = data[-1][0][4] - data[-1][0][3]
	ds = data[-1][0][2]
	start = 0
	music = []
	if z_unit is not None:
		music += [(z_unit[1], 0, ds, start, start + base_time)]
		start += base_time
	for c, tup_arr in zip(c_data, data):
		music += [(c, 0, ds, start, start + base_time)]
		start += base_time
		for tup in tup_arr:
			music += [(tup[0], tup[1], tup[2], start, start + base_time)]
			start += base_time		
	files = dp.mp.data_to_midi([music], True, default_step_unit)
	ctx_txt = "_".join(ctx_txt)
	os.makedirs(os.path.join(model.params["save_dir"], "generated"), exist_ok = True)
	if z_unit is None:
		files[0].save(os.path.join(model.params["save_dir"], "generated", "n_pred_" + save_name + "_unit_" + ctx_txt + "_" + time.strftime("%Y%m%d_%H.%M.%S", time.localtime()) + ".mid"))
	else:
		files[0].save(os.path.join(model.params["save_dir"], "generated", "n_pred_" + save_name + "_unit_" + ctx_txt + "z" + str(z_unit[0]) + "_" + time.strftime("%Y%m%d_%H.%M.%S", time.localtime()) + ".mid"))

'''
	Convenience function to interpolate between two latent vectors. Input units must contains exactly two units with latent vectors and an optional single unit with context. This function then interpolates linearly between
	the two latents vectors resulting from the input units, along with the given context, if applicable. Ignores the '-' unit.
	
	Params
	------
	
	model: MahlerNet
		The model to use
		
	dp: DurationSet
		The duration set to use for conversion from data to midi
		
	save_name: str
		The name to use as part of the name of saved output
		
	ctx: tuple of (int, numpy array) or None
		The unit number in the input file and the optional context to use during the entire interpolation.
		
	zs: list of size 2 with tuples like (int, numpy array)
		The two inputs to convert to latent space to interpolate between.
		
	intpol_sz: int
		The number of steps to interpolate in, excluding the starting state.
		
	meter: int
		The induced meter of the output which also affects which duration set is used internally while generating. This affects the range of each output unit which is limited to 2, 3 or 4 beats depending on the given meter.
		Should match what is reconstructed if this is known.
	
	triplet: bool
		Whether the default offset and / or duration should be triplet eighth note or duplet eighth note if offset or duration is not modelled by this model.
		
	default_ctx: str or None
		Default value to use for missing contexts, may either be None, which results in an empty context (all 0's) or "START" which results in the START context (start token only set to 1)		
		
		
'''
def interpolation_test(model, dp, save_name, ctx, zs, use_slerp = False, intpol_sz = 10, meter = 4, triplet = False, default_ctx = None):
	base_id1, inp1 = zs[0]
	base_id2, inp2 = zs[1]
	z1 = model.latent(inp1) # (1, latent_dim)
	z2 = model.latent(inp2)
	intpol_sz = max(intpol_sz, 1)
	ts = np.linspace(0, 1, intpol_sz + 1)
	intpol = get_interpolation(z1[0], z2[0], ts, use_slerp)
	data = [model.generate_sample(ctx = (True, ctx[1]) if ctx is not None else default_ctx, inp = (False, intpol[i: i + 1]), meter = meter, length = 1, triplet = triplet) for i in range(len(intpol))] # array of (data_repr (array of 5-tuple), step_unit) in each
	default_step_unit = data[0][1]
	data = list(map(lambda a: a[0][0], data)) # [0][0]first takes every array of 5-tuples and picks first (and only) 5-tuple since we only generated one
	base_time = data[-1][4] - data[-1][3]
	start = 0
	music = []
	if ctx is not None:
		music += [(inp1, 0, data[-1][2], start, start + base_time)]
		start += base_time
	music += [(inp1, 0, data[-1][2], start, start + base_time)]
	start += base_time
	for i in range(len(data)):
		music += [(data[i][0], data[i][1], data[i][2], start, start + base_time)]
		start += base_time
	music += [(inp2, 0, music[0][2], start, start + base_time)]
	files = dp.mp.data_to_midi([music], True, default_step_unit)
	os.makedirs(os.path.join(model.params["save_dir"], "generated"), exist_ok = True)
	if ctx is not None:
		files[0].save(os.path.join(model.params["save_dir"], "generated", "intpol_" + save_name + "_unit" + str(base_id1) + "-" + str(base_id2) + "_ctx" + str(ctx[0]) + "_intpol_sz_" + str(intpol_sz) + "_" + time.strftime("%Y%m%d_%H.%M.%S", time.localtime()) + ".mid"))
		print("[RUN]: generated interpolation in " + str(intpol_sz) + " steps with " + str(meter) + "-based meter from unit " + str(base_id1) + " to unit " + str(base_id2) + " with context " + str(ctx[0]) + " in input file "  + "with slerp" if use_slerp else "with linear interpolation")
	else:
		files[0].save(os.path.join(model.params["save_dir"], "generated", "intpol_" + save_name + "_unit" + str(base_id1) + "-" + str(base_id2) + "_intpol_sz_" + str(intpol_sz) + "_" + time.strftime("%Y%m%d_%H.%M.%S", time.localtime()) + ".mid"))
		print("[RUN]: generated interpolation in " + str(intpol_sz) + " steps with " + str(meter) + "-based meter from unit " + str(base_id1) + " to unit " + str(base_id2) + " in input file " + ("with slerp" if use_slerp else "with linear interpolation"))			
		
def get_interpolation(v0, v1, t, use_slerp = True):
	dot = np.sum(v0 * v1) / (np.linalg.norm(v0) * np.linalg.norm(v1))
	alpha = np.arccos(dot)
	if alpha < 0.0000175 or not use_slerp: # corresponds to 1 degree in radians, use linear interpolation instead, almost the same thing at this short angle anyways
		t = t.reshape([len(t), 1])
		intpol = (1 - t) * v0 + t * v1 # (intpol_sz + 1, latent_dim)
	else:
		sin_alpha = np.sin(alpha)
		s0 = np.sin((1 - t) * alpha) / sin_alpha
		s1 = np.sin(t * alpha) / sin_alpha
		intpol = (s0.reshape([s0.shape[0], 1]) * v0.reshape([1, v0.shape[0]])) + (s1.reshape([s1.shape[0], 1]) * v1.reshape([1, v1.shape[0]]))
	
	return intpol
	
if __name__ == "__main__":
	main()
