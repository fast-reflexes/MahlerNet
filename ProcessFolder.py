import pickle, os, re, os.path, sys
from DataProcessor import DataProcessor
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='Process MIDI files into data representation, sequences of data vectors and back.')
parser.add_argument('root_dir', action = "store", nargs = "?", default = "", type = str, metavar='ROOT_DIR', help = 'overall root directory to use')
parser.add_argument('input_dir', action = "store", nargs = "?", default = "input", type = str, metavar='INPUT_DIR', help = 'input dir where input MIDIs reside')
parser.add_argument('data_dir', action = "store", nargs = "?", default = "data", type = str, metavar='DATA_DIR', help = 'data dir where MIDIs processed with the -d flag reside')
parser.add_argument('sequence_dir', action = "store", nargs = "?", default = "seq", type = str, metavar='SEQUENCE_DIR', help = 'sequence dir where sequences of MIDIs processed with the -s flag reside')
parser.add_argument('midi_dir', action = "store", nargs = "?", default = "midi", type = str, metavar='MIDI_DIR', help = 'output dir where data converted to MIDIs with the -m and -M flags reside')

parser.add_argument('--data', "-d", action = "store_true", default = False, help = 'converts MIDI files from the input dir to data in the data dir')
parser.add_argument('--seq', "-s", action = "store_true", default = False, help = 'converts data representation in the data dir to training sequences in the sequence dir')
parser.add_argument('--midi', "-m", action = "store_true", default = False, help = 'converts data representation in the data dir back to MIDI in the midi')
parser.add_argument('--train', "-t", action = "store_true", default = False, help = 'converts sequence representation to a generator with training / validation data and tests its functionality and outputs a summary')
parser.add_argument('--midi_train', "-M", action = "store_true", default = False, help = 'outputs concatenated data from the training generator and converts it into MIDI files in the midi dir')

parser.add_argument('--ctx_len', action = "store", default = 1, type = int, metavar = 'CTX_LEN', help = 'the number of sequence base units to use for context in the training sequences')
parser.add_argument('--inp_len', action = "store", default = 1, type = int, metavar = 'INP_LEN', help = 'the number of sequence base units to use for input in the training sequences')
parser.add_argument('--base', action = "store", default = "bar", type = str, metavar = 'BASE_UNIT', help = 'the unit of choice to build all training sequences on, should be\"bar\" or the name of a valid duration (for example \"d4\" or \"t8\" for duple time quarter or triple time eight note)')
parser.add_argument('--validation', action = "store", default = 0.0, type = float, metavar = 'VALIDATION_FRACTION', help = 'the ratio of all training sequences to use for validation data (must be in [0.0, 1.0])')
parser.add_argument('--batch_sz', action = "store", default = 1, type = int, metavar = 'BATCH_SZ', help = 'the number of sequences in a batch of training / validation data')

parser.add_argument('--orig', action = "store_true", default = False, help = 'runs a preconfigured baseline setup')
parser.add_argument('--redo', action = "store_true", default = False, help = 'whether to redo already existing output files or just do the files that don\'t exist')

dp = DataProcessor()

if __name__ == "__main__":
	params = parser.parse_args()

	# Turn MIDI files into a data representation
	if params.data:
		if params.orig:
			# inputs can be a tuple like (name, [list of adjustments on the form (time to apply adjustment, adjustment)], start offset, end offset, ignored time signatures)
			m6_1 = ("Mahler61.mid", [(476160, -180), (624960, 480), (826560, 480)], None, None, [])
			m6_4_1 = ("MAHL64O1.MID", [(960, -960), (501840, 240)], None, 722880, [])
			m6_4_2 = ("MAHL64O2.MID", [(960, -960)], None, None, [])
			m6_4_3 = ("MAHL64O3.MID", [(960, -960), (68160, -960)], 9600, None, [])
			m5_2 = ("Mahler5-2.mid", [(733440, -960), (742080, -960)], None, None, [(5, '*')])
			m5_3_1 = ("Mahlsy531.mid", [(480, -480)], None, None, [])
			m5_3_2 = ("Mahlsy532.mid", [(480, -480)], None, None, [])
			dp.midi_to_data([m6_1, [m6_4_1, m6_4_2, m6_4_3], [m5_3_1, m5_3_2], "Mahler62.mid", "Mahler63.mid", "BeatIt.mid"], root_dir = params.root_dir, input_dir = params.input_dir, save_dir = params.data_dir, redo_existing = params.redo)
		else:
			dp.midi_to_data(root_dir = params.root_dir, input_dir = params.input_dir, save_dir = params.data_dir, redo_existing = params.redo)

	# Turn the data representation into a binary matrix format with sequences, keeping track of the sizes of sequences
	if params.seq:
		dp.data_to_sequences(sequence_base = params.base, ctx_length = params.ctx_len, inp_length = params.inp_len, \
							root_dir = params.root_dir, input_dir = params.data_dir, save_dir = params.sequence_dir)

	# Turn data representation back to MIDI
	if params.midi:
		dp.data_to_midi(root_dir = params.root_dir, input_dir = params.data_dir, save_dir = params.midi_dir)
		
	# Find stats manually and inject
	if params.train:
		dp.setup_training(root_dir = params.root_dir, input_dir = params.sequence_dir, validation_set_ratio = params.validation)
		print("[PROC_DIR]: Total samples:", dp.total)
		print("[PROC_DIR]: Max timesteps context length:", dp.max_context_length)
		print("[PROC_DIR]: Max timesteps input length:", dp.max_input_length)
		print("[PROC_DIR]: Number of timestep features:", dp.features)
		print("[PROC_DIR]: Size of training set:", len(dp.training_set))
		print("[PROC_DIR]: Size of validation set:", len(dp.validation_set))

		# Generate batches instead of saving the entire training data_to_midi
		gen = dp.sequences_to_training_data_generator(params.batch_sz, validation = False, root_dir = params.root_dir, input_dir = params.sequence_dir)
		ds = dp.mp.default_common_duration_set
		if params.base == "bar":
			unit_length = ds.bar_duration.duration
		else:
			unit_length = ds.basic_set[params.base].duration
		inp_length = unit_length * params.inp_len
		ctx_length = unit_length * params.ctx_len	
		ti_output = []
		tc_output = []
		vi_output = []
		vc_output = []
		ctx_time = 0
		inp_time = 0
		ctx_lengths = []
		inp_lengths = []
		ctx_lengths_sum = 0
		inp_lengths_sum = 0
		total_batches = 0
		for batch in gen:
			ctx = np.copy(batch[0])
			inp = np.copy(batch[2])
			ctx_lengths += batch[1].tolist()
			#print(ctx_lengths)
			inp_lengths += batch[3].tolist()
			ctx_lengths_sum += np.sum(batch[1])
			inp_lengths_sum += np.sum(batch[3])
			if params.midi_train:
				tc_output += [(ctx[0, :, :], 0, ds, ctx_time, ctx_time + ctx_length)]
				ctx_time += ctx_length 
				ti_output += [(inp[0, :, :], 0, ds, inp_time, inp_time + inp_length)]
				inp_time += inp_length
			total_batches += len(batch[0]) # context data_to_midi
		print("[PROC_DIR]: Average ctx length:", ctx_lengths_sum / dp.total)
		print("[PROC_DIR]: Average inp length:", inp_lengths_sum / dp.total)
		ctx_lengths.sort()
		inp_lengths.sort()
		print("[PROC_DIR]: Median ctx length:", ctx_lengths[len(ctx_lengths) // 2] if (len(ctx_lengths) % 2) else np.sum(ctx_lengths[len(ctx_lengths) // 2 - 1: len(ctx_lengths) // 2]) / 2)
		print("[PROC_DIR]: Median inp length:", inp_lengths[len(inp_lengths) // 2] if (len(inp_lengths) % 2) else np.sum(inp_lengths[len(inp_lengths) // 2 - 1: len(inp_lengths) // 2]) / 2)
		print("[PROC_DIR]: Totally processed", total_batches, "training batches")
		if params.midi_train:
			midis = dp.mp.data_to_midi([ti_output], True)
			path = os.path.join(params.root_dir, params.midi_dir, "inp_training_generator")
			for i, midi in enumerate(midis):
				if len(midis) == 1:
					midi.save(path + ".mid")
				else:
					midi.save(path + "_" + str(i) + ".mid")
			midis = dp.mp.data_to_midi([tc_output], True)
			path = os.path.join(params.root_dir, params.midi_dir, "ctx_training_generator")
			for i, midi in enumerate(midis):
				if len(midis) == 1:
					midi.save(path + ".mid")
				else:
					midi.save(path + "_" + str(i) + ".mid")			