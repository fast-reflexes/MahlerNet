import os
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = 'T' # https://github.com/ContinuumIO/anaconda-issues/issues/905 needed to install SIGINT handler on Windows 64-bit with Anaconda, otherwise Scipy handler overrides somehow
import numpy as np
import functools
import os.path, pprint, signal
import tensorflow as tf
import numpy as np
from DataProcessor import DataProcessor
import math, time, sys, copy
from utilities import print_divider

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
tf.logging.set_verbosity(tf.logging.FATAL) # can use DEBUG, INFO, WARN, ERROR, FATAL
np.set_printoptions(threshold = sys.maxsize, precision = 30)

class MahlerNet:
			
	def __init__(self, params):
		self.params = {	
			# general
			"root_dir": "",
			"save_dir": "",
			"model_name": None,
			"validation_set_ratio": 0.0,
			"save_graph": False,
			"save_stats": False,
			"save_model": None, # use a number and then a letter e or s indicating if the number refers to epochs or global steps, for example "25e", use literal None not to save at all
			"end_of_epoch_fns": 1,
			"use_randomness": True,
			"use_gpu": True,
			"gpu": None,
			"verbose": 1, # 0 means no more printouts than training progress, 1 means basic output, 2 means verbose output
			"batch_sz" : 4,
			"epochs" : 1000,
			"num_pitches" : 96, 
			"num_x_features": None, 
			"num_y_features": None,
			"num_ctx_features": None,
			"num_xr_features": None,
			"num_features": None,
			"num_instruments" : None,
			"num_durations" : None,
			"num_beats": None,
			"num_special_tokens": None, 
			"learning_rate" : 0.001,	
			"optimizer": "rmsprop",
			"batch_norm": False, # puts batch normalization on all non-recurrent layers
			"batch_norm_before_act": True,
			"act": "tanh",
			"generation": {
				"default_durations": None,
				"default_bar": None,
				"default_beats": None,
				"sequence_base": None,
				"ctx_length": 1,
				"inp_length": 1,
				"default_duration_set": None,
				"default_duration_sets": None,
				"default_duplet_duration": None,
				"default_triplet_duration": None				
			},
			"property_order": {
				"ctx": ["offset", "beats", "duration", "pitch", "instrument", "special_tokens"],
				"inp": ["offset", "beats", "duration", "pitch", "instrument"],
				"dec": {
					"inp": ["offset", "beats", "duration", "pitch", "instrument", "active_instruments", "active_pitches"],
					"out": ["offset", "duration", "pitch", "instrument"]
				}
			},
			"modelling_properties": {
				"offset": {
					"include": True,
					"num_features": None,
					"key": "o",
					"indices": None,
					"dropout": 0.0, 
					"multiclass": False,
					"dist_metric": True,
					"generation_temperature": 1.0
					
				},
				"duration": {
					"include": True,
					"num_features": None,
					"key": "d", 
					"indices": None, 
					"dropout": 0.0,
					"multiclass": False,
					"dist_metric": True,
					"next_step_reduction": True,
					"generation_temperature": 1.0
				},
				"pitch": {
					"include": True,
					"num_features": None,
					"key": "p",
					"indices": None,
					"dropout": 0.0, 
					"multiclass": False,
					"same_step_reduction": True,
					"dist_metric": True,
					"generation_temperature": 1.0
				},				
				"instrument": {
					"include": True,
					"num_features": None,
					"key": "i", 
					"indices": None,
					"dropout": 0.0,
					"multiclass": False,
					"dist_metric": False,
					"generation_temperature": 1.0
				},
				"special_tokens": {
					"include": True,
					"num_features": None,
					"key": "s",
					"indices": None
				},
				"beats": {
					"include": True,
					"num_features": None,
					"key": "b",
					"indices": None
				},
				"active_pitches": {
					"include": True,
					"num_features": None,
					"key": "ap",
					"indices": None
				},
				"active_instruments": {
					"include": True,
					"num_features": None,
					"key": "ai",
					"indices": None
				}
			},
			"model" : {
				"ctx": True,
				"inp": True,
				"vae": True,
				"dec": True
			},
			"cell_versions": {
				"lstm": {
					"lstm": tf.nn.rnn_cell.LSTMCell,
					"block": tf.contrib.rnn.LSTMBlockCell,
					"bn": tf.contrib.rnn.LayerNormBasicLSTMCell
				},
				"gru": {
					"gru": tf.nn.rnn_cell.GRUCell,
					"blockv2": tf.contrib.rnn.GRUBlockCellV2,
					"keras_gru": tf.keras.layers.GRUCell
				}
			},
			# encoder input summary
			"inp_enc" : {
				"type": "lstm",
				"version": "lstm",
				"dropout": 0.5,
				"init": "var",
				"num_layers" : 1,
				"sz_state" : 128,
				"bidirectional_type": "default"
			},
			
			#encoder context summary
			"ctx_enc" : {
				"type": "lstm",
				"version": "lstm",
				"dropout": 0.5,
				"init": "var",
				"num_layers" : 1, 
				"sz_state" : 128,
				"bidirectional_type": "stack" # "stack" or "default", the former uses a bidirectional rnn where directions are mixed for each layer whereas the latter uses separate directions for layers and concatenates after
			},
			
			# vae
			"vae" : {
				"z_dim" : 64,
				"dropout": 0.0
			},
			
			# decoder
			"dec" : {
				"num_layers" : 1,
				"sz_state" : 256,
				"model": "lstm", # "balstm" or "lstm"
				"framework": "rnn", # Tensorflow framework to use: "seq2seq" or "rnn"
				"type": "lstm", # Cell type to use: "gru" or "lstm"
				"version": "lstm",
				"init": "z",	# How to initialize the states of the decoder: "var" (trainable variables) or "zeros" or "z" (latent vector from vae)
				"layer_strategy": "same", # How to initialize different layers (if more than one): "same" or "diff"
				"scheduled_sampling": False,
				"scheduled_sampling_scheme": "sigmoid", # "sigmoid" or "linear"
				"scheduled_sampling_min_truth": 0.0, 
				"scheduled_sampling_mode": "sample", # "sample" or "max" depending on if we want to sample from distribution or take the most probable class
				"scheduled_sampling_rate": 1000, # ln(rate) * rate is the number of steps when there will be a 50% chance that we sample from output instead of ground truth, for rate 1000 this is about 6907
				"dropout": 0.5,
				"balstm": {
					"sz_conv": 25, # the size of the BALSTM convolution, this is how many surrounding pitches are taken into account
					"init": "same", # "diff" or "same"
					"add_pitch": True,
					"num_filters": 64,
					"conv_dropout": 0.0
				},
				"feed": {
					"z": {
						"use": True,
						"strategy": "proj", # "raw" or "proj" can't combine "proj" with "layer_strategy" "diff" in which case the latters defaults to "same"
						"sz": 128
					},
					"ctx": {
						"use": True,
					}
				}
			},
			"output_layers": {
				"offset": [], # last output layer without activation to the correct number of output units will be taken care of automatically
				"pitch": [],
				"duration": [],
				"instrument": []
			},
			"scopes": {
				"offset": "",
				"pitch": "",
				"duration": "",
				"instrument": ""
			},
			"loss": {
				"vae": True,
				"recon": True,
				"regularization": 0.05,
				"free_bits": 0,
				"beta_annealing": False,
				"beta_max": 0.0,
				"beta_rate": 0.99999,
				"p_weight": 1.0,
				"o_weight": 1.0,
				"d_weight": 1.0,
				"i_weight": 1.0,
				"framework": "seq2seq" # "seq2seq" to use sequence_loss, "plain" to use regular loss
			},
			"training_output": {
				"dists": False # whether to continuously output the distribution of guesses, targets and the diff between
			}	
		}
		p = self.params
		
		def update_params(param_object, incoming):
			for key in incoming:
				if key in param_object:
					if isinstance(incoming[key], dict) and isinstance(param_object[key], dict):
						update_params(param_object[key], incoming[key])
					elif not isinstance(incoming[key], dict) and not isinstance(param_object[key], dict):
						param_object[key] = incoming[key]
			
		update_params(p, params)
		
		# adjust and process some input parameters
		if not p["use_randomness"]:
			tf.reset_default_graph()
			tf.set_random_seed(12345)
			np.random.seed(12345)
		if p["use_gpu"] and tf.test.is_gpu_available():
			p["gpu"] = True
			p["cell_versions"]["lstm"]["cudnn"] = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell
			p["cell_versions"]["gru"]["cudnn"] = tf.contrib.cudnn_rnn.CudnnCompatibleGRUCell
			config = tf.ConfigProto()
		else:
			p["gpu"] = False
			config = tf.ConfigProto(device_count = {'GPU': 0})
			os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # in case there is a gpu but we don't want to use it
			
		num_x_features = 0 # total numbers of features in the input to the decoder
		num_xr_features = 0 # the number of original features in the BALSTM input (if used) except for the pitch input
		num_y_features = 0 # total number of features in the output
		num_ctx_features = 0 # total number of features in the context
		num_features = 0
		p["modelling_properties"]["special_tokens"]["num_features"] = p["num_special_tokens"]
		p["modelling_properties"]["offset"]["num_features"] = p["num_durations"]
		p["modelling_properties"]["beats"]["num_features"] = p["num_beats"]
		p["modelling_properties"]["duration"]["num_features"] = p["num_durations"]
		p["modelling_properties"]["pitch"]["num_features"] = p["num_pitches"]
		p["modelling_properties"]["active_pitches"]["num_features"] = p["num_pitches"]
		p["modelling_properties"]["instrument"]["num_features"] = p["num_instruments"]
		p["modelling_properties"]["active_instruments"]["num_features"] = p["num_instruments"]
		
		for cat in p["modelling_properties"]:
			cat_params = p["modelling_properties"][cat]
			if cat_params["include"]:
				if cat in p["property_order"]["ctx"]:
					num_ctx_features += cat_params["num_features"]
				if cat in p["property_order"]["inp"]:
					num_y_features += cat_params["num_features"]
				if cat in p["property_order"]["dec"]["inp"]:
					num_x_features += cat_params["num_features"]
				if cat in p["property_order"]["dec"]["out"]:
					p["output_layers"][cat] += [(p["modelling_properties"][cat]["num_features"], None)]
			num_features += cat_params["num_features"]
		num_xr_features = num_x_features
		if p["modelling_properties"]["pitch"]["include"]:
			num_xr_features -= p["modelling_properties"]["pitch"]["num_features"]
		if p["modelling_properties"]["active_pitches"]["include"]:
			num_xr_features -= p["modelling_properties"]["pitch"]["num_features"]
		print("[MAHLERNET]: features in total (only based on what properties are being modelled): ", num_features)
		print("[MAHLERNET]: features in context (only based on what properties are being modelled): ", num_ctx_features)		
		print("[MAHLERNET]: features in input to decoder (only based on what properties are being modelled): ", num_x_features)
		print("[MAHLERNET]: features in non-pitch input to balstm (only based on what properties are being modelled): ", num_xr_features)
		print("[MAHLERNET]: features in output from decoder (only based on what properties are being modelled): ", num_y_features)
		
		p["num_x_features"] = num_x_features
		p["num_y_features"] = num_y_features
		p["num_ctx_features"] = num_ctx_features
		p["num_xr_features"] = num_xr_features
		p["num_features"] = num_features
		
		# correct input parameters if illegal combinations have been chosen
		if p["act"] not in ["relu", "tanh", "sigmoid", "leakyrelu"]:
			assert(False), '[MAHLERNET]: parameter error: unknown default activation function, please use one of "relu", "tanh", "sigmoid" or "leakyrelu"'
		if not p["model"]["vae"] and p["model"]["inp"]:
			assert(False), "[MAHLERNET]: parameter error: can't use input encode without using the vae"
		if p["model"]["vae"] and (not p["model"]["inp"] and not p["model"]["ctx"]):
			assert(False), "[MAHLERNET]: parameter error: can't use vae without either of input or context"			
		if not p["modelling_properties"]["pitch"]["include"] and p["modelling_properties"]["active_pitches"]["include"]:
			assert(False), "[MAHLERNET]: parameter error: can't use active pitches as additional pitch input when not modelling pitches"
		if not p["modelling_properties"]["offset"]["include"] and p["modelling_properties"]["beats"]["include"]:
			assert(False), "[MAHLERNET]: parameter error: can't model beats without modelling offset"
		if not p["modelling_properties"]["offset"]["include"] and p["modelling_properties"]["pitch"]["same_step_reduction"]:
			assert(False), "[MAHLERNET]: parameter error: can't preform same step reduction while modelling pitch without offset"			
		if not p["modelling_properties"]["instrument"]["include"] and p["modelling_properties"]["active_instruments"]["include"]:
			assert(False), "[MAHLERNET]: parameter error: can't use active instruments as additional instrument input when not modelling instruments"		
		if not p["modelling_properties"]["pitch"]["include"] and p["dec"]["model"] == "balstm":
			assert(False), "[MAHLERNET]: parameter error: can't use BALSTM without modelling pitch"
		if p["dec"]["num_layers"] == 1 and p["dec"]["layer_strategy"] == "diff":
			assert(False), "[MAHLERNET]: parameter error: only one layer in decoder, switch to layer_strategy \"same\""
		if p["dec"]["init"] == "z" and not p["model"]["vae"]:
			assert(False), "[MAHLERNET]: parameter error: can't use latent vector as initialization for decoder when not vae is in use"
		if p["dec"]["feed"]["z"]["use"] and not p["model"]["vae"]:
			assert(False), "[MAHLERNET]: parameter error: can't use latent vector as feed while decoding when not vae is in use"			
		if p["dec"]["scheduled_sampling"] and p["dec"]["model"] == "balstm":
			assert(False), "[MAHLERNET]: parameter error: can't use scheduled sampling togther with BALSTM since there is no way to sample the next timestep from single pitches"
		if p["dec"]["scheduled_sampling"] and p["modelling_properties"]["pitch"]["include"] and p["modelling_properties"]["active_pitches"]["include"]:
			assert(False), "[MAHLERNET]: parameter error: can't use scheduled sampling while including active pitches as an input feature"
		if p["dec"]["scheduled_sampling"] and p["modelling_properties"]["instrument"]["include"] and p["modelling_properties"]["active_instruments"]["include"]:
			assert(False), "[MAHLERNET]: parameter error: can't use scheduled sampling while including active instruments as an input feature"		
		if p["dec"]["scheduled_sampling"] and p["modelling_properties"]["beats"]["include"]:
			assert(False), "[MAHLERNET]: parameter error: scheduled sampling can not be used with beats since it affects the upcoming timesteps"
		if p["dec"]["scheduled_sampling"] and p["dec"]["framework"] == "rnn":
			assert(False), "[MAHLERNET]: parameter error: scheduled sampling can only be implemented in seq2seq framework"

		if p["act"] == "relu":
			p["act"] = tf.nn.relu
		elif p["act"] == "tanh":
			p["act"] = tf.nn.tanh
		elif p["act"] == "sigmoid":
			p["act"] = tf.nn.sigmoid
		else:
			p["act"] = tf.nn.leaky_relu
		for part, part_name in [("inp", "inp_enc"), ("ctx", "ctx_enc"), ("dec", "dec")]:
			if not p["gpu"] and p["model"][part] and p[part_name]["version"] == "cudnn":
				assert(False), "[MAHLERNET]: parameter error: cudnn cells can only be used with GPU"
		for cat in p["property_order"]["dec"]["out"]:
			if p["modelling_properties"][cat]["include"]:
				if p["modelling_properties"][cat]["dist_metric"] and p["modelling_properties"][cat]["multiclass"]:
					assert(False), "[MAHLERNET]: parameter error: can't use distance metric for multiclass properties"
				if p["modelling_properties"][cat]["multiclass"] and p["loss"]["framework"] == "seq2seq":
					assert(False), "[MAHLERNET]: parameter error: Altered parameter: can't use seq2seq framework for loss with multiclass properties"		

		# assert(param in kwargs and isinstance(kwargs[param], int) and kwargs[param] > 0), "Input param error: " + param " = " str(kwargs[param])

		
		self.weights = {
			"vae" : {},
			"output" : {},
			"conv" : {}
		}
		self.ops = lambda: None # object to store ops in

		config.gpu_options.allow_growth = True
		config.gpu_options.per_process_gpu_memory_fraction = 1.0
		self.session = tf.Session(config=config)
		print("[MAHLERNET]: Input parameters ok, printing used parameters:")
		pp = pprint.PrettyPrinter()
		pp.pprint(p)
		
		def check_params(param_object):
			for key in param_object:
				if isinstance(param_object[key], dict):
					check_params(param_object[key])
				else:
					if param_object[key] is None:
						print("BULL!", param_object, key)
						exit()
					
		check_params(p)
	
	def save_graph(self, output_folder = "graphs"):
		# start Tensorboard session with "tensorboard --logdir=graphs --host=localhost" then go to localhost in browser
		path = os.path.join(self.params["save_dir"], output_folder)
		if not os.path.exists(path):
			os.makedirs(path)				
		writer = tf.summary.FileWriter(path, self.session.graph)
		
	def save_model(self, save_file):
		saver = tf.train.Saver()
		path = os.path.join(self.params["save_dir"], "trained_models")
		os.makedirs(path, exist_ok = True)
		path = os.path.join(path, save_file)
		save_path = saver.save(self.session, path)
		print("[MAHLERNET]: successfully saved model to \"" + path + "\"")
		
	def load_model(self, file_name):
		saver = tf.train.Saver()
		path = os.path.join(self.params["save_dir"], "trained_models", file_name)
		saver.restore(self.session, path)
		print("[MAHLERNET]: successfully loaded model from \"" + path + "\"")
		
	def print_op(self, op, text = "PRINT"):
		return tf.Print(op, [op], summarize = 10000000, message = text)
		
	def variable_count(self):
		if not hasattr(self, "num_variables"):
			shapes = [var.get_shape() for var in tf.trainable_variables()]
			prods = [np.prod(shape) for shape in shapes]
			self.num_variables = np.sum(prods)
		return self.num_variables

	def build(self):
	
		def _inputs(o, p):
			
			with tf.variable_scope("inputs"):
			
				o.inp_enc_dropout = tf.placeholder(tf.float32, (), name = "inp_enc_dropout")
				o.ctx_enc_dropout = tf.placeholder(tf.float32, (), name = "ctx_enc_dropout")
				o.vae_dropout = tf.placeholder(tf.float32, (), name = "vae_dropout")
				o.dec_dropout = tf.placeholder(tf.float32, (), name = "dec_dropout")
				o.dec_balstm_conv_dropout = tf.placeholder(tf.float32, (), name = "dec_balstm_conv_dropout")
				o.pitch_dropout = tf.placeholder(tf.float32, (), name = "pitch_dropout")
				o.offset_dropout = tf.placeholder(tf.float32, (), name = "offset_dropout")
				o.duration_dropout = tf.placeholder(tf.float32, (), name = "duration_dropout")
				o.instrument_dropout = tf.placeholder(tf.float32, (), name = "instrument_dropout")
				o.batch_sz = tf.placeholder(tf.int32, (), name = "batch_sz")
				o.global_step = tf.get_variable("global_step", (), dtype = tf.float32, trainable = False, initializer=tf.constant_initializer(0.0))
				# placeholders for inputs needed during generation
				o.instrument_temp = tf.constant(1.0, dtype = tf.float32, name = "instrument_temp")
				o.pitch_temp = tf.constant(1.0, dtype = tf.float32, name = "pitch_temp")
				o.offset_temp = tf.constant(1.0, dtype = tf.float32, name = "offset_temp")
				o.duration_temp = tf.constant(1.0, dtype = tf.float32, name = "duration_temp")
				o.sample_from_predictions = tf.constant(False, dtype = tf.bool, name = "sample_from_predictions")
				o.training = tf.placeholder(tf.bool, (), name = "training")
				o.durations = tf.placeholder(tf.float32, (1, 1, p["num_durations"]), name = "durations")
				o.last_time = tf.placeholder(tf.float32, (), name = "last_time")
				
				o.bar = tf.placeholder(tf.float32, (), name = "bar")
				o.beats = tf.placeholder(tf.float32, (1, 1, p["num_beats"]), name = "beats")
				o.inp_lengths = tf.placeholder(tf.int32, [None], name = "inp_lengths")
				# calculate actual lengths for input
				o.inp_lengths_mask = tf.expand_dims(tf.sequence_mask(o.inp_lengths, dtype = tf.float32), axis = 2)
				o.inp_lengths_sum = tf.reduce_sum(o.inp_lengths, name = "inp_lengths_sum")
				o.inp_lengths_max = tf.reduce_max(o.inp_lengths, name = "inp_lengths_max")
				o.y_p_zero = tf.constant(0)
				o.y_i_zero = tf.constant(0)
				
				# placeholders for training data
				y = []
				if p["dec"]["model"] == "balstm": # have to split pitch properties and the rest into different parts
					x_rest = []
					o.x_p = None
					o.x_ap = None
				else: # all in the same part but to preserve original order, we put the active_pitches and instruments last
					x = []
					x_a = []
				if p["model"]["ctx"]:
					o.ctx_s = tf.placeholder(tf.float32, (None, None, p["num_special_tokens"]), name = "ctx_s")
					ctx = [o.ctx_s]
				if p["modelling_properties"]["offset"]["include"]:
					print("[MAHLERNET]: adding \"offset\" as a parameter to model")
					o.x_o = tf.placeholder(tf.float32, (None, None, p["num_durations"]), name = "x_o")
					o.y_o_ = tf.placeholder(tf.float32, (None, None, p["num_durations"]), name = "y_o_")
					if p["training_output"]["dists"]:
						o.y_o_guess_tot_dist_var = tf.get_variable("y_o_guesses_tot_var", [p["num_durations"]], dtype = tf.int64, trainable = False, initializer=tf.constant_initializer(0))
					if p["dec"]["model"] == "balstm":
						x_rest += [o.x_o]
					else:
						x += [o.x_o]
					if p["model"]["ctx"]:
						o.ctx_o = tf.placeholder(tf.float32, (None, None, p["num_durations"]), name = "ctx_o")
						ctx += [o.ctx_o]
					y += [o.y_o_]
				if p["modelling_properties"]["beats"]["include"]:
					print("[MAHLERNET]: adding \"beats\" as a parameter to model")
					o.x_b = tf.placeholder(tf.float32, (None, None, p["num_beats"]), name = "x_b")
					o.y_b_ = tf.placeholder(tf.float32, (None, None, p["num_beats"]), name = "y_b_")
					if p["dec"]["model"] == "balstm":
						x_rest += [o.x_b]
					else:
						x += [o.x_b]
					if p["model"]["ctx"]:
						o.ctx_b = tf.placeholder(tf.float32, (None, None, p["num_beats"]), name = "ctx_b")
						ctx += [o.ctx_b]
					y += [o.y_b_]
				if p["modelling_properties"]["duration"]["include"]:
					print("[MAHLERNET]: adding \"duration\" as a parameter to model")
					o.x_d = tf.placeholder(tf.float32, (None, None, p["num_durations"]),name = "x_d")
					o.y_d_ = tf.placeholder(tf.float32, (None, None, p["num_durations"]),name = "y_d_")
					if p["training_output"]["dists"]:
						o.y_d_guess_tot_dist_var = tf.get_variable("y_d_guesses_tot_var", [p["num_durations"]], dtype = tf.int64, trainable = False, initializer=tf.constant_initializer(0))
					if p["dec"]["model"] == "balstm":
						x_rest += [o.x_d]
					else:
						x += [o.x_d]
					if p["model"]["ctx"]:
						o.ctx_d = tf.placeholder(tf.float32, (None, None, p["num_durations"]), name = "ctx_d")
						ctx += [o.ctx_d]	
					y += [o.y_d_]
				if p["modelling_properties"]["pitch"]["include"]:
					print("[MAHLERNET]: adding \"pitch\" as a parameter to model")
					o.x_p = tf.placeholder(tf.float32, (None, None, p["num_pitches"]), name = "x_p")
					o.y_p_ = tf.placeholder(tf.float32, (None, None, p["num_pitches"]), name = "y_p_")
					o.pd_mask = tf.reduce_max(o.y_p_, axis = 2)
					o.y_p_zero = tf.reduce_sum(tf.multiply(tf.subtract(tf.constant(1.0, dtype = tf.float32), o.pd_mask), tf.squeeze(o.inp_lengths_mask, axis = 2)))
					if p["training_output"]["dists"]:
						o.y_p_guess_tot_dist_var = tf.get_variable("y_p_guesses_tot_var", [p["num_pitches"]], dtype = tf.int64, trainable = False, initializer=tf.constant_initializer(0))
					if p["dec"]["model"] != "balstm":
						x += [o.x_p]
					if p["modelling_properties"]["active_pitches"]["include"]:
						print("[MAHLERNET]: adding \"active_pitches\" as a helping input during decoding to the model")
						o.x_ap = tf.placeholder(tf.float32, (None, None, p["num_pitches"]), name = "x_ap")
						if p["dec"]["model"] != "balstm":
							x_a += [o.x_ap]
					if p["model"]["ctx"]:
						o.ctx_p = tf.placeholder(tf.float32, (None, None, p["num_pitches"]), name = "ctx_p")
						ctx += [o.ctx_p]							
					y += [o.y_p_]					
				if p["modelling_properties"]["instrument"]["include"]:
					print("[MAHLERNET]: adding \"instrument\" as a parameter to model")
					o.x_i = tf.placeholder(tf.float32, (None, None, p["num_instruments"]),name = "x_i")
					o.y_i_ = tf.placeholder(tf.float32, (None, None, p["num_instruments"]),name = "y_i_")
					if not hasattr(o, "pd_mask"):
						o.pd_mask = tf.reduce_max(o.y_i_, axis = 2)
					o.y_i_zero = tf.reduce_sum(tf.multiply(tf.subtract(tf.constant(1.0, dtype = tf.float32), o.pd_mask), tf.squeeze(o.inp_lengths_mask, axis = 2)))
					#o.y_i_zero = tf.reduce_sum(tf.multiply(tf.squeeze(tf.cast(tf.logical_not(tf.cast(tf.reduce_max(o.y_i_, axis = 2), tf.bool)), tf.float32)), tf.squeeze(o.inp_lengths_mask))) # was before
					if p["training_output"]["dists"]:
						o.y_i_guess_tot_dist_var = tf.get_variable("y_i_guesses_tot_var", [p["num_instruments"]], dtype = tf.int64, trainable = False, initializer=tf.constant_initializer(0))
					if p["dec"]["model"] == "balstm":
						x_rest += [o.x_i]
					else:
						x += [o.x_i]
					if p["modelling_properties"]["active_instruments"]["include"]:
						print("[MAHLERNET]: adding \"active_instrument\" as a helping input during decoding to the model")
						o.x_ai = tf.placeholder(tf.float32, (None, None, p["num_instruments"]),name = "x_ai")
						if p["dec"]["model"] == "balstm":
							x_rest += [o.x_ai]
						else:
							x_a += [o.x_ai]
					if p["model"]["ctx"]:
						o.ctx_i = tf.placeholder(tf.float32, (None, None, p["num_instruments"]), name = "ctx_i")
						ctx += [o.ctx_i]	
					y += [o.y_i_]
				if p["dec"]["model"] == "balstm":
					if len(x_rest) > 0:
						o.x_rest = tf.concat(x_rest, axis = 2) if len(x_rest) > 1 else x_rest[0]
					else:
						o.x_rest = None
				else:
					x = x + x_a
					if len(x) > 0:
						o.x = tf.concat(x, axis = 2) if len(x) > 1 else x[0]
					else:	
						o.x = None # should never happen since it implies that we are not modelling anything at all and this should be checked earlier
				o.inp = tf.concat(y, axis = -1) if len(y) > 1 else y[0]
				if p["model"]["ctx"]:
					o.ctx = tf.concat(ctx, axis = -1) if len(ctx) > 1 else ctx[0]
					o.ctx_lengths = tf.placeholder(tf.int32, [None], name = "ctx_lengths")
			
		def _stats(o, p):
		
			with tf.variable_scope("stats"):
				# for dynamic batch size with last batch possibly smaller than others
				if p["model"]["ctx"]:
					# calculate actual lengths for context
					o.ctx_lengths_mask = tf.expand_dims(tf.sequence_mask(o.ctx_lengths, dtype = tf.float32), axis = 2)
					o.ctx_lengths_sum = tf.reduce_sum(o.ctx_lengths, name = "ctx_lengths_sum") # total number of timesteps in the current context
					o.ctx_lengths_max = tf.reduce_max(o.ctx_lengths, name = "ctx_lengths_max")	
				
				if p["dec"]["model"] == "balstm":
					# augment input lengths to work with the balstm that treats each dimension of features as a single sequence
					balstm_lengths = tf.expand_dims(o.inp_lengths, axis = 1) # arrange in (batch_sz, 1)
					balstm_lengths = tf.tile(balstm_lengths, (1, p["num_pitches"])) # augment each length the same times as there are number of pitches -> (batch_sz, num_pitches)
					o.balstm_batch_sz = tf.multiply(o.batch_sz, p["num_pitches"], name = "balstm_batch_sz")
					o.balstm_lengths = tf.reshape(balstm_lengths, [o.balstm_batch_sz], name = "balstm_lengths")
						
		def _rnn_setup(batch_sz, cell_type, cell_version, versions, bidirectional, state_sz, num_layers, rnn_dropout, proj_dropout, init_state, prefix, training):

			with tf.variable_scope(prefix + "_rnn_setup"):
				tot_cells = 2 * num_layers if bidirectional else num_layers
				if cell_type == "gru":
					if cell_version == "blockv2":
						cells = [versions[cell_type][cell_version](num_units = state_sz) for i in range(tot_cells)]
					else:
						cells = [versions[cell_type][cell_version](state_sz) for i in range(tot_cells)]
				else:
					if cell_version == "lstm":
						cells = [versions[cell_type][cell_version](state_sz, state_is_tuple = True) for i in range(tot_cells)]
					else:
						cells = [versions[cell_type][cell_version](state_sz) for i in range(tot_cells)]
				if rnn_dropout is not None:
					cells = [tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = 1.0 - rnn_dropout) for cell in cells]
				if cell_type == "gru":
					cell = [tf.nn.rnn_cell.MultiRNNCell(cells[a: a + num_layers]) for a in range(0, tot_cells, num_layers)]
				else:
					cell = [tf.nn.rnn_cell.MultiRNNCell(cells[a: a + num_layers], state_is_tuple = True) for a in range(0, tot_cells, num_layers)]
					
				# get variables of size (1, state_sz) for all layers both forward and backwards and cell and hidden (if lstm)
				if init_state is not None: # if it is None, we don't take care of initial state at all
					if init_state == "zeros":
						# use 0's as initial state
						init = [c.zero_state(batch_sz, tf.float32) for c in cell]
					else: # initial state is "var" or a dict specifying a specific initialization with projection
						if isinstance(init_state, dict): # the dict contains "vector" which is the raw vector, "spec" which is a list of initializations and names for projections, typically named "projX"
							projections = { k: v for k, v in init_state.items() if "vector" not in k and "spec" not in k}
							projections["raw"] = init_state["vector"]
							raw = init_state["vector"]
							specs = init_state["spec"]
							assert(len(specs) == tot_cells * (1 + int(cell_type == "lstm"))), "Mismatch between specification of cell initializations and number of cells"
						else:
							specs = ["var" for i in range(2 * tot_cells)] if cell_type == "lstm" else ["var" for i in range(tot_cells)]
						vars = 0
						init = []
						for spec in specs:
							if spec == "var":
								init += [tf.tile(tf.get_variable("init_state_" + str(vars), [1, state_sz]), [batch_sz, 1])]
								vars += 1
							elif spec == "zeros":
								init += [tf.zeros([batch_sz, state_sz])]
							elif spec == "raw":
								init += [projections["raw"]] # must fit the dimensions of [batch_sz, state_sz]
							else:
								if spec not in projections:
									if p["batch_norm"] and p["batch_norm_before_act"]:
										to_add = p["act"](tf.layers.batch_normalization(tf.layers.Dense(state_sz, activation = None)(projections["raw"]), training = training))
									else:
										to_add = tf.layers.Dense(state_sz, activation = p["act"])(projections["raw"])
									if proj_dropout is not None:
										to_add = tf.nn.dropout(to_add, rate = proj_dropout)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										to_add = tf.layers.batch_normalization(to_add, training = training)
									projections[spec] = to_add
								init += [tf.identity(projections[spec])] # must have shape [batch_sz, state_sz], always use the identity so that we may set these states to different values later without accidentally setting them all
							
						# initial states should be a tuple of tensors (batch_sz, state_sz) for each state_sz in cell.state_size (if it is a tuple, otherwise (batch, statesz
						if cell_type == "gru":
							init = [tuple(init[a: a + num_layers]) for a in range(0, tot_cells, num_layers)] # each index holds initial states for all layers
						else:
							# create one LSTMStateTuple for each pair of states (c and h states), all in all amounting to one tuple per layer per direction
							init = [tf.nn.rnn_cell.LSTMStateTuple(init[a], init[a + 1]) for a in range(0, 2 * tot_cells, 2)] # one per layer and direction
							init = [tuple(init[a: a + num_layers]) for a in range(0, tot_cells, num_layers)]						
					return (cell, init)
				else: # don't bother with initial state, just return the cell
					return cell
				
		def _rnn(o, cell, input, batch_sz, seq_lengths, cell_type, bidirectional, prefix):

			with tf.variable_scope(prefix + "_rnn"):
			
				if input is None:
					input = tf.zeros([batch_sz, o.inp_lengths_max, 1])
				if bidirectional is not None:
					if bidirectional == "stack":
						args = [cell[0]._cells, cell[1]._cells]
						init_fw = list(getattr(o, prefix + "_init_fw"))
						init_bw = list(getattr(o, prefix + "_init_bw"))				
					else:
						args = [cell[0], cell[1]]
						init_fw = getattr(o, prefix + "_init_fw")
						init_bw = getattr(o, prefix + "_init_bw")
					if bidirectional == "stack":
						print("[MAHLERNET]: adding a bidirectional stacked " + prefix + "_rnn ")
						kwargs = {"inputs": input, "initial_states_fw": init_fw, "initial_states_bw": init_bw, "dtype": tf.float32, "sequence_length": seq_lengths}
						output, final_state_fw, final_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(*args, **kwargs)
						final_state = (final_state_fw, final_state_bw)
					else:
						print("[MAHLERNET]: adding a bidirectional default " + prefix + "_rnn ")
						kwargs = {"inputs": input, "initial_state_fw": init_fw, "initial_state_bw": init_bw, "dtype": tf.float32, "sequence_length": seq_lengths}
						output, final_state = tf.nn.bidirectional_dynamic_rnn(*args, **kwargs)
																							
					# output for bidirectional is a tuple (output_fw, output_bw) where output_fw / _bw has shape [batch_sz, max_steps, state_sz]
					# finalstates is also a tuple (fw, bw) where each part holds the final state of all layers. In case of LSTM, these states are LSTMStateTuples
					summary = list(map(lambda a: a[-1].h if cell_type == "lstm" else a[-1], final_state))
					# concatenate the final hidden states from the forward and backward rnns (indexed by 0 and 1,and cell state and hidden state indexed by c and h)
					summary = tf.concat(summary, axis = 1)
				else:
					print("[MAHLERNET]: adding a unidirectional " + prefix + "_rnn ")
					args = [cell[0]]
					init = getattr(o, prefix + "_init")
					kwargs = {"inputs": input, "initial_state": init, "dtype": tf.float32, "sequence_length": seq_lengths}
					# output is an array of shape (batch_size, time_steps, state_size) containing the hidden state ONLY (not cell state) from all timesteps, 
					# final state is cell AND hidden state from last valid time step like (cell_state, hidden_state)
					# final state has shape (2, state_size) if the cell is a single cell, and (layers, 2, state_size) if it is a multi cell
					output, final_state = tf.nn.dynamic_rnn(*args, **kwargs)
					
					# output is (num_layers)
					summary = final_state[-1].h if cell_type == "lstm" else final_state[-1]

				setattr(o, prefix + "_summary", summary)
				setattr(o, prefix + "_output", output)
				setattr(o, prefix + "_final_state", final_state)

		def _seq2seq(o, cell, input, batch_sz, seq_lengths, cell_type, prefix, fn = None, aux = None):

			with tf.variable_scope(prefix + "_seq2seq"):
				init = getattr(o, prefix + "_init")
				#o.dummy = tf.Print(o.dummy, [init, input], message = "IN SEQ", summarize = 1000000)
				# HELPER FUNCTIONS
				# one-hot encodes a chosen class by the helper, simplest form of embedding to use with ScheduledEmbeddingTrainingHelper
				#emb_fctn = lambda a: tf.one_hot(a, self.params["num_pitches"]) # for ScheduledEmbedding.... since it samples a CLASS INDEX, this is the embedding so to speak
				
				# samples or chooses the argmax of the output layer then one-hot encodes it, sampling gives the same function as ScheduledEmbeddingTrainingHelper
				# and arg_max gives the same results as TrainingHelper (although TrainingHelper feeds the ground truth always)
				#out_sample_fctn = lambda a: tf.one_hot(tf.squeeze(tf.random.categorical(a, 1, dtype = tf.int32), axis = 1), self.params["num_pitches"]) # samples from classes
				#out_max_fctn = lambda a: tf.one_hot(tf.argmax(a, axis = -1), self.params["num_pitches"]) # takes MOST PROBABLE class
				
				# HELPERS - load data and returns outputs at each step of the decoding
				# plain helper, feeds targets to next input using teacher forcing, sampled ids are argmax of the output layer
				if not p["dec"]["scheduled_sampling"]:
					helper = tf.contrib.seq2seq.TrainingHelper(input, seq_lengths) 
				
				# scheduled sampling where output are treated as logits and a sample is drawn from this distribution, embedding layer encodes chosen class
				# sample ids are -1 for outputs where no sampling was made, otherwise the sampled output
				#helper = tf.contrib.seq2seq.ScheduledEmbeddingTrainingHelper(self.ops.decoder_input, seql, emb_fctn, 0.3)
				
				# scheduled sampling, more versatile than previous, function for sampling and adding to chosen inputs can be given by the user
				# sample ids are False for outputs where no sampling took place, True otherwise
				else:
					rate = tf.cast(p["dec"]["scheduled_sampling_rate"], tf.float32)
					if p["dec"]["scheduled_sampling_scheme"] == "sigmoid":
						prob = (1 - (rate / (rate + tf.exp(tf.divide(o.global_step, rate))))) ** 2
					else: # linear, max(e, k - ci) where e is the minimum truth and k is the maximum truth and c is a linerear coefficient factor to global steps (i)
						prob = 1 - tf.math.maximum(p["dec"]["scheduled_sampling_min_truth"], tf.constant(1.0, dtype = tf.float32) - tf.multiply(p["dec"]["scheduled_sampling_rate"], o.global_step))
					helper = tf.contrib.seq2seq.ScheduledOutputTrainingHelper(input, seq_lengths, prob, next_inputs_fn = fn, auxiliary_inputs = aux)
				
				kwargs = {"maximum_iterations": o.inp_lengths_max, "impute_finished": True}				
				decoder = tf.contrib.seq2seq.BasicDecoder(cell[0], helper, init, None)
				
				''' 
				output from dynamic_decode are:
				(decoder_outputs, decoder_final_states, seq_lengths)
				... where decoder_outputs has porperties .rnn_output and .sample_id
				
				decoder_output.rnn_output is exactly the output from the corresponding dynamic_rnn with the same sequence lengths run through a linear layer
				decoder_output.sample_id is exactly the argmax of the output of a dynamic_rnn with same sequence lengths run through a linear layer
				decoder_final_states are exactly the final states of the corresponding dynamic_rnn
				seq_lengths is a list of sequence length with the same content as the input sequence lengths if max_iterations was set equal to or higher
				than the longest sequence
				'''
				decoder_output_tuple, decoder_final_state, sequences = tf.contrib.seq2seq.dynamic_decode(decoder, **kwargs)
				
				setattr(o, prefix + "_final_state", decoder_final_state)
				setattr(o, prefix + "_output", decoder_output_tuple.rnn_output)
				setattr(o, prefix + "_preds", decoder_output_tuple.sample_id) # sample ids are arg max of rnn_output with regular traininghelper
				
		def _encoder_component(o, p):
			summaries = []
			for key in ["inp", "ctx"]:
				if p["model"][key]:
					print("[MAHLERNET]: adding " + key + " encoder")
					data = getattr(o, key)
					lengths = getattr(o, key + "_lengths")
					key = key + "_enc"
					bidirectional_type = p[key]["bidirectional_type"]
					dropout = getattr(o, key + "_dropout") if p[key]["dropout"] > 0.0 else None
					cell, init = _rnn_setup(o.batch_sz, p[key]["type"], p[key]["version"], p["cell_versions"], True, p[key]["sz_state"], 
												p[key]["num_layers"], dropout, None, p[key]["init"], key, o.training)
					setattr(o, key + "_init_fw", init[0])
					setattr(o, key + "_init_bw", init[1])
					_rnn(o, cell, data, o.batch_sz, lengths, p[key]["type"], bidirectional_type, key)
					summaries += [getattr(o, key + "_summary")]
			# now either (batch_sz, 2 * inp_state + 2 * ctx_state) or (batch_sz, 2 * inp_state)
			o.vae_input = tf.concat(summaries, axis = 1) if len(summaries) > 1 else summaries[0]
		
		def _vae(o, p, w):
			# vae input
			with tf.variable_scope("vae"):
				vae_inp_sz = 2 * (p["inp_enc"]["sz_state"] * int(p["model"]["inp"]) + p["ctx_enc"]["sz_state"] * int(p["model"]["ctx"]))
				z_dim = p["vae"]["z_dim"]
				w["vae"]['mu_W'] = tf.get_variable("vae_mu_W", (vae_inp_sz, z_dim), initializer=tf.random_normal_initializer(stddev=0.001))
				w["vae"]["mu_b"] = tf.get_variable("vae_mu_bias", (z_dim), initializer=tf.constant_initializer(0.0))
				o.mu = tf.matmul(o.vae_input, w["vae"]["mu_W"]) + w["vae"]["mu_b"]
				
				w["vae"]["log_sigma_sq_W"] = tf.get_variable("vae_log_sigma_sq_W", (vae_inp_sz, z_dim), initializer=tf.random_normal_initializer(stddev=0.001))
				w["vae"]["log_sigma_sq_b"] = tf.get_variable("vae_log_sigma_sq_bias", (z_dim), initializer=tf.constant_initializer(0.0))			
				o.log_sigma_sq = tf.nn.softplus(tf.matmul(o.vae_input, w["vae"]["log_sigma_sq_W"]) + w["vae"]["log_sigma_sq_b"]) + 1e-10
				
				# sample a vector
				if p["use_randomness"]:
					eps = tf.random_normal((o.batch_sz, z_dim), 0, 1, dtype=tf.float32)
				else:
					eps = tf.random_normal((o.batch_sz, z_dim), 0, 1, dtype=tf.float32, seed = 1)
				o.z = tf.add(o.mu, tf.multiply(tf.sqrt(tf.exp(o.log_sigma_sq)), eps), name = "sampled_z")	
			
		def _decoder_ctx_setup(o, p):
			
			with tf.variable_scope("init_decoder_context"):
				# decoder context
				o.num_dec_ctx_features = 0
				o.dec_init_raw = None
				dec_ctx = []
				if p["model"]["ctx"] and p["dec"]["feed"]["ctx"]["use"]:
					o.num_dec_ctx_features += 2 * p["ctx_enc"]["sz_state"]
					dec_ctx += [o.ctx_enc_summary]
				if p["model"]["vae"]:
					if p["dec"]["init"] == "z": # set up the initial state as a projection from the latent vector
						layer_states = 2 if p["dec"]["type"] == "lstm" else 1
						tot_states = layer_states * p["dec"]["num_layers"]
						if p["dec"]["model"] == "balstm":
							# ONLY set up the initial state for the balstm if we want a custom setup, that is, use special projections for its initial state
							if p["dec"]["balstm"]["init"] == "diff": # use different projections of latent space for all rnn instances
								if p["dec"]["layer_strategy"] == "diff":
									if p["batch_norm"] and p["batch_norm_before_act"]:
										o.dec_init = [p["act"](tf.layers.batch_normalization(tf.layers.Dense(p["dec"]["sz_state"] * p["num_pitches"], activation = None)(o.z), training = o.training)) for _ in range(tot_states)] # (batch, num_p * state_sz)
									else:
										o.dec_init = [tf.layers.Dense(p["dec"]["sz_state"] * p["num_pitches"], activation = p["act"])(o.z) for _ in range(tot_states)] # (batch, num_p * state_sz)
									if p["vae"]["dropout"] > 0.0:
										o.dec_init = [tf.nn.dropout(init_layer, rate = o.vae_dropout) for init_layer in o.dec_init] # (batch, num_p * state_sz)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										o.dec_init = [tf.layers.batch_normalization(init_layer, training = o.training) for init_layer in o.dec_init] # (batch, num_p * state_sz)
									o.dec_init = [tf.reshape(in_st, [o.balstm_batch_sz, p["dec"]["sz_state"]]) for in_st in o.dec_init]
								else: # same setup for every layer
									if p["batch_norm"] and p["batch_norm_before_act"]:
										o.dec_init = [p["act"](tf.layers.batch_normalization(tf.layers.Dense(p["dec"]["sz_state"] * p["num_pitches"], activation = None)(o.z), training = o.training)) for _ in range(layer_states)] # (batch, num_p * state_sz)
									else:
										o.dec_init = [tf.layers.Dense(p["dec"]["sz_state"] * p["num_pitches"], activation = p["act"])(o.z) for _ in range(layer_states)] # (batch, num_p * state_sz)										
									if p["vae"]["dropout"] > 0.0:
										o.dec_init = [tf.nn.dropout(init_layer, rate = o.vae_dropout) for init_layer in o.dec_init] # (batch, num_p * state_sz)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										o.dec_init = [tf.layers.batch_normalization(init_layer, training = o.training) for init_layer in o.dec_init] # (batch, num_p * state_sz)
									o.dec_init = [tf.reshape(in_st, [o.balstm_batch_sz, p["dec"]["sz_state"]]) for in_st in o.dec_init]
									o.dec_init = [[tf.identity(in_st) for in_st in o.dec_init] for _ in range(p["dec"]["num_layers"])] # all layers gets the same initial states
							else: # use the same projection from latent space on all rnn instances
								if p["dec"]["layer_strategy"] == "diff": # different between layers however
									if p["batch_norm"] and p["batch_norm_before_act"]:
										o.dec_init = [p["act"](tf.layers.batch_normalization(tf.layers.Dense(p["dec"]["sz_state"], activation = None)(o.z), training = o.training)) for _ in range(p["dec"]["num_layers"])] # (batch, sz_state)
									else:
										o.dec_init = [tf.layers.Dense(p["dec"]["sz_state"], activation = p["act"])(o.z) for _ in range(p["dec"]["num_layers"])] # (batch, sz_state)
									if p["vae"]["dropout"] > 0.0:
										o.dec_init = [tf.nn.dropout(init_layer, rate = o.vae_dropout) for init_layer in o.dec_init] # (batch, sz_state)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										o.dec_init = [tf.layers.batch_normalization(init_layer, training = o.training) for init_layer in o.dec_init] # (batch, sz_state)
									o.dec_init = [tf.tile(in_st, [1, p["num_pitches"]]) for in_st in o.dec_init]
									o.dec_init = [tf.reshape(in_st, [o.balstm_batch_sz, p["dec"]["sz_state"]]) for in_st in o.dec_init]
									o.dec_init = [tf.identity(in_st) for in_st in o.dec_init]
								else: # same for all instances and for all layers
									if p["batch_norm"] and p["batch_norm_before_act"]:
										o.dec_init_raw = p["act"](tf.layers.batch_normalization(tf.layers.Dense(p["dec"]["sz_state"], activation = None, name = "dec_init_raw")(o.z), training = o.training)) # (batch, sz_state)
									else:
										o.dec_init_raw = tf.layers.Dense(p["dec"]["sz_state"], activation = p["act"], name = "dec_init_raw")(o.z) # (batch, sz_state)
									if p["vae"]["dropout"] > 0.0:	
										o.dec_init_raw = tf.nn.dropout(o.dec_init_raw, rate = o.vae_dropout)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										o.dec_init_raw = tf.layers.batch_normalization(o.dec_init_raw, training = o.training)
									o.dec_init = tf.tile(o.dec_init_raw, [1, p["num_pitches"]]) # (batch, num_pitches * state_sz)
									o.dec_init = tf.reshape(o.dec_init, [o.balstm_batch_sz, p["dec"]["sz_state"]]) # (batch * num_pitches, state_sz)
									o.dec_init = [tf.identity(o.dec_init) for _ in range(tot_states)]		
							if p["dec"]["type"] == "lstm":
								# create one LSTMStateTuple for each pair of states (c and h states), all in all amounting to one tuple per layer
								o.dec_init = [tf.nn.rnn_cell.LSTMStateTuple(o.dec_init[a], o.dec_init[a + 1]) for a in range(0, tot_states, 2)] # one per layer
							o.dec_init = [tuple(o.dec_init)] # input init state is supposed to be a tuple of tensors and the function expects a list

					if p["dec"]["feed"]["z"]["use"]: # use the latent vector as part of the input at every time step during decoding
						if p["dec"]["feed"]["z"]["strategy"] == "raw":
							o.num_dec_ctx_features += p["vae"]["z_dim"]
							dec_ctx += [o.z]
						else:
							if o.dec_init_raw is None: # either we are NOT using the BALSTM OR we are using a strategy with the BALSTM that makes it impossible to use a single projection of z as a feed during decoding
								# if we use a regular lstm, no initial states have been created and if layer strategy is same, the one created below will be reused,
								# otherwise, new projections will be created for each layer.
								# under these circumstances, we need to add a new layer for projections of z since either different projections or none has been used earlier
								if p["dec"]["init"] != "z" or p["dec"]["model"] == "balstm" or (p["dec"]["model"] == "lstm" and p["dec"]["layer_strategy"] == "diff"): 
									# this is the scenario where there might be other projections of z but not compatible with the use as input feed during decoding. These scenarios include the use of BALSTM with "diff" as either
									# layer_strategy or initialization or regular LSTM with "diff" as layer strategy (none of these cases allows the use of a single vector to feed during decoding since thee is no way to pick a single
									# and we can't feed them all, thus we create a new one using the specification under "feed". The third case is where not z is used as initialization at all, then we don't have to adhere to the
									# state size of the decoder
									if p["batch_norm"] and p["batch_norm_before_act"]:
										o.dec_init_raw = p["act"](tf.layers.batch_normalization(tf.layers.Dense(p["dec"]["feed"]["z"]["sz"], activation = None, name = "dec_init_raw")(o.z), training = o.training)) # (batch, sz_state)
									else:
										o.dec_init_raw = tf.layers.Dense(p["dec"]["feed"]["z"]["sz"], activation = p["act"], name = "dec_init_raw")(o.z) # (batch, sz_state)
									if p["vae"]["dropout"] > 0.0:
										o.dec_init_raw = tf.dropout(o.dec_init_raw, rate = o.vae_dropout) # (batch, sz_state)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										o.dec_init_raw = tf.layers.batch_normalization(o.dec_init_raw, training = o.training)
									o.num_dec_ctx_features += p["dec"]["feed"]["z"]["sz"]
								else: 
									# in this scenario, the BALSTM is not used at all (since then, either dec_init_raw was set or a separate projection was created above), thus we are using the regular LSTM model with z_dim
									# initialization and layer strategy same, thus we must make the projection so that it matches the size of the decoder state so that we may use the same for feeding and initialization.
									if p["batch_norm"] and p["batch_norm_before_act"]:
										o.dec_init_raw = p["act"](tf.layers.batch_normalization(tf.layers.Dense(p["dec"]["sz_state"], activation = None, name = "dec_init_raw")(o.z), training = o.training)) # (batch, sz_state)
									else:
										o.dec_init_raw = tf.layers.Dense(p["dec"]["sz_state"], activation = p["act"], name = "dec_init_raw")(o.z) # (batch, sz_state)
									if p["vae"]["dropout"] > 0.0:
										o.dec_init_raw = tf.nn.dropout(o.dec_init_raw, rate = o.vae_dropout) # (batch, sz_state)
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										o.dec_init_raw = tf.layers.batch_normalization(o.dec_init_raw, training = o.training)
									o.num_dec_ctx_features += p["dec"]["sz_state"]
							else: # if dec_init_raw was created, it was due to the use of BALSTM with layer and init strategy "same" and thus it has the size of the decoder state
								o.num_dec_ctx_features += p["dec"]["sz_state"]
							dec_ctx += [o.dec_init_raw]
				if len(dec_ctx) > 1:
					o.init_dec_ctx = tf.concat(dec_ctx, axis = 1) # (batch, z_dim + 2 * ctx_encoder state)
				elif len(dec_ctx) > 0:
					o.init_dec_ctx = dec_ctx[0]
				else:
					o.init_dec_ctx = None
										
		def _balstm_decoder(o, p, w):
			# it is assumed that if we reach here, pitch is modelled in the network
			
			with tf.variable_scope("balstm_decoder"):
			
				# 1D convolution along the feature axis of every time step
				convs = []
				feats = 0
				
				if p["modelling_properties"]["pitch"]["include"]:
					feats += p["dec"]["balstm"]["num_filters"]
					# Since 1D convolution treats the features as channels, we can't achieve the kind of padding desired, convert to 2D and do convolution there
					w["conv"]["p_W"] = tf.get_variable("conv_p_W", (1, p["dec"]["balstm"]["sz_conv"], 1, p["dec"]["balstm"]["num_filters"])) # features will be treated as width after reshape to 2D
					w["conv"]["p_b"] = tf.get_variable("conv_p_bias", [p["dec"]["balstm"]["num_filters"]], initializer=tf.constant_initializer(0.0))
					conv_p = tf.reshape(o.x_p, (o.batch_sz, o.inp_lengths_max, p["num_pitches"], 1)) # reshape to 2D
					conv_p = tf.nn.conv2d(conv_p, w["conv"]["p_W"], strides = [1, 1, 1, 1], padding = "SAME")
					conv_p = tf.add(conv_p, w["conv"]["p_b"])
					if p["batch_norm"] and p["batch_norm_before_act"]:
						conv_p = tf.layers.batch_normalization(conv_p, training = o.training)
					o.conv_p_output = p["act"](conv_p) # is now (batch_sz, num_steps, num_pitches, filters)
					convs += [o.conv_p_output]
					
				if p["modelling_properties"]["active_pitches"]["include"]:
					feats += p["dec"]["balstm"]["num_filters"]
					# Since 1D convolution treats the features as channels, we can't achieve the kind of padding desired, convert to 2D and do convolution there
					w["conv"]["a_W"] = tf.get_variable("conv_a_W", (1, p["dec"]["balstm"]["sz_conv"], 1, p["dec"]["balstm"]["num_filters"])) # features will be treated as width after reshape to 2D
					w["conv"]["a_b"] = tf.get_variable("conv_a_bias", [p["dec"]["balstm"]["num_filters"]], initializer=tf.constant_initializer(0.0))
					conv_a = tf.reshape(o.x_ap, (o.batch_sz, o.inp_lengths_max, p["num_pitches"], 1)) # reshape to 2D
					conv_a = tf.nn.conv2d(conv_a, w["conv"]["a_W"], strides = [1, 1, 1, 1], padding = "SAME")
					conv_a = tf.add(conv_a, w["conv"]["a_b"])
					if p["batch_norm"] and p["batch_norm_before_act"]:
						conv_a = tf.layers.batch_normalization(conv_a, training = o.training)
					o.conv_a_output = p["act"](conv_a) # is now (batch_sz, num_steps, num_pitches, 1)			
					convs += [o.conv_a_output]
					
				o.conv_output = tf.concat(convs, axis = -1) if len(convs) > 1 else convs[0]
				if p["dec"]["balstm"]["conv_dropout"] > 0.0:
					o.conv_output = tf.nn.dropout(o.conv_output, rate = o.dec_balstm_conv_dropout)
				if p["batch_norm"] and not p["batch_norm_before_act"]:
					o.conv_output = tf.layers.batch_normalization(o.conv_output, training = o.training)
				# decoder_context
				if o.init_dec_ctx is not None:
					o.dec_ctx = tf.tile(o.init_dec_ctx, [1, o.inp_lengths_max * p["num_pitches"]]) # repeat for each batch the ctx and x stes * pitches times
					o.dec_ctx = tf.reshape(o.dec_ctx, [o.balstm_batch_sz, o.inp_lengths_max, o.num_dec_ctx_features])
				else:
					o.dec_ctx = None
					
				if o.x_rest is not None:
					# in the x_rest vector, (batch, steps, features), for each sequence, repeat the sequence as a whole num_pitches time since each sequence is
					# about to be turned into num_pitches sequences of with features size 1
					repeated_x_rest = tf.tile(o.x_rest, [1, p["num_pitches"], 1]) # is now (batch, num_pitches * steps, num_features - num_pitches)
					repeated_x_rest = tf.reshape(repeated_x_rest, [o.balstm_batch_sz, o.inp_lengths_max, p["num_xr_features"]])
					o.dec_ctx = tf.concat((repeated_x_rest, o.dec_ctx), axis = 2) if o.dec_ctx is not None else repeated_x_rest
					
				if p["dec"]["balstm"]["add_pitch"]:
					pitch = tf.range(p["num_pitches"], dtype = tf.int32)
					pitch = tf.reshape(pitch, [p["num_pitches"], 1])
					# in the balstm we have (batch * pitch, steps, feat) so the SAME pitch must be provided for all timesteps, inp_lengths_max in a row and repeated for each sequence in the batch
					pitch = tf.tile(pitch, [o.batch_sz, o.inp_lengths_max])
					pitch = tf.one_hot(pitch, depth = p["num_pitches"], dtype = tf.float32)
					o.dec_ctx = tf.concat([o.dec_ctx, pitch], axis = 2) if o.dec_ctx is not None else pitch
			
				# decoder input
				o.dec_inp = tf.transpose(o.conv_output, [0, 2, 1, 3]) # switch steps and pitch convolutions so we get (batch, pitch, steps, 1-2)
				o.dec_inp = tf.reshape(o.dec_inp, [o.balstm_batch_sz, o.inp_lengths_max, feats]) # now (batch * pitch, steps, 1-2), all sequences are of features size 1-2
				o.dec_inp = tf.concat([o.dec_inp, o.dec_ctx], axis = 2) if o.dec_ctx is not None else o.dec_inp
				rnn_dropout = o.dec_dropout if p["dec"]["dropout"] > 0.0 else None
				proj_dropout = o.vae_dropout if p["vae"]["dropout"] > 0.0 else None
				if p["model"]["vae"] and p["dec"]["init"] == "z":
					cell = _rnn_setup(o.balstm_batch_sz, p["dec"]["type"], p["dec"]["version"], p["cell_versions"], False, p["dec"]["sz_state"], p["dec"]["num_layers"], rnn_dropout, proj_dropout, None, "dec", o.training)
					init = o.dec_init
				else:
					cell, init = _rnn_setup(o.balstm_batch_sz, p["dec"]["type"], p["dec"]["version"], p["cell_versions"], False, p["dec"]["sz_state"], p["dec"]["num_layers"], rnn_dropout, proj_dropout, p["dec"]["init"], "dec", o.training)
				inits = []
				for i in range(p["dec"]["num_layers"]):
					if p["dec"]["type"] == "lstm":
						setattr(o, "dec_init_h_" + str(i), init[0][i].h)
						setattr(o, "dec_init_c_" + str(i), init[0][i].c)
						inits += [tf.nn.rnn_cell.LSTMStateTuple(getattr(o, "dec_init_c_" + str(i)), getattr(o, "dec_init_h_" + str(i)))]
					else:
						setattr(o, "dec_init_" + str(i), init[0][i])
						inits += [getattr(o, "dec_init_" + str(i))]
				o.dec_init = tuple(inits)					
				if p["dec"]["framework"] == "seq2seq":
					print("[MAHLERNET]: using seq2seq framework for decoding")
					_seq2seq(o, cell, o.dec_inp, o.balstm_batch_sz, o.balstm_lengths, p["dec"]["type"], "dec")
				else:
					print("[MAHLERNET]: using regular rnn framework for decoding")
					o.dec_inp = tf.concat([o.dec_inp, o.dec_ctx], axis = 2) if o.dec_ctx is not None else o.dec_inp # now each sequence has multiple features
					_rnn(o, cell, o.dec_inp, o.balstm_batch_sz, o.balstm_lengths, p["dec"]["type"], None, "dec")
	
		def _lstm_decoder(o, p, w, fn = None):
			
			with tf.variable_scope("lstm_decoder"):
				
				# decoder context
				if o.init_dec_ctx is not None:
					o.dec_ctx = tf.expand_dims(o.init_dec_ctx, axis = 1) # (batch, 1, num_dec_ctx_features)
					o.dec_ctx = tf.tile(o.dec_ctx, [1, o.inp_lengths_max, 1]) # tile the context so that it is available to all time steps
				else:
					o.dec_ctx = None
				if p["model"]["vae"] and p["dec"]["init"] == "z":
					if p["dec"]["layer_strategy"] == "same": 
						init_state = {"vector": o.z, "spec": ["proj1"] * p["dec"]["num_layers"] * (1 + int(p["dec"]["type"] == "lstm"))}
						if o.dec_init_raw is not None: # is none when z feed is not used
							init_state["proj1"] = o.dec_init_raw # reuse the one we made in decoder ctx setup
					else: # create new projections, no matter what
						init_state = {"vector": o.z, "spec": ["proj" + str(i) for i in range (p["dec"]["num_layers"] * (1 + int(p["dec"]["type"] == "lstm")))]}
				else:
					init_state = p["dec"]["init"]
				rnn_dropout = o.dec_dropout if p["dec"]["dropout"] > 0.0 else None
				proj_dropout = o.vae_dropout if p["vae"]["dropout"] > 0.0 else None					
				cell, init = _rnn_setup(o.batch_sz, p["dec"]["type"], p["dec"]["version"], p["cell_versions"], False, p["dec"]["sz_state"], p["dec"]["num_layers"], rnn_dropout, proj_dropout, init_state, "decoder", o.training)
				inits = []
				for i in range(p["dec"]["num_layers"]):
					if p["dec"]["type"] == "lstm":
						setattr(o, "dec_init_h_" + str(i), init[0][i].h)
						setattr(o, "dec_init_c_" + str(i), init[0][i].c)
						inits += [tf.nn.rnn_cell.LSTMStateTuple(getattr(o, "dec_init_c_" + str(i)), getattr(o, "dec_init_h_" + str(i)))]
					else:
						setattr(o, "dec_init_" + str(i), init[0][i])
						inits += [getattr(o, "dec_init_" + str(i))]																	
				o.dec_init = tuple(inits)
				if p["dec"]["framework"] == "seq2seq":
					print("[MAHLERNET]: using seq2seq framework for decoding")
					if p["dec"]["scheduled_sampling"]:
						print("[MAHLERNET]: using scheduled sampling with inverse sigmoid schedule and rate", p["dec"]["scheduled_sampling_rate"])
						_seq2seq(o, cell, o.x, o.batch_sz, o.inp_lengths, p["dec"]["type"], "dec", fn, o.dec_ctx)
					else:
						print("[MAHLERNET]: not using scheduled sampling")
						o.dec_inp = tf.concat([o.x, o.dec_ctx], axis = 2) if o.dec_ctx is not None else o.x
						_seq2seq(o, cell, o.dec_inp, o.batch_sz, o.inp_lengths, p["dec"]["type"], "dec")
				else:
					print("[MAHLERNET]: using rnn framework")
					o.dec_inp = tf.concat([o.x, o.dec_ctx], axis = 2) if o.dec_ctx is not None else o.x
					_rnn(o, cell, o.dec_inp, o.batch_sz, o.inp_lengths, p["dec"]["type"], None, "dec")
			
		def _output_layer(ops, p, input, multiclass, mask, sample, temp, layers, num_out_feat, prefix, full):
		
			logits = input
			for num, layer in enumerate(layers):
				if p["batch_norm"] and p["batch_norm_before_act"]:
					act = layer.activation
					layer.activation = None
					logits = act(tf.layers.batch_normalization(layer(logits), training = ops.training, name = prefix + "_BN"))
					layer.activation = act
				else:
					logits = layer(logits)
				if num < len(layers) - 1: # no dropout or post activation BN on last output layer
					if p["modelling_properties"][prefix]["dropout"] > 0.0: 
						logits = tf.nn.dropout(logits, rate = p["modelling_properties"][prefix]["dropout"])
				if p["batch_norm"] and not p["batch_norm_before_act"]:
					logits = tf.layers.batch_normalization(logits, training = ops.training, name = prefix + "_BN")
			logits = tf.divide(logits, temp)
			logits = tf.reshape(logits, [-1, num_out_feat])
			
			
			if multiclass: # assume each index is a binary variable non-exclusive class (several classes can be chosen independently)
				if full:
					probs = tf.nn.sigmoid(logits)
				sample_fn = tf.cast(tf.distributions.Bernoulli(logits = logits).sample(), tf.float32)
				max_fn = tf.cast(tf.greater(logits, tf.constant(0.0, dtype = tf.float32)), tf.float32) # same as y = sigmoid > 0.5
				preds = tf.cond(sample, lambda: sample_fn, lambda: max_fn)
			else: # assume all indices are mutually exclusive dependent classes (only one class can be right)
				if full:
					probs = tf.nn.softmax(logits)
				sample_fn = lambda: tf.squeeze(tf.random.categorical(logits, 1, dtype = tf.int32), axis = -1) # doesn't remove a dim by default
				max_fn = lambda: tf.argmax(logits, axis = -1, output_type = tf.int32)
				
				preds = tf.cond(sample, sample_fn, max_fn)
				preds = tf.one_hot(preds, num_out_feat)
			if full:
				probs = tf.reshape(probs, [o.batch_sz, o.inp_lengths_max, num_out_feat])
				probs = tf.multiply(probs, mask)	
				preds = tf.reshape(preds, [o.batch_sz, o.inp_lengths_max, num_out_feat])
				preds = tf.multiply(preds, mask)		
				logits = tf.reshape(logits, [o.batch_sz, o.inp_lengths_max, num_out_feat])
				logits = tf.multiply(logits, mask)		
				return logits, probs, preds
			else:
				preds = tf.reshape(preds, [-1, num_out_feat])
				return preds

		'''
		Takes necessary contextual parameters stored in o, for operations, and p, for parameters and a flag indicating whether the returned function should be
		used during scheduled sampling or not. When the function is used as a post processor (flag is True), then certain values may be stored in the o object
		which are otherwise not stored since results during scheduled sampling is, intuitively, less final, and are only meant to be used as the next input
		to the decoding rnn. Because of this, when the returned function is used for scheduled sampling, local variables are used whereas when used as a post
		processor, variables are stored in ops. To facilitate this and reuse code, the variables concerned during both scheduled sampling and post processing
		are stored as local variables and then, if the function is used as a post processor, attached to the o object before returning. This concerns the
		dec_output, y and y_d, y_p and y_i variables which in turn depends on respective preds and logits variable. These groups are then used as local variable.
		
		During scheduled sampling, the decoding rnn uses the output from the rnn to determine the next time step input, as opposed to using teacher forced targets
		instead. Because offset is the property that moves time forward, this property cannot be modelled with scheduled sampling. To see why, sampling a new offset
		would affect both the target beats vector and render subsequent steps invalid. Sampling longer offsets would also result in too long inputs depending on what
		default input length is used. A second notice is that scheduled sampling may not be used with the BALSTM since the BALSTM models individual pitches
		in the decoding rnn and to determine the next input based on the output of a input sequence, one would have to sample all the adjacent pitches which
		would call for a custom helper function with high complexity. Since scheduled sampling cannot (this is assumed to have been checked) be used with the
		BALSTM or with offset, some of the calculatations taking part with respect to these will attach variables to the o object straight away since these will
		only be called, if ever, when the returned function is used as a post processor after decoding is complete.
		
		Parameters
		----------
		
		o: object
			An object with Tensorflow operations attached to it
			
		p: dict
			A dictionary with param
			
		post: bool
			A boolean indicating whether the returned function is going to be used for scheduled sampling or post processing. The returned function returns
			the resulting vectors if used as a scheduled sampling function, otherwise, results are attached to the input o object.
			
		Yields
		------
		
		function
			A function that can be used with decoder output (complete or for a time step) to accomplish post processing with dense layers.
			
		'''				
		def _output_fn(o, p, post):
		
			# below function will be returned BOTH for proper post-processing after the decoder rnn is done but ALSO DURING decoding if scheduled sampling is
			# used. When used after the decoder, post will be True, otherwise pos will be False. Some properties on o will not be set DURING decoding and
			# we have to make sure that control cannot reach those statements when this function is used during decoding.
			#
			# This function takes the output of an rnn (dec_output) which is fed as an argument to the returned below function. Noweher, not during decoding
			# or post-decoding will this function attach the growing dec_output object to o since it is useless after all outputs have been run. However, 
			# when used in post-decoding, it will be attached to the o object prior to this function but taken from it and added as an argument. This is not
			# the case when it is passed during decoding as an argument only. As stated, when used in post-decoding, the output results will be attached to
			# the o object so that the losses and metrics can be taken against it. In both decoding and post-decoding, dec_output is passed through the pipeline
			# of outputs and calculated outputs are added to condition the remaining outputs. The y variable stores the next input vector and will only be used
			# when the function is used in decoding since there is no next input to be calculated in post-decoding. To allow for the dec_output variable to work in
			# both decoding and post-decoding, every output result is attached to local variables on the form y_X and then added to dec_output. The main idea
			# driving these design choices is that we don't want to attach anything to the o object when this function is used during decoding but still be able
			# to use the same pipeline in both contexts.						   
			
			def output(dec_output):
				if post:
					sample_from_predictions = o.sample_from_predictions
				else:
					sample_from_predictions = tf.constant(p["dec"]["scheduled_sampling_mode"] == "sample", dtype = tf.bool)
				with tf.variable_scope("outputs"):
						
					y = None
					y_features = 0
					if p["dec"]["model"] == "balstm": # can't be used with scheduled sampling so all only happens if post == True
						o.pitch_dec_output = dec_output # save the decoder output where each pitch has its own sequence
						# reshape from treating every pitch as a sequence of its own to the original representation for use to predict offset, duration and instrument
						reshaped_dec_output = tf.reshape(dec_output, [o.batch_sz, p["num_pitches"], o.inp_lengths_max, p["dec"]["sz_state"]])
						dec_output = tf.transpose(reshaped_dec_output, [0, 2, 1, 3]) # now we have (batch_sz, steps, pitches, decoder_sz)
						dec_output = tf.reshape(dec_output, [o.batch_sz, o.inp_lengths_max, p["dec"]["sz_state"] * p["num_pitches"]])
						
					# input is the processed output from the rnn and has shape (batch, steps, features)
					###################################################################################################################################################
					### OFFSET for the event we are predicting ########################################################################################################
					###################################################################################################################################################
					if p["modelling_properties"]["offset"]["include"]: # can now be used with scheduled sampling
					
						with tf.variable_scope(p["scopes"]["offset"]): #tf.variable_scope("offset_output"):
							args = [o, p, dec_output]
							kwargs = {"multiclass": p["modelling_properties"]["offset"]["multiclass"], "mask": o.inp_lengths_mask, "sample": sample_from_predictions, 
									"temp": o.offset_temp, "layers": p["output_layers"]["offset"], "num_out_feat": p["num_durations"], "prefix": "offset", "full": post}
							
							if post:
								o.y_o_logits, o.y_o_probs, o.y_o_preds = _output_layer(*args, **kwargs)
								o.y_o = y_o = tf.identity(o.y_o_preds, name = "y_o")
								if p["training_output"]["dists"]:
									o.y_o_guess_dist = tf.reduce_sum(tf.cast(o.y_o_preds, tf.int64), axis = [0, 1]) # (features) with stats over which features have been guessed most
									o.y_o_target_dist = tf.reduce_sum(tf.cast(o.y_o, tf.int64), axis = [0, 1])
									o.y_o_diff_dist = tf.abs(tf.subtract(o.y_o_guess_dist, o.y_o_target_dist))
									o.y_o_guess_tot_dist = o.y_o_guess_tot_dist_var.assign(tf.add(o.y_o_guess_tot_dist_var, o.y_o_guess_dist))
									y_features += p["num_durations"] # only every needed by the balstm which only is available in post mode since scheduled sampling is unapplicable to it
							else:
								y_o_preds = _output_layer(*args, **kwargs)
								y_o = y_o_preds
								y = y_o
							dec_output = tf.concat([dec_output, y_o], axis = -1)		

							# when targets (and beats) has not been supplied when running (as in training) we need to derive the beats information directly from the offset results
							actual_offset = tf.multiply(y_o, o.durations) # actual_offset now holds a vector of possibly several durations
							actual_offset = tf.reduce_sum(actual_offset, reduction_indices = -1, keepdims = True) # (batch, steps, 1)
							o.time = tf.add(o.last_time, actual_offset, name = "out_time") # keep track of current time
															
							###################################################################################################################################################
							### BEATS information regarding where in the current bar the event to come takes place ############################################################
							###################################################################################################################################################		
		
							if p["modelling_properties"]["beats"]["include"]: # not possible with scheduled sampling so happens if post == True
								bar_offset = tf.mod(o.time, o.bar) # time in current bar
								o.y_b_preds = tf.cast(tf.equal(bar_offset, o.beats), tf.float32) # results in a beat vector where alignment on a beat is marked by 1
								o.y_b = y_b = tf.identity(o.y_b_preds, name = "y_b")
								dec_output = tf.concat([dec_output, y_b], axis = -1)
								y_features += p["num_beats"]
							
					###################################################################################################################################################
					### DURATION for the event we are predicting ##################################################################################################
					###################################################################################################################################################
					if p["modelling_properties"]["duration"]["include"]:
					
						with tf.variable_scope(p["scopes"]["duration"]): #tf.variable_scope("duration_output"):
							
							y_d_logits = dec_output
							for num, layer in enumerate(p["output_layers"]["duration"]):
								if p["batch_norm"] and p["batch_norm_before_act"]:
									act = layer.activation
									layer.activation = None
									y_d_logits = act(tf.layers.batch_normalization(layer(y_d_logits), training = o.training, name = "duration_BN"))
									layer.activation = act
								else:
									y_d_logits = layer(y_d_logits)
								if num < len(p["output_layers"]["duration"]) - 1: # no dropout or post activation BN on last output layer
									if p["modelling_properties"]["duration"]["dropout"] > 0.0:
										y_d_logits = tf.nn.dropout(y_d_logits, rate = p["modelling_properties"]["duration"]["dropout"])
								if p["batch_norm"] and not p["batch_norm_before_act"]:
									y_d_logits = tf.layers.batch_normalization(y_d_logits, training = o.training, name = "duration_BN")

							if p["modelling_properties"]["offset"]["include"] and p["modelling_properties"]["duration"]["next_step_reduction"]:
								# adjust the output logits if the current offset is non-zero since a zero duration is only allowed with a zero offset
								if p["modelling_properties"]["offset"]["multiclass"]:
									# 0 duration after 0 offset is forbidden so whenever offset is 0, we make it impossible to predict 0 duration
									next_timestep = tf.count_nonzero(y_o, axis = -1, dtype = tf.int32) # (batch, steps) with 0 if time offset from previous was 0, otherwise 1
								else:
									next_timestep = tf.argmax(y_o, axis = -1, output_type = tf.int32) # (batch, steps) with 0 if time offset from previous was 0, otherwise 1
								next_timestep = tf.equal(next_timestep, 0) # (batch, steps) with True if no time from previous has passed
								next_timestep = tf.cast(next_timestep, dtype = tf.float32) # (batch, steps) with 1 if no time from previous has passed, otherwise 0
								next_timestep = tf.expand_dims(next_timestep, axis = -1) # output is (batch, step, 1)
								weight_mask = tf.multiply(next_timestep, tf.one_hot(0, depth = p["num_durations"], dtype = tf.float32))
								mask = tf.subtract(tf.constant(1.0), weight_mask) # (batch, steps, features) with 1 for all durations > 0 if no time has passed, or also for 0 if time has passed
								y_d_logits = tf.multiply(mask, y_d_logits) # 0 for class 0 where no time has passed
								weight_mask = tf.multiply(weight_mask, tf.constant(1.0e38))
								y_d_logits = tf.subtract(y_d_logits, weight_mask) # -large number for class 0 where no time has passed = impossible (almost) choice
							y_d_logits = tf.divide(y_d_logits, o.duration_temp)
							y_d_logits = tf.reshape(y_d_logits, [-1, p["num_durations"]])
							if p["modelling_properties"]["duration"]["multiclass"]:
								if post:
									o.y_d_probs = tf.nn.sigmoid(y_d_logits)
								sample_fn = lambda: tf.cast(tf.distributions.Bernoulli(logits = y_d_logits).sample(), tf.float32)
								max_fn = lambda: tf.cast(tf.greater(y_d_logits, tf.constant(0.0, dtype = tf.float32)), tf.float32) # same as y = sigmoid > 0.5
								y_d_preds = tf.cond(sample_from_predictions, sample_fn, max_fn)
							else:
								if post: 	
									o.y_d_probs = tf.nn.softmax(y_d_logits)
								sample_fn = lambda: tf.squeeze(tf.random.categorical(y_d_logits, 1, dtype = tf.int32), axis = -1) # doesn't remove a dim by default
								max_fn = lambda: tf.argmax(y_d_logits, axis = -1, output_type = tf.int32)
								y_d_preds = tf.cond(sample_from_predictions, sample_fn, max_fn)
								y_d_preds = tf.one_hot(y_d_preds, p["num_durations"])								
							if post:
								o.y_d_probs = tf.reshape(o.y_d_probs, [o.batch_sz, o.inp_lengths_max, p["num_durations"]])
								o.y_d_probs = tf.multiply(o.y_d_probs, o.inp_lengths_mask)	
								o.y_d_preds = tf.reshape(y_d_preds, [o.batch_sz, o.inp_lengths_max, p["num_durations"]])
								o.y_d_preds = tf.multiply(o.y_d_preds, o.inp_lengths_mask)		
								o.y_d_logits = tf.reshape(y_d_logits, [o.batch_sz, o.inp_lengths_max, p["num_durations"]])
								o.y_d_logits = tf.multiply(o.y_d_logits, o.inp_lengths_mask)	
								o.y_d = y_d = tf.identity(o.y_d_preds, name = "y_d")
								if p["training_output"]["dists"]:
									o.y_d_guess_dist = tf.reduce_sum(tf.cast(o.y_d_preds, tf.int64), axis = [0, 1]) # (features) with stats over which features have been guessed most
									o.y_d_target_dist = tf.reduce_sum(tf.cast(o.y_d, tf.int64), axis = [0, 1])
									o.y_d_diff_dist = tf.abs(tf.subtract(o.y_d_guess_dist, o.y_d_target_dist))
									o.y_d_guess_tot_dist = o.y_d_guess_tot_dist_var.assign(tf.add(o.y_d_guess_tot_dist_var, o.y_d_guess_dist))
								y_features += p["num_durations"]
							else:
								y_d = tf.reshape(y_d_preds, [-1, p["num_durations"]], name = "y_d_samp")	
								y = y_d if y is None else tf.concat([y, y_d], axis = -1)
							dec_output = tf.concat([dec_output, y_d], axis = -1)				

					###################################################################################################################################################
					### PITCH for the event we are predicting #########################################################################################################
					###################################################################################################################################################
					if p["modelling_properties"]["pitch"]["include"]:
						# predict the pitch by concatenating the true target offset and beat information from this very time step
						
						with tf.variable_scope(p["scopes"]["pitch"]): #tf.variable_scope("pitch_output"):
							# first compute the logits which is done differently depending on if we use the balstm or not
							if p["dec"]["model"] == "balstm":
								# pitch is predicted using the initial output from the decoder lstm, that is, every pitch is still seen as a separate sequence of 
								# events, as derived by the convolution on surrounding pitches from last time step
								if y is not None:
									repeated_y = tf.tile(y, [1, params["num_pitches"], 1]) # (batch, steps * num_pitches, features)
									repeated_y = tf.reshape(repeated_y, [o.balstm_batch_sz, o.inp_lengths_max, y_features])			
									o.pitch_dec_output = tf.concat((o.pitch_dec_output, repeated_y), axis = 2)
								y_p_logits = o.pitch_dec_output
								for num, layer in enumerate(p["output_layers"]["pitch"]):
									if p["batch_norm"] and p["batch_norm_before_act"]:
										act = layer.activation
										layer.activation = None
										y_p_logits = act(tf.layers.batch_normalization(layer(y_p_logits), training = o.training, name = "pitch_BN"))
										layer.activation = act
									else:
										y_p_logits = layer(y_p_logits)
									if num < len(p["output_layers"]["pitch"]) - 1: # no dropout or post activation BN on last output layer
										if p["modelling_properties"]["pitch"]["dropout"] > 0.0: 
											y_p_logits = tf.nn.dropout(y_p_logits, rate = p["modelling_properties"]["pitch"]["dropout"])
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										y_p_logits = tf.layers.batch_normalization(y_p_logits, training = o.training, name = "pitch_BN")
								y_p_logits = tf.reshape(y_p_logits, [o.batch_sz, p["num_pitches"], o.inp_lengths_max])
								y_p_logits = tf.transpose(y_p_logits, [0, 2, 1]) # back to (batch, steps, pitches) order			
							else:
								y_p_logits = dec_output
								for num, layer in enumerate(p["output_layers"]["pitch"]):						
									if p["batch_norm"] and p["batch_norm_before_act"]:
										act = layer.activation
										layer.activation = None
										y_p_logits = act(tf.layers.batch_normalization(layer(y_p_logits), training = o.training, name = "pitch_BN"))
										layer.activation = act
									else:
										y_p_logits = layer(y_p_logits)
									if num < len(p["output_layers"]["pitch"]) - 1: # no dropout or post activation BN on last output layer
										if p["modelling_properties"]["pitch"]["dropout"] > 0.0: 
											y_p_logits = tf.nn.dropout(y_p_logits, rate = p["modelling_properties"]["pitch"]["dropout"])
									if p["batch_norm"] and not p["batch_norm_before_act"]:
										y_p_logits = tf.layers.batch_normalization(y_p_logits, training = o.training, name = "pitch_BN")

							if p["modelling_properties"]["offset"]["include"] and p["modelling_properties"]["pitch"]["same_step_reduction"]:
								# adjust the output logits if the offset from previous pitch was 0. This implies that no pitches below the previous pitch may be sampled
								# output is (batch, step) with 1 where time has not moved on (0 non-zero indices indicating offset = 0) max arg, if it is 0 ,we are at the 
								#same time as before and should restrict softmax output range
								if p["modelling_properties"]["offset"]["multiclass"]:
									next_timestep = tf.count_nonzero(o.y_o, axis = 2, dtype = tf.int32) # (batch, steps) with 0 if time offset from previous was 0, otherwise 1
								else:
									next_timestep = tf.argmax(o.y_o, axis = 2, output_type = tf.int32) # (batch, steps) with 0 if time offset from previous was 0, otherwise 1
								next_timestep = tf.equal(next_timestep, 0) # (batch, steps) with True if no time from previous has passed
								next_timestep = tf.cast(next_timestep, dtype = tf.int32) # (batch, steps) with 1 if no time from previous has passed, otherwise 0
								pitch_thresholds = tf.argmax(o.x_p, axis = 2, output_type = tf.int32) # (batch, steps) with the index of the previous played note, 0 if no note was played
								
								mask = tf.multiply(next_timestep, pitch_thresholds) # (batch, steps) with 1 * threshold if 0 time has passed, otherwise 0 after elementwise multi
								mask = tf.reshape(mask, [o.batch_sz, o.inp_lengths_max, 1]) # output is (batch, step, 1)
								pitch_indices = tf.range(0, p["num_pitches"], dtype = tf.int32) # (pitches)
								
								mask = tf.greater_equal(pitch_indices, mask) # (batch, steps, pitches)
								mask = tf.cast(mask, tf.float32)
								y_p_logits = tf.multiply(mask, y_p_logits)
								mask = tf.subtract(tf.constant(1.0), mask) # inverted mask, 1 for forbidden features
								neg_weight_mask = tf.multiply(mask, tf.constant(1e38, dtype = tf.float32))
								y_p_logits = tf.subtract(y_p_logits, neg_weight_mask)		
								
							# continue processing the pitch logits
							y_p_logits = tf.divide(y_p_logits, o.pitch_temp)
							y_p_logits = tf.reshape(y_p_logits, [-1, p["num_pitches"]])
							sample_fn = lambda: tf.squeeze(tf.random.categorical(y_p_logits, 1, dtype = tf.int32), axis = -1)
							max_fn = lambda: tf.argmax(y_p_logits, axis = -1, output_type = tf.int32)
							y_p_preds = tf.cond(sample_from_predictions, sample_fn, max_fn)
							y_p_preds = tf.one_hot(y_p_preds, p["num_pitches"])
							if post:
								o.y_p_probs = tf.nn.softmax(y_p_logits)
								o.y_p_probs = tf.reshape(o.y_p_probs, [o.batch_sz, o.inp_lengths_max, p["num_pitches"]])
								o.y_p_probs = tf.multiply(o.y_p_probs, o.inp_lengths_mask)								
								o.y_p_preds = tf.reshape(y_p_preds, [o.batch_sz, o.inp_lengths_max, p["num_pitches"]])
								o.y_p_preds = tf.multiply(o.y_p_preds, o.inp_lengths_mask)
								o.y_p_logits = tf.reshape(y_p_logits, [o.batch_sz, o.inp_lengths_max, p["num_pitches"]])
								o.y_p_logits = tf.multiply(o.y_p_logits, o.inp_lengths_mask)		
								o.y_p = y_p = tf.identity(o.y_p_preds, name = "y_p")
								
								if p["training_output"]["dists"]:
									o.y_p_guess_dist = tf.reduce_sum(tf.cast(o.y_p_preds, tf.int64), axis = [0, 1]) # (features) with stats over which features have been guessed most
									o.y_p_target_dist = tf.reduce_sum(tf.cast(o.y_p, tf.int64), axis = [0, 1])
									o.y_p_diff_dist = tf.abs(tf.subtract(o.y_p_guess_dist, o.y_p_target_dist))
									o.y_p_guess_tot_dist = o.y_p_guess_tot_dist_var.assign(tf.add(o.y_p_guess_tot_dist_var, o.y_p_guess_dist))													
							else:
								y_p = tf.reshape(y_p_preds, [-1, p["num_pitches"]])				
								y = y_p if y is None else tf.concat([y, y_p], axis = -1)
							# use the sampled predictions as ground truth if targets were not supplied in this run, e. g. when generating new music, during training, targets
							# will be available and so the above and this op result in different values, during generation they won't but then the above won't be evaluated since
							# one usually aren't interested in metrics during generation (actually metrics are undefined with generation since there is no ground truth)

							dec_output = tf.concat([dec_output, y_p], axis = -1)

					##############################################################################################################################################
					### INSTRUMENT for the event we are predicting ###############################################################################################
					##############################################################################################################################################
					if p["modelling_properties"]["instrument"]["include"]:
					
						with tf.variable_scope(p["scopes"]["instrument"]): #tf.variable_scope("instrument_output"):
							args = [o, p, dec_output]
							kwargs = {"multiclass": p["modelling_properties"]["instrument"]["multiclass"], "mask": o.inp_lengths_mask, "sample": sample_from_predictions, 
									"temp": o.instrument_temp, "layers": p["output_layers"]["instrument"], "num_out_feat": p["num_instruments"], "prefix": "instrument", "full": post}
							if post:
								o.y_i_logits, o.y_i_probs, o.y_i_preds = _output_layer(*args, **kwargs)
								o.y_i = y_i = tf.identity(o.y_i_preds, name = "y_i")
								if p["training_output"]["dists"]:
									o.y_i_guess_dist = tf.reduce_sum(tf.cast(o.y_i_preds, tf.int64), axis = [0, 1]) # (features) with stats over which features have been guessed most
									o.y_i_target_dist = tf.reduce_sum(tf.cast(o.y_i, tf.int64), axis = [0, 1])
									o.y_i_diff_dist = tf.abs(tf.subtract(o.y_i_guess_dist, o.y_i_target_dist))
									o.y_i_guess_tot_dist = o.y_i_guess_tot_dist_var.assign(tf.add(o.y_i_guess_tot_dist_var, o.y_i_guess_dist))								
							else:
								y_i_preds = _output_layer(*args, **kwargs)
								y_i = y_i_preds
								y = y_i if y is None else tf.concat([y, y_i], axis = -1)
							
				if not post:
					return y
					
			return output
					
		def _metrics(preds, targets, include_distance, classes, prefix, zero_hot_correction = tf.constant(0.0)):
			
			with tf.variable_scope(prefix + "_metrics"):
			
				# derives performance metrics in terms of true positives (tp), true negatives (tn), false positives (fp) and false negatives (fn)
				# then use these to establish:
				#	-accuracy (tp + tn) / (tp + tn + fp + fn), e.g. how many correct (positive and negative) predictions were made out of the total
				#	-precision tp / (tp + fp), e.g. what percentage of positives were correctly classified
				# 	-recall tp / (tp + fn), e.g. what percentage of positive were found
				tp = tf.reduce_sum(tf.multiply(preds, targets)) # correctly classified as 1's
				fn = tf.reduce_sum(targets) - tp # incorrectly classified as 0's (should be 1's)
				fp = tf.reduce_sum(preds) - tp - zero_hot_correction # incorrectly classified as 1's (should be 0's)
				tn = tf.cast(tf.multiply(o.inp_lengths_sum, classes), tf.float32) - (tp + fn + fp) + zero_hot_correction
				acc = tf.divide(tf.add(tp, tn), tf.add(tf.add(tp, tn), tf.add(fn, fp)))
				prec = tf.divide(tp, tf.add(tp, fp))
				rec = tf.divide(tp, tf.add(tp, fn))
				if include_distance:
					# only works correctly for mutually exclusive classes, e.g. one hot vectors
					dist = tf.reduce_sum(tf.abs(tf.subtract(tf.argmax(preds, axis = -1, output_type = tf.int32), tf.argmax(targets, axis = -1, output_type = tf.int32))))
					dist_mean = tf.divide(dist, o.inp_lengths_sum)
					return tp, tn, fp, fn, acc, prec, rec, dist, dist_mean
				else:
					return tp, tn, fp, fn, acc, prec, rec
		
		def _loss(o, p):
			
			with tf.variable_scope("loss"):
				losses = []
				losses_mean = []
				# cross-entropy reconstruction losses
				if p["loss"]["recon"]:
					recon_losses = []
					for cat in ["offset", "duration", "pitch", "instrument"]:									  
						if p["modelling_properties"][cat]["include"]:
							if cat in ["offset", "duration"]:
								mask = tf.squeeze(o.inp_lengths_mask, axis = 2)
							else:
								mask = o.pd_mask
							cat_params = p["modelling_properties"][cat]
							logits = getattr(o, "y_" + cat_params["key"] + "_logits")
							targets = getattr(o, "y_" + cat_params["key"])
							if cat_params["multiclass"]:
								# loss is the mean cross entropy loss over all the duration features (binary classification per partial (basic) duration)
								loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = logits, labels = targets) # (batch, steps, feat)
								loss = tf.divide(tf.reduce_sum(loss, axis = 2), tf.constant(cat_params["num_features"], dtype = tf.float32)) # (batch, steps) (mean over features)
							else:
								# loss is the usual softmax cross entropy loss (categorical classification)
								if p["loss"]["framework"] == "plain":
									loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = targets) # (batch, steps)
								else:
									loss = tf.contrib.seq2seq.sequence_loss(
										logits, 
										tf.argmax(targets, axis = -1),
										mask,
										average_across_timesteps = True,
										average_across_batch = True,
										softmax_loss_function = tf.nn.sparse_softmax_cross_entropy_with_logits) # scalar
							setattr(o, "loss_" + cat, loss)
							recon_losses += [loss]
				
					o.recon_loss = tf.add_n(recon_losses)
					
					if p["loss"]["framework"] == "plain":
						# reduce loss over all time steps and divide each sample in the batch with the actual length of the batch so that we get avg loss / time step
						o.recon_loss = tf.multiply(o.recon_loss, mask) # (batch, step) with 0 for all inactive time steps
						sum_then_divide_by_total_timesteps = True
						if sum_then_divide_by_total_timesteps:
							o.recon_loss = tf.reduce_sum(o.recon_loss, axis = 1) # (batch)
							o.recon_loss_sum = tf.reduce_sum(o.recon_loss)
							o.recon_loss_mean = tf.divide(o.recon_loss_sum, tf.cast(o.inp_lengths_sum, dtype = tf.float32))
						else:
							o.recon_loss = tf.reduce_sum(o.recon_loss, axis = 1) # (batch), loss per sample in the batch
							# if any of the samples in a batch is of length 0, nans will be introduced into the loss by the below operation
							o.recon_loss_mean = tf.reduce_mean(tf.divide(o.recon_loss, tf.cast(o.inp_lengths, dtype = tf.float32))) # mean over batch samples and time steps
						losses += [o.recon_loss]
					else:
						o.recon_loss_mean = o.recon_loss
						del o.recon_loss # this is not really the loss like (batch, steps) but only the mean loss if seq2seq is used
					
					losses_mean += [o.recon_loss_mean]
				# calculate the KL divergence loss from the VAE which yields one loss per sample in a batch (since we sample from the VAE only once per sample)
				if p["model"]["vae"] and p["loss"]["vae"]:
					o.latent_loss = -0.5 * tf.reduce_mean(1 - tf.exp(o.log_sigma_sq) - o.mu ** 2 + o.log_sigma_sq, axis = 1) # (batch)
					if p["loss"]["free_bits"] > 0:
						free_nats = p["loss"]["free_bits"] * tf.log(2.0)
						o.latent_loss = tf.maximum(o.latent_loss - free_nats, 0)
					o.latent_loss_mean = tf.reduce_mean(o.latent_loss, axis = 0) # mean over batch samples (not over time steps in any way)
					if p["loss"]["beta_annealing"]:
						beta = (1.0 - tf.pow(p["loss"]["beta_rate"], o.global_step)) * p["loss"]["beta_max"]
						losses += [tf.multiply(o.latent_loss, beta)]
						losses_mean += [tf.multiply(o.latent_loss_mean, beta)]
					else:
						losses += [o.latent_loss]
						losses_mean += [o.latent_loss_mean]
				
				o.loss = tf.add_n(losses) if len(losses) > 0 else None # losses per sequence in a batch
				o.loss_mean = tf.add_n(losses_mean) # loss per time step and batch for recon added to loss per batch for VAE
				# add together the reconstruction loss and the KL loss and divide by batch size to yield loss per sample and time step e.g. loss per time step
				if p["loss"]["regularization"] > 0.0:
					print("[MAHLERNET]: adding regularization to trainable variables ...")
					reg_losses = []
					for var in tf.trainable_variables():
						if not ("noreg" in var.name or "Bias" in var.name or "bias" in var.name or "init_state" in var.name):
							reg_losses += [tf.nn.l2_loss(var)]
							print("[MAHLERNET]:        added regularization to", var.name)
						else:
							print("[MAHLERNET]:        DIDN'T add regularization to", var.name)
					reg_loss = p["loss"]["regularization"] * tf.add_n(reg_losses)
					o.loss = tf.add(o.loss, reg_loss)
					o.loss_mean = tf.add(o.loss_mean, reg_loss)
				
		def _optimizer(o, p):
			if p["optimizer"] == "rmsprop":
				optimizer = tf.train.RMSPropOptimizer(p["learning_rate"])
			else:
				optimizer = tf.train.AdamOptimizer(self.params["learning_rate"])
			o.gradients = optimizer.compute_gradients(o.loss_mean)
			o.clipped_gradients = []
			zero_gradients = []
			print("[MAHLERNET]: printing trainable variables ...")
			for grad, var in o.gradients:
				if grad is not None:
					o.clipped_gradients += [(tf.clip_by_value(grad, -1., 1.), var)]
					print("[MAHLERNET]:        Found variable with non-ZERO gradient: ", var.name, var.shape)
				else:
					zero_gradients += [var]
					o.clipped_gradients += [(grad, var)]
			if len(zero_gradients) == 0:
				print("[MAHLERNET]: found 0 zero gradients, all variables have gradients")
			else:
				print("[MAHLERNET]: found variables with zero gradients, investigate if unexpected")
				for zv in zero_gradients:
					print("[MAHLERNET]:         ", zv.name, zv.shape)
			o.train_step = optimizer.apply_gradients(o.clipped_gradients, o.global_step)
			update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			update_op = [op for op in update_op if "while" not in op.name] # removes update ops arising from using batch normalization with scheduled sampling
			o.train_step = tf.group([o.train_step, update_op])

		# build the model
		o = self.ops
		p = self.params
		w = self.weights
		
		o.dummy = tf.constant(1) # dummy op that can be fetched and used for printing operations if needed
		
		#######################################################################################################################################################
		### INPUT setup placeholders for the input ############################################################################################################
		#######################################################################################################################################################
		
		_inputs(o, p)
		
		#######################################################################################################################################################
		### STATS about the current input batch to use during further processing ##############################################################################
		#######################################################################################################################################################
		
		_stats(o, p)
		
		#######################################################################################################################################################
		### ENCODER ###########################################################################################################################################
		#######################################################################################################################################################
		
		if p["model"]["inp"] or p["model"]["ctx"]:
			_encoder_component(o, p)
		
		#######################################################################################################################################################
		### VAE ###############################################################################################################################################
		#######################################################################################################################################################
		
		if p["model"]["vae"] and (p["model"]["inp"] or p["model"]["ctx"]):
			print("[MAHLERNET]: adding VAE component") if p["verbose"] >= 1 else 0
			_vae(o, p, w)
		
		#######################################################################################################################################################
		### DECODER for decoding the latent vector ############################################################################################################
		#######################################################################################################################################################
				
		# set up output layers
		if p["dec"]["model"] == "balstm":
			p["output_layers"]["pitch"][-1] = (1, p["output_layers"]["pitch"][-1][1]) # if using the balstm, the output layer will only contain one output (per pitch of course)
		for prop in ["offset", "pitch", "duration", "instrument"]:
			if p["modelling_properties"][prop]["include"]:
				layers = []
				for num, (units, activation) in enumerate(p["output_layers"][prop]):
					layers += [tf.layers.Dense(units, activation = activation, kernel_initializer = tf.random_normal_initializer(stddev=0.001))]
				p["output_layers"][prop] = layers
				with tf.variable_scope(prop + "_output", reuse = tf.AUTO_REUSE) as scope:
					p["scopes"][prop] = scope
				
		_post_output = _output_fn(o, p, True)
		_decoding_output = _output_fn(o, p, False)
		_decoder_ctx_setup(o, p)
		if self.params["modelling_properties"]["pitch"]["include"] and self.params["dec"]["model"] == "balstm":
			# can only use balstm with pitch included, otherwise it s just overly complicated
			print("[MAHLERNET]: adding BALSTM decoder")
			_balstm_decoder(o, p, w)
		else:
			print("[MAHLERNET]: adding regular rnn decoder")
			_lstm_decoder(o, p, w, _decoding_output)
						
		#######################################################################################################################################################
		### OUTPUT layers from decoder output to predictions ##################################################################################################
		#######################################################################################################################################################

		_post_output(o.dec_output)

		#######################################################################################################################################################
		### METRICS to evaluate training ######################################################################################################################
		#######################################################################################################################################################

		for (prop, extra) in [("offset", None), ("duration", None), ("pitch", o.y_p_zero), ("instrument", o.y_i_zero)]:	
			if p["modelling_properties"][prop]["include"]:
				cat_params = p["modelling_properties"][prop]
				cat = "y_" + cat_params["key"]
				preds = getattr(o, cat + "_preds")
				targets = getattr(o, cat)
				if extra is not None:
					ret = _metrics(preds, targets, cat_params["dist_metric"], cat_params["num_features"], prop, extra)
				else:
					ret = _metrics(preds, targets, cat_params["dist_metric"], cat_params["num_features"], prop)
				if cat_params["dist_metric"] and not cat_params["multiclass"]:
					tp, tn, fp, fn, acc, prec, rec, dist, dist_mean = ret
					setattr(o, cat + "_dist", dist)
					setattr(o, cat + "_dist_mean", dist_mean)
				else:
					tp, tn, fp, fn, acc, prec, rec = ret
				setattr(o, cat + "_tp", tp)
				setattr(o, cat + "_tn", tn)
				setattr(o, cat + "_fp", fp)
				setattr(o, cat + "_fn", fn)
				setattr(o, cat + "_acc", acc)
				setattr(o, cat + "_prec", prec)
				setattr(o, cat + "_rec", rec)
	
		#######################################################################################################################################################
		### LOSSES for training ###############################################################################################################################
		#######################################################################################################################################################
		
		_loss(self.ops, self.params)
		
		#######################################################################################################################################################
		### OPTIMIZER for training ############################################################################################################################
		#######################################################################################################################################################
		
		_optimizer(self.ops, self.params)
		
		#######################################################################################################################################################
		#######################################################################################################################################################
		#######################################################################################################################################################

		if self.params["save_graph"]:
			self.save_graph()			
			
	def train(self, generator_fctn, samples, ts_samples, vs_samples, save_file = None, init_vars = True, eoe_fns = None):
		bar_width = 30
		batches_per_epoch = math.ceil(ts_samples / self.params["batch_sz"])
		mode_metrics = [None, None]
		times = [None, None]
		epoch_losses = [[], []] # avg losses over epoch for training and validation
		step_losses = [[], []] # avg loss over step for training and validation (for validation, step stands still so this is the same av for epochs)
		epoch_prec_rec = {key: [[], []] for cat, key in list(filter(lambda a: self.params["modelling_properties"][a[0]]["include"],[("offset", "o"), ("duration", "d"), ("pitch", "p"), ("instrument", "i")]))}
		step_prec_rec = {key: [[], []] for cat, key in list(filter(lambda a: self.params["modelling_properties"][a[0]]["include"],[("offset", "o"), ("duration", "d"), ("pitch", "p"), ("instrument", "i")]))}
		epoch_dist = {key: [[], []] for cat, key in list(filter(lambda a: self.params["modelling_properties"][a[0]]["include"] and self.params["modelling_properties"][a[0]]["dist_metric"],[("offset", "o"), ("duration", "d"), ("pitch", "p"), ("instrument", "i")]))}
		step_dist = {key: [[], []] for cat, key in list(filter(lambda a: self.params["modelling_properties"][a[0]]["include"] and self.params["modelling_properties"][a[0]]["dist_metric"],[("offset", "o"), ("duration", "d"), ("pitch", "p"), ("instrument", "i")]))}		
		interruption = False
		
		def SIGINT_handler(num, frame):
			nonlocal interruption
			print_divider("[MAHLERNET]: Abort requested, will terminate gracefully after current epoch is over", " ")
			if interruption:
				raise KeyboardInterrupt("Pressed ctrl-c")
			else:
				interruption = True
			
		signal.signal(signal.SIGINT, SIGINT_handler)
		with self.session:
			if init_vars:
				self.session.run(tf.global_variables_initializer())
			
			print("[MAHLERNET]: starting training " + str(self.variable_count()) + " variables on " + str(samples) +  " samples (" + str(ts_samples) + " / " + str(vs_samples) + " for training / validation)...")
			for epoch_no in range(1, self.params["epochs"] + 1):
				for gen_index, (gen, samples) in enumerate(zip(generator_fctn(self.params["batch_sz"]), (ts_samples, vs_samples))):
					if gen_index == 1:
						validation = True
					else:
						validation = False
					metrics = { 
						"p": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "acc": 0, "prec": 0, "rec": 0, "dist_tot": 0, "dist_avg": 0}, 
						"o": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "acc": 0, "prec": 0, "rec": 0, "dist_tot": 0, "dist_avg": 0}, 
						"d": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "acc": 0, "prec": 0, "rec": 0, "dist_tot": 0, "dist_avg": 0},  
						"i": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "acc": 0, "prec": 0, "rec": 0},  
						"loss": {"tot": 0, "avg": 0},					
						"samples": {"tot": 0},
						"timesteps": {"tot": 0}
					}
					e_time = time.time()
					for step, batch in enumerate(gen):
						s_time = time.time()
						
						# set up feeds to the graph and what ops to run ######################################################################################
						run_ops = [self.ops.dummy, self.ops.loss_mean, self.ops.inp_lengths_sum, self.ops.global_step]
						feed_dict = {
								self.ops.batch_sz: batch["batch_sz"],
								self.ops.sample_from_predictions: False,
								self.ops.inp_lengths: batch["inp_lengths"]
						}
						if self.params["model"]["ctx"]:
							feed_dict[self.ops.ctx_s] = batch["ctx_s"]
							feed_dict[self.ops.ctx_lengths] = batch["ctx_lengths"]
						for cat in ["offset", "duration", "pitch", "instrument"]:
							if self.params["modelling_properties"][cat]["include"]:
								cat_params = self.params["modelling_properties"][cat]
								types = ["tp", "tn", "fp", "fn"]
								if self.params["training_output"]["dists"]:
									types += ["guess_dist", "target_dist", "diff_dist", "guess_tot_dist"]
								if cat_params["dist_metric"]:
									types += ["dist", "dist_mean"]
								for type in types:
									run_ops += [getattr(self.ops, "y_" + cat_params["key"] + "_" + type)]
								y_key = "y_" + cat_params["key"]
								x_key = "x_" + cat_params["key"]
								ctx_key = "ctx_" + cat_params["key"]
								feed_dict[getattr(self.ops, y_key)] = feed_dict[getattr(self.ops, y_key + "_")] = batch[y_key]
								feed_dict[getattr(self.ops, x_key)] = batch[x_key]
								if cat == "offset" and self.params["modelling_properties"]["beats"]["include"]:
									feed_dict[getattr(self.ops, "y_b")] = feed_dict[getattr(self.ops, "y_b_")] = batch["y_b"]
									feed_dict[getattr(self.ops, "x_b")] = batch["x_b"]	
									if self.params["model"]["ctx"]:
										feed_dict[getattr(self.ops, "ctx_b")] = batch["ctx_b"]
								if cat == "pitch" and self.params["modelling_properties"]["active_pitches"]["include"]:
									feed_dict[getattr(self.ops, "x_ap")] = batch["x_ap"]
								if cat == "instrument" and self.params["modelling_properties"]["active_instruments"]["include"]:
									feed_dict[getattr(self.ops, "x_ai")] = batch["x_ai"]
								if self.params["model"]["ctx"]:
									feed_dict[getattr(self.ops, ctx_key)] = batch[ctx_key]
						if validation:
							feed_dict[self.ops.training] = False
							feed_dict[self.ops.inp_enc_dropout] = 0.0
							feed_dict[self.ops.ctx_enc_dropout] = 0.0
							feed_dict[self.ops.vae_dropout] = 0.0
							feed_dict[self.ops.dec_dropout] = 0.0
							feed_dict[self.ops.dec_balstm_conv_dropout] = 0.0
							feed_dict[self.ops.pitch_dropout] = 0.0
							feed_dict[self.ops.offset_dropout] = 0.0
							feed_dict[self.ops.duration_dropout] = 0.0
							feed_dict[self.ops.instrument_dropout] = 0.0
						else:
							feed_dict[self.ops.training] = True
							run_ops += [self.ops.train_step] # IMPORTANT! only run the train_step which results in updated weights on TRAINING, not on validation or inference
							feed_dict[self.ops.inp_enc_dropout] = self.params["inp_enc"]["dropout"]
							feed_dict[self.ops.ctx_enc_dropout] = self.params["ctx_enc"]["dropout"]
							feed_dict[self.ops.vae_dropout] = self.params["vae"]["dropout"]
							feed_dict[self.ops.dec_dropout] = self.params["dec"]["dropout"]
							feed_dict[self.ops.dec_balstm_conv_dropout] = self.params["dec"]["balstm"]["conv_dropout"]	
							feed_dict[self.ops.pitch_dropout] = self.params["modelling_properties"]["pitch"]["dropout"]
							feed_dict[self.ops.offset_dropout] = self.params["modelling_properties"]["offset"]["dropout"]
							feed_dict[self.ops.duration_dropout] = self.params["modelling_properties"]["duration"]["dropout"]
							feed_dict[self.ops.instrument_dropout] = self.params["modelling_properties"]["instrument"]["dropout"]							
						
						# run one step of computations #######################################################################################################
						ret = self.session.run(run_ops, feed_dict, tf.RunOptions(report_tensor_allocations_upon_oom = True))
						
						# take care of the results from computations #########################################################################################
						fetched = 4
						it = iter(ret)
						dummy = next(it) # dummy node used for printouts in the graph (since it is retrieved, all computations that depend on it will get executed)
						l = next(it) # mean loss per time step
						if not validation:
							step_losses[0] += [l]
						timesteps = next(it) # total number of time steps processed in this batch
						global_step = next(it) # total number of gradient updates that has been done since started training
						metrics["samples"]["tot"] += len(batch["inp_lengths"])
						metrics["loss"]["tot"] += l
						metrics["loss"]["avg"] = metrics["loss"]["tot"] / (step + 1)
						metrics["timesteps"]["tot"] += timesteps
						property_string = ""
						for cat in ["offset", "duration", "pitch", "instrument"]:
							if self.params["modelling_properties"][cat]["include"]:
								cat_params = self.params["modelling_properties"][cat]
								key = cat_params["key"]
								tp, tn, fp, fn = next(it), next(it), next(it), next(it)
								if not validation:
									step_prec_rec[key][0] += [fp]
								fetched += 4
								if self.params["training_output"]["dists"]:
									guesses, targets, diff, guesses_tot = next(it), next(it), next(it), next(it)
									print(cat + " guess distribution:", guesses)
									print(cat + " target distribution:", targets)
									print(cat + " difference distribution:", diff) 
									print(cat + " guess total distribution:",guesses_tot)
									fetched += 4
								if cat_params["dist_metric"]:
									fetched += 2
									dst, dst_mean = next(it), next(it)
									if not validation:
										step_dist[key][0] += [dst_mean]
									metrics[key]["dist_tot"] += dst
									metrics[key]["dist_avg"] = metrics[key]["dist_tot"] / metrics["timesteps"]["tot"]
								metrics[key]["tp"] += tp
								metrics[key]["tn"] += tn
								metrics[key]["fp"] += fp
								metrics[key]["fn"] += fn	

								# Run metrics
								#	-accuracy (tp + tn) / (tp + tn + fp + fn), e.g. how many correct (positive and negative) predictions were made out of the total
								#	-precision tp / (tp + fp), e.g. what percentage of positives were correctly classified
								# 	-recall tp / (tp + fn), e.g. what percentage of positive were found	
								# since we only guess once per vector, and every vector contains one 1, precision and recall are bound to be the same, there
								# are exceptions to this since pitch and instrument may be all zeros in the input data but this is corrected in the graphs by
								# counting the number of such time steps and subracting and adding this number to the appropriate property
								# can't have a zero denominator
								metrics[key]["acc"] = (metrics[key]["tp"] + metrics[key]["tn"]) / (metrics[key]["tp"] + metrics[key]["tn"] + metrics[key]["fp"] + metrics[key]["fn"])
								# gets a zero denominator if the original data contains at least one event with a 1 in the input
								if (metrics[key]["tp"] + metrics[key]["fn"]) == 0:
									metrics[key]["rec"] = 1.0 # since there were no 1's, we found them all
								else:
									metrics[key]["rec"] = metrics[key]["tp"] / (metrics[key]["tp"] + metrics[key]["fn"])
								# gets a 0 denominator if all quesses are negative
								if (metrics[key]["tp"] + metrics[key]["fp"]) == 0:
									if metrics[key]["fn"] > 0:
										metrics[key]["prec"] = 0.00 # there were positives but we found none, that's a 0.0 precision
									else:
										metrics[key]["prec"] = 1.00 # there were no positives, then we have per definition found all of them
								else:
									metrics[key]["prec"] = metrics[key]["tp"] / (metrics[key]["tp"] + metrics[key]["fp"])
								if not cat_params["multiclass"]:
									assert(metrics[key]["prec"] == metrics[key]["rec"]), "[MAHLERNET]: ERROR - precision and recall difference for key \"" + cat + "\":" + str(metrics[key]["prec"]) + " / " + str(metrics[key]["rec"])
								if "dist_avg" in metrics[key]:
									property_string += (" " + key + "(" + ('%.2f' % metrics[key]["rec"]) + "/" + ('%.2f' % metrics[key]["dist_avg"]) + ")")
								else:
									property_string += (" " + key + "(" + ('%.2f' % metrics[key]["rec"]) + ")")
						if not validation:
							train_step = next(it)
							fetched += 1
						assert(fetched == len(ret)), "[MAHLERNET]: ERROR - fetched values from session and used values differ (" + str(fetched) + "/" + str(len(ret)) + ")"
						
						# progress bar
						filled_bars = math.floor((metrics["samples"]["tot"] / samples) * float(bar_width))
						filled = ['=' for _ in range(filled_bars)]
						if len(filled) < bar_width:
							filled += [">"]
						unfilled = ["-" for _ in range(bar_width - len(filled))]
						bar = filled + unfilled
						bar = "".join(bar)
					
						# time elapsed during step
						passed_time = time.time() - s_time
						if passed_time >= 3600:
							time_string = time.strftime("%H h %M m. %S s.", time.gmtime(passed_time))
						elif passed_time >= 60:
							time_string = time.strftime("%M m. %S s.", time.gmtime(passed_time))
						else:
							time_string = time.strftime("%S s.", time.gmtime(passed_time))
						mode = "TR-"
						if validation:
							mode = "VA-"
						print("[MAHLERNET]: " + mode + "Epoch " + str(epoch_no) + " (gs " + str(int(global_step)) + ") |" + bar + "| t: " + time_string + ", l: " + ('%.2f' % metrics["loss"]["avg"]) + ", p(/dst):" + property_string + ")", end = '\r')
					mode_metrics[gen_index] = copy.deepcopy(metrics) # save the training metrics while validating
					times[gen_index] = (e_time, time.time())
				# time elapsed during epoch
				for t_ind, (t_s, t_e) in enumerate(times):
					passed_time = t_e - t_s
					if passed_time >= 3600:
						time_string = time.strftime("%H h %M m. %S s.", time.gmtime(passed_time))
					elif passed_time >= 60:
						time_string = time.strftime("%M m. %S s.", time.gmtime(passed_time))
					else:
						time_string = time.strftime("%S s.", time.gmtime(passed_time))					
					time_string = time.strftime("%H h %M m. %S s.", time.gmtime(t_e - t_s))
					times[t_ind] = time_string
				tr_string = ""
				va_string = ""
				for cat in ["offset", "duration", "pitch", "instrument"]:
					if self.params["modelling_properties"][cat]["include"]:
						key = self.params["modelling_properties"][cat]["key"]
						epoch_prec_rec[key][0] += [mode_metrics[0][key]["rec"]]
						epoch_prec_rec[key][1] += [mode_metrics[1][key]["rec"]]
						step_prec_rec[key][1] += [mode_metrics[1][key]["rec"]]
						if "dist_avg" in metrics[key]:
							epoch_dist[key][0] += [mode_metrics[0][key]["dist_avg"]]
							step_dist[key][0] += [mode_metrics[0][key]["dist_avg"]]
							tr_string += (" " + key + "(" + ('%.2f' % mode_metrics[0][key]["rec"]) + "/" + ('%.2f' % mode_metrics[0][key]["dist_avg"]) + ")")
							va_string += (" " + key + "(" + ('%.2f' % mode_metrics[1][key]["rec"]) + "/" + ('%.2f' % mode_metrics[1][key]["dist_avg"]) + ")")
						else:
							tr_string += (" " + key + "(" + ('%.2f' % mode_metrics[0][key]["rec"]) + ")")								
							va_string += (" " + key + "(" + ('%.2f' % mode_metrics[1][key]["rec"]) + ")")
				epoch_losses[0] += [mode_metrics[0]["loss"]["avg"]]
				epoch_losses[1] += [mode_metrics[1]["loss"]["avg"]]
				step_losses[1] += [mode_metrics[1]["loss"]["avg"]]
				print("[MAHLERNET]: epoch " + str(epoch_no).zfill(3) + " TRAIN, t: " + times[0] + ", l: " + ('%.2f' % mode_metrics[0]["loss"]["avg"]) + ", p(/dst):" + tr_string + ")")
				print("[MAHLERNET]:           VALID, t: " + times[1] + ", l: " + ('%.2f' % mode_metrics[1]["loss"]["avg"]) +  ", p(/dst):" + va_string + ")")
				
				if self.params["save_model"] != 0:
					save = False
					if self.params["save_model"][-1] == "s" and (global_step % int(self.params["save_model"][:-1])) == 0:
						save = True
					if self.params["save_model"][-1] == "e" and (epoch_no % int(self.params["save_model"][:-1])) == 0:
						save = True
					if save:
						name = self.params["model_name"] + "_e" + str(epoch_no) + "_s" + str(int(global_step)) + "_tl" + ('%.2f' % mode_metrics[0]["loss"]["avg"]) + "_vl" + ('%.2f' % mode_metrics[1]["loss"]["avg"])
						self.save_model(name)
						print("[MAHLERNET]: saved model after " + str(epoch_no) + " epochs (" + str(int(global_step)) + " steps) under name " + name)	
				if eoe_fns is not None and self.params["end_of_epoch_fns"] > 0 and (epoch_no % self.params["end_of_epoch_fns"]) == 0:
					for fn in eoe_fns:
						fn(self, epoch_no, global_step)
				if interruption:
					break
		# session is automatically closed when used as a context manager, otherwise call session.close()
		return (epoch_losses, step_losses, epoch_prec_rec, step_prec_rec, epoch_dist, step_dist)
		
	# predict a single time step
	def predict(self, data, decoder_state, last_time, z = None, ctx_enc_summary = None, teacher_forced_input = None, tries = 1):
		use_teacher_forcing = (teacher_forced_input is not None)
		feed_dict = {
			self.ops.training: False,
			self.ops.batch_sz: 1,
			self.ops.bar: self.params["generation"]["default_bar"],
			self.ops.durations: self.params["generation"]["default_durations"],
			self.ops.beats: self.params["generation"]["default_beats"],
			self.ops.inp_lengths: [1],
			self.ops.vae_dropout: 0.0,
			self.ops.dec_dropout: 0.0,
			self.ops.dec_balstm_conv_dropout: 0.0,
			self.ops.pitch_dropout: 0.0,
			self.ops.offset_dropout: 0.0,
			self.ops.duration_dropout: 0.0,
			self.ops.instrument_dropout: 0.0,
		}
		if use_teacher_forcing:
			feed_dict[self.ops.sample_from_predictions] = False
		else:
			feed_dict[self.ops.sample_from_predictions] = True
			feed_dict[self.ops.instrument_temp] = self.params["modelling_properties"]["instrument"]["generation_temperature"]
			feed_dict[self.ops.pitch_temp] = self.params["modelling_properties"]["pitch"]["generation_temperature"]
			feed_dict[self.ops.offset_temp] = self.params["modelling_properties"]["offset"]["generation_temperature"] + ((tries - 1) * 0.1)
			feed_dict[self.ops.duration_temp] = self.params["modelling_properties"]["duration"]["generation_temperature"]
		run_ops = [self.ops.dummy, self.ops.dec_final_state]
		if self.params["modelling_properties"]["offset"]["include"]:
			feed_dict[self.ops.last_time] = last_time
			run_ops += [self.ops.time]
		if self.params["dec"]["model"] == "balstm":
			feed_dict[self.ops.balstm_lengths] = [1] * self.params["num_pitches"]
		if self.params["model"]["ctx"]:
			assert(ctx_enc_summary is not None), "[MAHLERNET]: ERROR - predicting with a model using ctx but context was not provided"
			if ctx_enc_summary is not None:
				feed_dict[self.ops.ctx_enc_summary] = ctx_enc_summary
		if self.params["model"]["vae"]:
			assert(z is not None), "[MAHLERNET]: ERROR - predicting with a model using vae but latent vector was not provided"
			if z is not None:
				feed_dict[self.ops.z] = z											 
		for cat in self.params["property_order"]["dec"]["inp"]:
			cat_params = self.params["modelling_properties"][cat]
			if cat_params["include"]:
				key = cat_params["key"]
				if cat in ["active_pitches", "active_instruments"]:
					feed_dict[getattr(self.ops, "x_" + key)] = [(data[:, cat_params["indices"][0]: cat_params["indices"][1]] > 0).astype(np.uint8)]
				else:
					feed_dict[getattr(self.ops, "x_" + key)] = [data[:, cat_params["indices"][0]: cat_params["indices"][1]]]
					if use_teacher_forcing:
						feed_dict[getattr(self.ops, "y_" + key)] = [teacher_forced_input[:, cat_params["indices"][0]: cat_params["indices"][1]]]
						if cat in self.params["property_order"]["dec"]["out"]:
							run_ops += [getattr(self.ops, "y_" + key + "_probs")] # probabilities for calculation of perplexity
					run_ops += [getattr(self.ops, "y_" + key + "_preds")] # for example, y_o and y_o_preds are the same when generating except for when teacher forcing is used
		if decoder_state is not None:
			for i in range(self.params["dec"]["num_layers"]):
				if self.params["dec"]["type"] == "lstm":
					feed_dict[getattr(self.ops, "dec_init_h_" + str(i))] = decoder_state[i].h
					feed_dict[getattr(self.ops, "dec_init_c_" + str(i))] = decoder_state[i].c
				else:
					feed_dict[getattr(self.ops, "dec_init_" + str(i))] = decoder_state[i]
		# generate the next time step
		ret = self.session.run(run_ops, feed_dict)
		fetched = 2
		pred = {}
		prob = {}
		it = iter(ret)
		_ = next(it) # dummy
		decoder_state = next(it)
		if self.params["modelling_properties"]["offset"]["include"]:
			last_time = next(it)[0][0][0]
			fetched += 1
		else:
			last_time = last_time + 1
		for cat in ["offset", "beats", "duration", "pitch", "instrument"]:
			cat_params = self.params["modelling_properties"][cat]
			if cat_params["include"]:
				if cat != "beats" and use_teacher_forcing:
					prob[cat_params["key"]] = next(it)[0]
					fetched += 1
				pred[cat_params["key"]] = next(it)[0]
				fetched += 1
		assert(fetched == len(ret)), "assertion failed: different numbers of fetched items from run " + str(len(ret)) + " " + str(fetched)
		return decoder_state, last_time, pred, prob
	
	def empty_context(self, start):
		if start: # generate a start context
			print("[MAHLERNET]: generated a START context")
			ctx = np.zeros([1, self.params["num_features"]], dtype = np.uint8) # generate full-scale time step, sub ranges will be picked in context()
			ctx[0][self.params["modelling_properties"]["special_tokens"]["indices"][0]] = 1
			return (True, ctx)	
		else:
			print("[MAHLERNET]: generated an all zeros context")
			ctx = np.zeros([1, self.params["ctx_enc"]["sz_state"] * 2], dtype = np.uint8)
			return (False, ctx)		
			
	# data is a dict with the appropriate keys set depending on what we're modelling: for example s, o, b, p
	def context(self, data):
		run_ops = self.ops.ctx_enc_summary
		feed_dict = { self.ops.batch_sz: 1, self.ops.ctx_lengths: [data.shape[0]], self.ops.ctx_enc_dropout: 0.0 }
		for cat in self.params["property_order"]["ctx"]:
			cat_params = self.params["modelling_properties"][cat] 
			if cat_params["include"]:
				feed_dict[getattr(self.ops, "ctx_" + cat_params["key"])] = [data[:, cat_params["indices"][0]: cat_params["indices"][1]]]
		return self.session.run(run_ops, feed_dict)
		
	# data is a dict with the appropriate keys set depending on what we're modelling: for example s, o, b, p, y_o, y_p, y_d, y_i
	def latent(self, data, ctx_enc_summary = None):
		run_ops = self.ops.z
		feed_dict = { self.ops.batch_sz: 1, self.ops.inp_lengths: [data.shape[0]], self.ops.vae_dropout: 0.0, self.ops.inp_enc_dropout: 0.0 }
		assert(not self.params["model"]["ctx"] or ctx_enc_summary is not None), "[MAHLERNET]: ERROR - trying to generate a latent state without context in a model that uses context"
		if ctx_enc_summary is not None:
			feed_dict[self.ops.ctx_enc_summary] = ctx_enc_summary							
		for cat in self.params["property_order"]["inp"]:
			cat_params = self.params["modelling_properties"][cat] 
			if cat_params["include"]:
				feed_dict[getattr(self.ops, "y_" + cat_params["key"] + "_")] = [data[:, cat_params["indices"][0]: cat_params["indices"][1]]]
		return self.session.run(run_ops, feed_dict)
		
	'''
		Takes a specification for generation and executes it. Generation can be done either with an input context or an empty context only containing a
		start token. The model can be started with a specific input to generate the latent space vector, or this vector can be sampled. The latter takes
		place at all subsequent generated bases except for the first one, if encode is set to yes. The data object is expected to contain whatever is needed
		to generate a context and / or an input representation for the latent space. Generation is always done in batches of size 1. All structures containing
		music data are assumed to be 2-dimensional with (steps, features) here. The first dimension, batch, which is always one when generating is added
		to all data right before feeding it to the graph.
		
		Params:
		-------
		sequence_base: int
			The length of a sequence base, or unit. Should ideally be what the modelled was trained for and if this model models the property offset, the 
			default value for sequence_base is the length of the sequence_base multiplied by the size of the input length. This is the unit the model was trained
			for and should thus be the output at every iteration now, however, other values are permitted. If offset is not used in the model, a default size
			unit is assigned as offset to all output and a sequence base should then be the number of such offsets that fit in the size of the input that this
			model was trained to generate. That is, if offset is not modelled, the sequence base is seen as the number of timesteps to generate.
			
		sequence_length: int
			The number of sequence units to generate. These will be stored separately in the returned array.

		ctx: numpy array
			If this model uses a context and ctx is not None, then ctx is assumed to be a 2D numpy matrix with the data representation of a context to use
			for the first sequence base of generation. If the model uses context but ctx is None, then an initial context will be generated consisting of a
			start token only. If this model does not use a context, then ctx will be ignored. Non-modelled properties will be ignored no matter.
			
		inp: tuple of (bool, numpy array)
			If this model uses a vae and inp is not None, then inp should be a tuple where the first element indicates whether to run the second element through
			the model before considering it as a latent space samle. If this boolean is False, then the second element is considered to be the latent space
			vector without further processing. If this model uses a vae but inp is None, the latent space vector will be sampled from a normal unit distribution
			and if this model does not use a vae, this argument will be ignored. Non-modelled properties will be ignored no matter.

		Returns: array of numpy arrays
		--------
			An array with 2D numpy arrays of full scale with all properties not modelled set to all zeros. Each index in the array corresponds to one
			sequence base and so the output contains sequence_length elements if this model does not use a context, otherwise sequence_length + 1 since
			the initial context is forwarded into the output as well.
	'''
	# generate several time steps, data is a tuple with either only context keys set (s, o etc...) or also y keys such as y_o, y_i which is then to be encoded
	def generate(self, ds, sequence_base, sequence_length, ctx = None, inp = None, use_teacher_forcing = False):
		assert(not use_teacher_forcing or inp is not None), "[MAHLERNET]: ERROR - must supply a concrete input to use teacher forcing during generation"
		if use_teacher_forcing:
			sequence_length = 1 # only allow reconstruction of input and that's all
		base = np.zeros([100, self.params["num_features"]], dtype = np.uint8) # stores the generated content of the current base
		generated_bases = 0
		out = []
		qs = {}
		probs = {"o": [], "d": [], "p": [], "i": []} # perplexity probabilities
		empty_init_ctx = False
		if self.params["model"]["ctx"]:
			if ctx is None or (isinstance(ctx, str) and ctx == "START"):
				if isinstance(ctx, str) and ctx == "START":										 
					ctx = self.empty_context(True)
				else:									 
					ctx = self.empty_context(False)
				empty_init_ctx = True
			else: # context was already supplied to the input of this function, might be empty or impossible to include in output anyway 
				if not ctx[0] or (len(ctx[1]) == 1 and ctx[1][0][self.params["modelling_properties"]["special_tokens"]["indices"][0]] == 1):
					empty_init_ctx = True
				else:
					out += [ctx[1]] # one sequence base at every index generated, starting with the initial context
		for active_part in ["active_pitches", "active_instruments"]:
			cat_params = self.params["modelling_properties"][active_part]
			if cat_params["include"]:
				qs[cat_params["key"]] = []	
		# generate desired output
		tries = 1
		while generated_bases < sequence_length:
			decoder_state = None
			last_time = 0
			base_index = 1 # time step in currently generated sequence_base
			if self.params["model"]["ctx"]:
				if ctx is not None: # true at most the first time
					ctx_enc_summary = self.context(ctx[1]) if ctx[0] else ctx[1] # if ctx[0] is false, ctx[1] contains the ready ctx vector
				else:
					ctx_enc_summary = self.context(out[generated_bases - 1] if empty_init_ctx else out[generated_bases])
			else:
				ctx_enc_summary = None
			if self.params["model"]["vae"]:
				if inp is not None: # true at most the first time
					z = self.latent(inp[1], ctx_enc_summary) if inp[0] else inp[1] # if inp[0] is false, inp[1] contains the ready z vector
				else:
					z = np.random.normal(0.0, 1.0, [1, self.params["vae"]["z_dim"]])
			else:
				z = None
			while not use_teacher_forcing or (use_teacher_forcing and base_index <= inp[1].shape[0]): # last time step is base_index - 1, current one to be predicted is base_index
				# the above condition could just as well be replaced with "while True" but since we know beforehand how many time steps to generate with teacher forcing, this is an optimization so that we don't have to generate
				# one extra time step of content for every time-forced reconstruction sample that we run, especially since they are often many to the number when running n_recon on a dataset.
				if (base_index + 1) >= base.shape[0]: # + 1 because the active pitches and instruments sets the upcoming vector, not the current one
					next_base = np.zeros([base.shape[0] * 2, base.shape[1]], dtype = np.uint8) # double the steps dimension
					next_base[: base.shape[0], :] = base
					base = next_base
				# generate a sequence base using the same context)
				done = False
				
				while not done: 
					# make sure to generate durations and offsets that are allowed in the current duration set, this is because we might be using a 2/4 duration set which disallows longer durations used in 4/4
					done = True
					if use_teacher_forcing:
						if base_index > 1:
							std_input = inp[1][base_index - 2: base_index - 1]
						else:
							std_input = base[base_index - 1: base_index]
						tf_input = inp[1][base_index - 1: base_index]
					else:
						std_input = base[base_index - 1: base_index]
						tf_input = None
					next_decoder_state, next_time, pred, prob = self.predict(std_input, decoder_state, last_time, z, ctx_enc_summary, tf_input, tries)
					if self.params["modelling_properties"]["offset"]["include"]:
						offset_index = np.argmax(pred["o"][0])
						if "li" + str(offset_index) not in ds.inverse_mappings or ds.inverse_mappings["li" + str(offset_index)].duration > sequence_base:
							done = False
					if self.params["modelling_properties"]["duration"]["include"]:
						duration_index = np.argmax(pred["d"][0])
						if "li" + str(duration_index) not in ds.inverse_mappings:
							done = False
				if (next_time < sequence_base and not use_teacher_forcing) or (use_teacher_forcing and base_index <= inp[1].shape[0]): # acceptable durations and within the time frame of the given base
					last_time = next_time
					decoder_state = next_decoder_state
					# remove instruments and pitches that will have stopped sounding at the next time step to process (which we are currently preparing)
					for active_part in ["active_pitches", "active_instruments"]:
						cat_params = self.params["modelling_properties"][active_part]
						if cat_params["include"]:
							while len(qs[cat_params["key"]]) > 0 and qs[cat_params["key"]][0][0] <= last_time: # [0][0] holds ending time
								base[base_index][self.params["modelling_properties"][active_part]["indices"][0] + qs[cat_params["key"]][0][1]] -= 1
								qs[cat_params["key"]] = qs[cat_params["key"]][1:]						
					o_zero = True if (self.params["modelling_properties"]["offset"]["include"] and pred["o"][0][0]) else False
					d_zero = True if (self.params["modelling_properties"]["duration"]["include"] and pred["d"][0][0]) else False
					assert(use_teacher_forcing or not self.params["modelling_properties"]["duration"]["next_step_reduction"] or not (o_zero and d_zero)), "Forbidden generation, generated 0 duration and 0 offset in same timestep" + str(pred) + str(prob)
					skip_p_i = True if (self.params["modelling_properties"]["duration"]["include"] and d_zero) else False
					# decide what to take with us from this computation (if offset and duration is zero, the rest is not valid)
					to_process = ["offset", "beats", "duration"] if skip_p_i else ["offset", "beats", "duration", "pitch", "instrument"]
					for active_part, key in [("active_pitches", "p"), ("active_instruments", "i")]:
						cat_params = self.params["modelling_properties"][active_part]
						if cat_params["include"]:
							# copy currernt time step to next
							base[base_index + 1][cat_params["indices"][0]: cat_params["indices"][1]] = base[base_index][cat_params["indices"][0]: cat_params["indices"][1]]
							if not skip_p_i:
								index = np.argmax(pred[key][0])
								duration = ds.inverse_mappings["li" + str(duration_index)].duration
								qs[cat_params["key"]] += [(last_time + duration, index)]
								qs[cat_params["key"]].sort()
								base[base_index + 1][cat_params["indices"][0] + index] += 1
						# make the representation for the current time step (will not be used more) binary, just for formalia, this doesn't really matter
						base[base_index][self.params["modelling_properties"][active_part]["indices"][0]: self.params["modelling_properties"][active_part]["indices"][1]] = (base[base_index][self.params["modelling_properties"][active_part]["indices"][0]: self.params["modelling_properties"][active_part]["indices"][1]] > 0).astype(np.uint8)
					for cat in to_process:
						cat_params = self.params["modelling_properties"][cat] 
						if cat_params["include"]:
							base[base_index, cat_params["indices"][0]: cat_params["indices"][1]] = pred[cat_params["key"]][0]
							if use_teacher_forcing and cat != "beats":
								if np.max(tf_input[0][cat_params["indices"][0]: cat_params["indices"][1]]) > 0.0:
									prediction_prob = np.sum(prob[cat_params["key"]][0] * tf_input[0][cat_params["indices"][0]: cat_params["indices"][1]])
									probs[cat_params["key"]] += [prediction_prob]
					base_index += 1
				else: # the current predicted step starts later than the current base, don't include it (or maybe include it as a seed???? future?)
					break
			if base_index > 1:
				tries = 1
				out += [np.copy(base[1: base_index])]
				base.fill(0)
				generated_bases += 1
				inp = None # if nothing was generated (that is, its time fell out of the allowed window), we will used the same ctx and z and try again
				ctx = None
			else:
				tries += 1
		if use_teacher_forcing:
			for key in probs:
				assert(len(probs[key]) == 0 or np.min(probs[key]) > 0.0), "[MAHLERNET]: ERROR - probability <= 0.0 in probability vector to calculate perplexity " + key + " " + str(probs)
			probs = {k: (np.sum(np.log(v)) / len(v)) if len(v) > 0 else "N/A" for k, v in probs.items()} # individual categories
			probs["tot"] = [v for k, v in probs.items() if v != "N/A"]
			probs["tot"] = (np.sum(probs["tot"]) / len(probs["tot"])) if len(probs["tot"]) > 0 else "N/A" # all categories
			return out, probs
		else:
			return out		
		
	def generate_sample(self, ctx = None, inp = None, meter = 4, length = 1, triplet = False, use_teacher_forcing = False):
		assert(meter < 5 and meter > 1), "Illegal meter, must be in [2, 4]"
		ds = self.params["generation"]["default_duration_sets"][meter]
		if self.params["generation"]["sequence_base"] == "bar":
			sequence_base_length = ds.bar_duration.duration 
		else:
			sequence_base_length = ds.basic_set[self.params["generation"]["sequence_base"]].duration
		if self.params["modelling_properties"]["offset"]["include"]:
			steps = sequence_base_length * self.params["generation"]["inp_length"] # steps is really time here, not steps
			step_unit = None
		else:
			step_unit = self.params["generation"]["default_triplet_duration"] if triplet else self.params["generation"]["default_duplet_duration"] 
			# + 1 because in generation, we increase time and THEN check if it is LESS than the full length of base to generate = if we want to generate 6 time steps and steps is 6, this means that after having generated
			# data on time steps 0 (check if 1 < 6), 1 (2 < 6), 2 (3 < 6), 3 (4 < 6) and 4 (5 < 6) we have generated 5 units and won't generate the 6th since 6 is not less than 6 and time steps are added to the result
			# AFTER this check to conform to sampling when the length of the sampled time steps is unknown
			steps = (sequence_base_length // ds.basic_set[step_unit].duration) + 1
		if use_teacher_forcing:
			music, perplexity = self.generate(ds, steps, length, ctx, inp, use_teacher_forcing)
		else:
			music = self.generate(ds, steps, length, ctx, inp, use_teacher_forcing)
		data_repr = []
		time = 0
		for seq_base in music:
			data_repr += [(seq_base, 0, ds, time, time + sequence_base_length)]
			time += sequence_base_length
		if use_teacher_forcing:
			return data_repr, step_unit, perplexity
		else:
			return data_repr, step_unit
			
