from MidiPreprocessor import MidiPreprocessor
import pickle, os, re, os.path, math, mido, gc
from random import shuffle, seed
import numpy as np
from utilities import print_divider


class DataProcessor:

	def __init__(self, use_randomness = True):
		if not use_randomness:
			seed(1)
		self.mp = MidiPreprocessor()
		self.training_set = None
		self.validation_set = None
		self.input_data = None
		self.total = None
		self.max_context_length = None
		self.max_input_length = None
		self.GENERATOR_CACHE_SIZE = 10 # the number of simultaneous bins to randomly choose from while generating training data
		# the max size of the data files with accumulated samples with the same input length
		self.CACHE_THRESHOLD = 4096 # 8 * 512 which is the largest assumed batch size
		
	'''files can be the NAME Of a directory, a single file name or a tuple with file name and adjustments or an array of filenames or tuples of file names 
	and adjustments takes one or several files and outputs its data representation in a given folder based on input name
	
		@param files = 	a list of filenames or tuples with filenames and adjustments (can be combined) or lists thereof, or the name of a folder in which all 
						files are to be processed
		@param save_dir = the name of the directory at the current place in the file tree where the output is to be stored
		@param root_dir = the name of the directory, if any, at the current place in the file tree from which input is to be fetched
	'''
	
	def setup_dirs(self, root_dir, input_dir, suffix, save_dir = "", files = 0): # files = 0 implies that the call is not interested in files output
		save_dir = os.path.join(root_dir, save_dir)
		input_dir = os.path.join(root_dir, input_dir)
		regex = "(" + ("|".join(suffix)) + ")$" if isinstance(suffix, list) else suffix + "$"
		if files is None: # files is the name of a directory
			saved_path = os.getcwd()
			files = []
			os.chdir(input_dir)
			gen = os.walk(".")
			for path, ds, fs in gen:
				#files += list(map(lambda file: os.path.join(path, file), list(filter(lambdafiles))
				files += list(map(lambda file: os.path.join(path, file), list(filter(lambda file: re.search(regex, file, flags=re.IGNORECASE) is not None, fs))))
			#files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))] # after this files do not contain full search path
			os.chdir(saved_path)
		if len(save_dir) > 0 and not os.path.exists(save_dir): # create the save dir if it doesn't exist
			os.makedirs(save_dir)
		return input_dir, save_dir, files
		
	def midi_to_data(self, files = None, root_dir = "", input_dir = "input", save_dir = "data", redo_existing = False):
		input_dir, save_dir, files = self.setup_dirs(root_dir, input_dir, ["mid", "midi"], save_dir, files)
		skipped_files = []
		pitch_statistics = np.zeros((1, self.mp.MAX_PITCHES), dtype="int")
		
		# process the list of file names or tuples or lists of filenames or tuples (that are to be considered a single input)
		for f in files:	
			try:
				if not isinstance(f, list): # not a concatenation of several files, put it in a list for conform further treatment
					f = [f]
				for i, file in enumerate(f): # add the root_dir part of the path to each file name
					if isinstance(file, tuple):
						# this input file has 4 additional arguments to forward to the preprocessor concerning its treatment
						f[i] = (os.path.join(input_dir, file[0]), file[1], file[2], file[3], file[4])
					else:
						# this file can be used with default treatment (no additional parameters)
						f[i] = os.path.join(input_dir, f[i])
						
				filename = map(lambda x: os.path.basename(x[0]) if isinstance(x, tuple) else os.path.basename(x), f) # list with names in case of tuple, otherwise trivially a list of one name
				filename = map(lambda x: x[:-4] if len(x) > 4 and re.fullmatch(".mid", x[-4:], flags=re.IGNORECASE) is not None else x, filename) # remove suffix
				filename = "_".join(filename) # join with "_" in between
				outfile = filename + ".pickle"
				path = os.path.join(save_dir, outfile)
				
				exists = os.path.isfile(path) 
				if not exists or redo_existing:
					if exists:
						os.remove(path)
					dr, no, ps = self.mp.midi_to_data(f)
					with open(path, mode = "xb") as file:
						pickle.dump(dr, file) # dr is a data representation and is a list of 5-tuples on the form (vectors, statistics, duration_set, start time, end time)
					del dr, no
					pitch_statistics += ps
				else:
					print("[MIDI_TO_DATA]: skipping already converted file", f)
			except IOError as ie:
				print("[MIDI_TO_DATA]: skipping file", f, " due to error: ",ie)
				skipped_files += [(f, str(ie))]
			except ValueError as ve:
				print("[MIDI_TO_DATA]: skipping file", f, " due to error: ", ve)
				skipped_files += [(f, str(ve))]
			except KeyError as ke:
				print("[MIDI_TO_DATA]: skipping file", f, " due to error: ", ke)
				skipped_files += [(f, str(ke))]
			except AssertionError as ae:
				print("[MIDI_TO_DATA]: skipping file", f, " due to error: ", ae)
				skipped_files += [(f, str(ae))]			
			except mido.midifiles.meta.KeySignatureError as kse:
				print("[MIDI_TO_DATA]: skipping file", f, " due to error: ", kse)
				skipped_files += [(f, str(kse))]	
			except EOFError as eoe:
				print("[MIDI_TO_DATA]: skipping file", f, " due to error: ", eoe)
				skipped_files += [(f, str(eoe))]	
		print("[MIDI_TO_DATA]:  listing", len(skipped_files), "skipped files")
		for file, reason in skipped_files:
			print("[MIDI_TO_DATA]:        File", file, "-", reason)
		print("[MIDI_TO_DATA]: done converting MIDI to data, totally skipped", len(skipped_files), "files")
		print("[MIDI_TO_DATA]: printing pitch statistics from conversion")
		print(pitch_statistics)

	def data_to_midi(self, files = None, root_dir = "", input_dir = "data", save_dir = "midi"):
		input_dir, save_dir, files = self.setup_dirs(root_dir, input_dir, ".pickle", save_dir, files)

		for f in files:
			if re.fullmatch(".pickle", f[-7:], flags=re.IGNORECASE) is not None:
				print("[DATA_TO_MIDI]: Processing file", f)
				path = os.path.join(input_dir, f)
				dr_f = open(path, mode = "rb")
				dr = pickle.load(dr_f)
				dr_f.close()
				midi_files = self.mp.data_to_midi(dr, True)
				filename = f[:-7]
				len_midis = len(midi_files)
				for i, file in enumerate(midi_files):
					if len_midis > 1:
						path = os.path.join(save_dir, filename + "_" + str(i) + '.mid')
					else:
						path = os.path.join(save_dir, filename + '.mid')
					if os.path.isfile(path):
						os.remove(path)
					file.save(path)

	''' generates a list of context / input patterns from a directory of files
		@param files = the name of a directory in which all files are to be turned into sequences, or a list with filenames of files to turn into sequences
	'''
	def data_generator(self, file, root_dir = "", input_dir = "data", sequence_base = "bar", ctx_length = 1, inp_length = 1):
		input_dir, _, _ = self.setup_dirs(root_dir, input_dir, ".pickle")
		
		path = os.path.join(input_dir, file)
		dr_f = open(path, mode = "rb")
		# load music file in data representation on the form [5-tuples ...] where 5-tuple is (vectors, statistics, duration set, start, end) and 
		# represents a discontinous piece of a music piece
		dr = pickle.load(dr_f) 
		dr_f.close()
		len_chunk = len(dr) # the number of discontiguous range in this piece of music
		for i, chunk in enumerate(dr): 
			gen = self.mp.sequence_generator(chunk, sequence_base, ctx_length, inp_length)
			for (ctx, inp) in gen:
				yield (ctx, inp)
		
	''' generates a list of context / input patterns from a directory of files
		@param files = the name of a directory in which all files are to be turned into sequences, or a list with filenames of files to turn into sequences
	'''
	def data_to_sequences(self, files = None, sequence_base = "bar", ctx_length = 1, inp_length = 1, root_dir = "", input_dir = "data", save_dir = "seq"):
		input_dir, save_dir, files = self.setup_dirs(root_dir, input_dir, ".pickle", save_dir, files)
		
		skipped_files = []
		index = []
		processed_samples = 0
		cache = [[]] # the cache is 1-indexed and so the 0'th bin representes data of length 0 (of which there exists none)
		stored = [0] # the number of files stored on disk so far for the sequence length at the corresponding index in the cache
		max_inp_len = 0 # keep track of the largest ctx and inp length encountered during this process
		max_ctx_len = 0
		
		# evicts a bin from the cash, storing its full contents on disk, filing a record for it in the index
		def evict(inp_len):
			nonlocal cache, stored, index, processed_samples
			outfile = str(inp_len) + "_" + str(stored[inp_len]) + ".pickle"
			stored[inp_len] += 1
			path = os.path.join(save_dir, outfile)
			if os.path.isfile(path):
				os.remove(path)
			file = open(path, mode = "xb")
			pickle.dump(cache[inp_len], file) # files are simply just a list of tuples on the form (ctx, inp)
			file.close()
			num_samples = len(cache[inp_len]) # the number of samples in the filed bin
			print("[DATA_TO_SEQ]: dumped file with", num_samples, "samples of length", inp_len, "to disk")
			processed_samples += num_samples
			cache[inp_len] = [] # reset cache
			# file the filename and the size of the max input in the index (input is more important than ctx)
			index += [(inp_len, num_samples, outfile)]
			
		tot_files = len(files)
		for fn, f in enumerate(files):
			print("[DATA_TO_SEQ]: processing file", fn, "/", tot_files, ":", f)
			if re.fullmatch(".pickle", f[-7:], flags=re.IGNORECASE) is not None: # only process files with .pickle ending
				path = os.path.join(input_dir, f)
				dr_f = open(path, mode = "rb")
				# load music file in data representation on the form [5-tuples ...] where 5-tuple is (vectors, statistics, duration set, start, end) and 
				# represents a discontinous piece of a music piece
				dr = pickle.load(dr_f) 
				dr_f.close()
				len_chunk = len(dr) # the number of discontiguous range in this piece of music
				for i, chunk in enumerate(dr): 
					try:
						gen = self.mp.sequence_generator(chunk, sequence_base, ctx_length, inp_length)
						for sample_num, (ctx, inp) in enumerate(gen): # generates new numpy arrays in two dimensions (timesteps, features) that do not need to be copied here
							max_ctx_len = max(max_ctx_len, len(ctx))
							if len(inp) > max_inp_len: # encountered a longer input sequence than seen before, expand the cache so that the 1-indexing works as intended
								delta = len(inp) - max_inp_len
								add = [[] for _ in range(delta)]
								cache += add
								stored += ([0] * delta) # works because integers are not objects and refer to different data always
								max_inp_len = len(inp)
							cache[len(inp)] += [(ctx, inp, sample_num, f)]
							if len(cache[len(inp)]) >= self.CACHE_THRESHOLD:
								evict(len(inp))
					except AssertionError as ae:
						print("[DATA_TO_SEQ]: Skipping file '", f, "'due to failed assertion:", ae)
						skipped_files += [(f, str(ae))]		
		print("[DATA_TO_SEQ]: done processing files, please hold on while writing non-terminated files from cache to disk. This may take a while ...")
		for i in range(len(cache)): # empty the remaining cache since we are done processing files
			if len(cache[i]) > 0:
				evict(i)
		print("[DATA_TO_SEQ]: cache is now empty, writing summary index to disk ...")
		if len(index) > 0: # save the index
			index = (max_ctx_len, max_inp_len, processed_samples, sequence_base, ctx_length, inp_length, index)
			path = os.path.join(save_dir, "index")
			if os.path.isfile(path):
				os.remove(path)
			file = open(path, mode = "xb")
			pickle.dump(index, file)
			file.close()
		print("[DATA_TO_SEQ]: listing", len(skipped_files), "skipped files")
		for file, reason in skipped_files:
			print("[DATA_TO_SEQ]:        File", file, "-", reason)
		print("[DATA_TO_SEQ]: done converting data to sequences, totally processed " + str(processed_samples) + " samples")
		print("[DATA_TO_SEQ]: totally skipped " + str(len(skipped_files)) + " files")
		
	''' an important purpose of this generator function is to balance a number os aspect:
		For efficiency:
			
			E1-Data with similar number of time steps must be in the same batches, otherwise, the treatment of masked out timesteps due to single long sequences
			will yield unnecessary works
			
			E2-Data must be provided to the training algorithm at a decent speed which implies that the number of disk reads should not be exaggerated.
			
		For performance:
			
			P1-The different batches during an epoch must come in different orders every time, otherwise the model might end up in a certain training pattern
			that doesn't take it to new and hopefully better places of the loss function surface
			
			P2-Even though it doesn't matter which order the data in a batch has, one should try to mix the contents of the batches so that batches contain
			different members between epochs
	
		Restrictions:
			
			R1-All training data cannot be kept in memory at once since this takes too much RAM
			
		Solution: The solution is first to sort all the data durng preprocessing by the length of the input so that similar length pattern end up together (E1).
		For the sake of R1 we can't store all these patterns in the same file and for the sake of E2, we should not store a single pattern in a single file.
		For the sake of P1, we must generate the data in an order that is different between runs and so we need to find a decent file size to store file in.
		We then randomly order these files and take data from them continuously. Now, if every pattern would be in a file on its own, then this would ensure
		P1 but E2 would be violated. If we did the opposite R1 would be violated and also both P1 and P2 since no ordering of files would be possible. Let's say
		we find a middle point, where a certain amount of P1 would be fulfilled by randomly order files. If we had very large files, then the random ordering
		wouldn't do much good but with very small files, P2 would not be fulfilled. To ensure P2, we need to generate a number of batches from the same data
		so that we can ensure that shuffling it beforehand will lead to batches with slightly differing content. On the other hand, the smaller the files, the
		greater impact will randomly ordering the files have on P1. We may then choose to have a cache of several streams of training data and pick randomly
		from them as well, then P1 would be ensured both by random file ordering and by random cache / bin choices during generation. Furthermore, P2 would be
		ensured since the files would be large enough, relative to batch size, so that an internal shuffling leads to batches of different content. E1 would be
		fulfilled since the files are bigger than the batch size (even a multiple) and so mostly, batches would contain same size lengths and in the few cases
		where content from different bins need to be used simulatenously, it is still about very few times after all. E2 would be sufficiently fulfilled and
		this is not believed to be a very hard restrictions. In worst case, we can multi-thread and prepare the next batch while the former batch is run by the
		training algorithm. Finally, R1 is in the danger since to ensure most of the previous, we would have to have quite much data in memory at once. So, if
		memory becomes an issue, increase the number of reads, that is, decrease the size of files.
		
		Given a batch size of 32-512, 512 * 8 is thought to give large enough files so that internal shuffling ensures a mixture between bathes from a certain
		seq length. Still, the files would be sufficiently small so that shuffling file names would ensure different batch orderings and if that wouldn't be enough,
		a cache of size 10 makes it highly unlikely that we pick several batches in a row from the same bin.Of course, all parameters are subjects to testing
	'''
	def sequences_to_training_data_generator(self, batch_size, validation = False, root_dir = "", input_dir = "seq", random_ordering = True, max_limit = None):
		if self.input_files is None or self.validation_set is None or self.training_set is None or self.total is None or self.max_context_length is None or self.max_input_length is None:
			self.setup_training(root_dir, input_dir, ".pickle")
		
		input_dir, _, _ = self.setup_dirs(root_dir, input_dir, ".pickle")
		
		# pick what set of file indices to use and shuffle their order to improve on the random ordering between batches during training
		if validation:
			if random_ordering:
				shuffle(self.validation_set) # randomly shuffle files whenever a generator is created
			indices = self.validation_set 
		else:
			if random_ordering:
				shuffle(self.training_set)
			indices = self.training_set
		context_data = np.zeros((batch_size, self.max_context_length, self.mp.timestep_sz), dtype=np.uint8)
		context_lengths = np.zeros((batch_size), dtype = np.uint32)
		input_data = np.zeros((batch_size, self.max_input_length, self.mp.timestep_sz), dtype=np.uint8)
		input_lengths = np.zeros((batch_size), dtype = np.uint32)
		sample_nums = np.zeros((batch_size), dtype = np.uint32)
		file_names = [None] * batch_size
		processed_samples = 0
		processed_batches = 0
		cache = [[] for _ in range(self.GENERATOR_CACHE_SIZE)]
		bin_index = [0] * self.GENERATOR_CACHE_SIZE
		input_index = 0
		exhausted = False
		while not exhausted:
			if random_ordering:
				src_bin = np.random.randint(self.GENERATOR_CACHE_SIZE)
			else:
				src_bin = 0
			batch_index = 0
			max_ctx_len = 0
			max_inp_len = 0
			while batch_index < batch_size:
				if bin_index[src_bin] == len(cache[src_bin]): # fill this bin with the next file
					cache[src_bin] = []
					bin_index[src_bin] = 0
					if input_index < len(indices):
						f = self.input_files[indices[input_index]][2]# input_index has tuples with (input max length, number of samples, list with (ctx, inp) pairs)
						path = os.path.join(input_dir, f)
						dr_f = open(path, mode = "rb")
						dr = pickle.load(dr_f) # tuples with (ctx, inp) patterns
						if random_ordering:
							shuffle(dr) # in place shuffling of elements
						dr_f.close()
						cache[src_bin] = dr
						input_index += 1
					else: # no more new files to pick from
						src_bin = 0
						while src_bin < self.GENERATOR_CACHE_SIZE and bin_index[src_bin] == len(cache[src_bin]):
							src_bin += 1
						if src_bin == self.GENERATOR_CACHE_SIZE: # no more data in the cache at all and no more files to read into the cache
							exhausted = True
							break
				seq_pair = cache[src_bin][bin_index[src_bin]] # seq pair is a tuple (ctx, inp, sample_num, file) where sample_num is the unit number in the file f (used for tracking purposes)
				good = True
				if good and (max_limit is None or len(seq_pair[1]) < max_limit):
					sample_nums[batch_index] = seq_pair[2]
					file_names[batch_index] = seq_pair[3]
					context_lengths[batch_index] = len(seq_pair[0])
					max_ctx_len = max(max_ctx_len, len(seq_pair[0]))
					context_data[batch_index][0: len(seq_pair[0])] = seq_pair[0]
					max_inp_len = max(max_inp_len, len(seq_pair[1]))
					
					input_lengths[batch_index] = len(seq_pair[1])
					input_data[batch_index][0: len(seq_pair[1])] = seq_pair[1]
					batch_index += 1
					bin_index[src_bin] += 1
					processed_samples += 1
				else:
					bin_index[src_bin] += 1	
			if batch_index > 0:
				yield (context_data[: batch_index, :max_ctx_len], context_lengths[: batch_index], input_data[: batch_index, :max_inp_len], input_lengths[: batch_index], sample_nums[: batch_index], file_names[: batch_index])
				processed_batches += 1
				context_data.fill(0)
				context_lengths.fill(0)
				input_data.fill(0)
				input_lengths.fill(0)
				sample_nums.fill(0)
				file_names = [None] * batch_size
		print_divider("[GENERATOR]: Generator exhausted! Yielded " + str(processed_batches) + " batches with a total of " + str(processed_samples) + " samples", " ")

	# the main purpose of this method is to setup the use of a generator for training data. However, not everything in this process needs to be done several
	# times and since a training generator should provide data multiple times, these things are abstracted away into this method to make it more efficient.
	def setup_training(self, root_dir = "", input_dir = "seq", validation_set_ratio = 0.1, random_ordering = True):
		input_dir, _, _ = self.setup_dirs(root_dir, input_dir, ".pickle")
		
		# index is a tuple with (max_ctx, max_inp, tot_samples, file_index)
		# each index in file index holds (input length, number of samples, file name)
		path = os.path.join(input_dir, "index")
		index_f = open(path, mode = "rb")
		index_tuple = pickle.load(index_f)
		index_f.close()
		self.inp_length = index_tuple[5]
		self.ctx_length = index_tuple[4]
		self.sequence_base = index_tuple[3]
		self.input_files = index_tuple[6]
		self.total = index_tuple[2]
		self.max_input_length = index_tuple[1]
		self.max_context_length = index_tuple[0]
		file_indices = [i for i in range(len(self.input_files))] # all indices into the index array ordered on input max length and then ctx max length
		num_validation_samples = math.ceil(float(self.total) * validation_set_ratio)
		if random_ordering:
			shuffle(file_indices)
		breakpoint = 0
		acc_v_samples = 0
		while breakpoint < len(file_indices) and acc_v_samples < num_validation_samples:
			acc_v_samples += self.input_files[file_indices[breakpoint]][1]
			breakpoint += 1
		self.sz_validation_set = acc_v_samples
		self.sz_training_set = self.total - acc_v_samples
		self.validation_set = file_indices[: breakpoint]
		self.training_set = file_indices[breakpoint:]
		self.features = self.mp.timestep_sz
		print("[DATA_PROC]: Set up a training session with",  self.sz_training_set, "samples for training and", self.sz_validation_set, "for validation")
				