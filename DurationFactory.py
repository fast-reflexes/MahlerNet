import numpy as np

class DurationFactory:

	class AbstractBasicDuration:

		def __init__(self, num, denom, index, name, triplet_based, duplet_based):
			self.fraction_num = num
			self.fraction_denom = denom
			self.distributed_index = index
			self.name = name
			self.triplet_based = triplet_based
			self.duplet_based = duplet_based
			
		def __str__(self):
			return "(" + self.name + ", " + str(self.distributed_index) + "," + ("T" if self.is_duplet_based() else "F") + ("T" if self.is_triplet_based() else "F") + ")"
			
		def get_distributed_index(self):
			return self.distributed_index
			
		def to_basic_duration(self, tpb):
			return DurationFactory.BasicDuration(int((tpb * self.fraction_num) / self.fraction_denom), self) # must be int by itself without casting
			
		def is_triplet_based(self):
			return self.triplet_based
			
		def is_duplet_based(self):
			return self.duplet_based
			
	class BasicDuration:

		def __init__(self, duration, abstract):
			self.duration = duration
			self.abstract = abstract
			
		def get_distributed_index(self):
			return self.abstract.get_distributed_index()
			
		def __str__(self):
			return str(self.abstract) + " - " + str(self.duration)
			
		def is_triplet_based(self):
			return self.abstract.is_triplet_based()
			
		def is_duplet_based(self):
			return self.abstract.is_duple_based()
			
	class AbstractDuration:

		def __init__(self, partials, frequency_class, pos_info):
			self.partials = partials
			self.one_hot_index = -1
			self.frequency_class = frequency_class
			self.duration_ratio = 0.0
			self.marker_offset = pos_info[0]
			self.marker = pos_info[1]
			for basic_duration in self.partials:
				self.duration_ratio += float(basic_duration.fraction_num / basic_duration.fraction_denom)
			
		def to_duration(self, basic_set):
			out_partials = []
			duration = 0
			for partial in self.partials:
				next_duration = basic_set[partial.name]
				duration += next_duration.duration
				out_partials += [next_duration]
			return DurationFactory.Duration(out_partials, duration, self, basic_set)
			
		def is_triplet_based(self):
			return len([p for p in self.partials if not p.is_triplet_based()]) == 0
			
		def is_duplet_based(self):
			return not self.is_triplet_based()
			
		def contains(self, basic_duration_name):
			return len([partial for partial in self.partials if partial.name == basic_duration_name]) > 0
			
	class Duration:
		
		def __init__(self, partials, total_duration, abstract, basic_set):
			self.partials = partials
			self.duration = total_duration
			self.abstract = abstract
			if self.abstract.marker == 0:
				self.required_starting_position = 0
			elif self.abstract.marker == -1:
				self.required_starting_position = -1
			else:
				# the marker is given as a multiple of quarters and the offset as a negative value before this marker
				self.required_starting_position = basic_set[self.abstract.marker].duration - basic_set[self.abstract.marker_offset].duration
				self.marker_position = basic_set[self.abstract.marker].duration
				
			
		def get_one_hot_index(self):
			return self.abstract.one_hot_index
			
		def get_frequency_class(self):
			return self.abstract.frequency_class
			
		def is_basic(self):
			return len(self.partials) == 1

		def is_compound(self):
			return not self.is_basic()

		def get_basic_component(self):
			return self.partials[len(self.partials) - 1]

		def get_number_of_partials(self):
			return len(self.partials)

		def is_triplet_based(self):
			return self.abstract.is_triplet_based()
			
		def is_duplet_based(self):
			return self.abstract.is_duplet_based()

		def contains(self, basic_duration_name):
			return self.abstract.contains(basic_duration_name)
			
		def __str__(self):
			st = str(self.get_one_hot_index()) + "/" +  ". " + str(self.duration) + " = "
			i = 0
			for partial in self.partials:
				if i > 0:
					st += ", "
				st += partial.__str__()
				i += 1
			st += ", is basic? " + str(self.is_basic())
			return st
			
	class AbstractDurationSet:
	
		def __init__(self, ts_num, ts_denom, ratios, bar_duration_ratio):
			assert((ts_num, ts_denom)  in [(2, 2), (3, 2), (4, 2), (2, 4), (3, 4), (4, 4), (2, 8), (3, 8), (4, 8), (6, 8), (6, 4), (9, 8), (9, 4), (12, 4), (12, 8)]),\
				"Illegal time signature detected, can only handle time signatures in triple and duple time"
			self.ts_num = ts_num
			self.ts_denom = ts_denom
			self.multiplier = 1.0 # assume quarter-based time signature in duple time or eighth-based in triple time
			
			if ts_num % 3 != 0 or ts_num == 3: # duple time
				if ts_denom == 2:
					self.multiplier = 2.0 # half-note-based time signature
				elif ts_denom == 8:
					self.multiplier = 0.5 # eighth-based time signature
				self.abstract_durations = list(filter(lambda x: x.duration_ratio < (bar_duration_ratio + 1.0), ratios))
			else: # triplet time
				self.multiplier = 1.5 # the value of a duplet quarter will increase by 1.5, thus interpreting actual quarters as triplet quarters as desired
				if ts_denom == 4:
					self.multiplier *= 2.0 # quarter-based triplet time maps to half note based duple time
				self.abstract_durations = list(filter(lambda x: x.duration_ratio < (bar_duration_ratio + 1.0) and x.is_triplet_based(), ratios))
			self.bar_duration_index = [i for i, abstract_duration in enumerate(self.abstract_durations) if abstract_duration.duration_ratio == bar_duration_ratio]
			self.bar_duration_index = self.bar_duration_index[0]
			
		def __eq__(self, other):
			return self.ts_num == other.ts_num and self.ts_denom == other.ts_denom

	class DurationSet:
	
		def __init__(self, tpb, basic_set, abstract_duration_set):
			self.tpb = tpb
			self.abstract_duration_set = abstract_duration_set
			self.basic_set = basic_set
			# create the actual length in ticks for each of the allowed units, add a 0 length for durations that should be excluded
			self.durations = list(map(lambda x: x.to_duration(self.basic_set), self.abstract_duration_set.abstract_durations))
			self.bar_duration = self.durations[self.abstract_duration_set.bar_duration_index]
			self.max_duration = self.durations[len(self.durations) - 1]
			self.inverse_mappings = {}
			for duration in self.durations:
				self.inverse_mappings['l' + str(duration.duration)] = duration
				self.inverse_mappings['li' + str(duration.get_one_hot_index())] = duration
			for duration_name in self.basic_set:
				self.inverse_mappings['b' + str(self.basic_set[duration_name].duration)] = self.basic_set[duration_name]
				self.inverse_mappings['bi' + str(self.basic_set[duration_name].get_distributed_index())] = self.basic_set[duration_name]

		def __eq__(self, other):
			return self.tpb == other.tpb and self.abstract_duration_set == other.abstract_duration_set
			
		def filter_durations(self, start, threshold, excludes = []):
		
			def contains_excludes(duration, excludes):
				for exclude in excludes:
					if duration.contains(exclude):
						return True
				return False
				
			return list(filter(lambda x: not contains_excludes(x, excludes) and x.required_starting_position >= 0 and (x.required_starting_position == 0 or abs(x.required_starting_position - (start % x.marker_position)) <= threshold), self.durations))
			
	def __init__(self):
		# all subdivisions in 4/4 by duple and triple meter, eash row is a quarter: the lowest is the first quarter, second lowest the second etc... the upper row is a whole note and a 
		# little extra to account for syncopations into the next beat
		# #D = note, #. = dotted note, #T = triplet note, 1, 2, 4, 8, 16, 32 = whole, half, quarter, eighth, sixteenth and 32nd note, #-# two tied notes

		ABD = self.AbstractBasicDuration
		AD = self.AbstractDuration
		T = True
		F = False
		
		# basic abstract units, that all other units are constructed from, in fractions of the beat unit (quarter per default)
		# extended to the same denominator for easier indexing (indexed from smallest to largest)
		abds = {	'd1': ABD(96, 24, 10, "d1", T, T), 	'd2': ABD(48, 24, 8, "d2", T, T), 	'd4': ABD(24, 24, 6, "d4", T, T), 	'd8': ABD(12, 24, 4, "d8", T, T), 	'd16': ABD(6, 24, 2, "d16", F, T), 	'd32': ABD(3, 24, 0, "d32", F, T),\
					't1': ABD(64, 24, 9, "t1", T, F), 	't2': ABD(32, 24, 7, "t2", T, F),	't4': ABD(16, 24, 5, "t4", T, F),	't8': ABD(8, 24, 3, "t8", T, F), 	't16': ABD(4, 24, 1, "t16", T, F),\
					'none': ABD(0, 1, -1, "none", T, T)}
		self.abds = abds
		self.sz_basic_set = len(abds) - 1

		# each abstract duration has a number indicating the likeliness of its appearence, 0 is most common and are basic durations. Dotted notes are class 1
		# Two notes which are not a single dotted note is class 2, class 3 is 3 notes which corresponds to double dotted note and class 4 are 3 notes which
		# do not correspond to a single double dotted note
		self.duple_ratios = list([ \
			AD([abds['d32'], abds['d16'], abds['d8'], abds['d1']], 5, (-1, -1)),\
			AD([abds['d16'], abds['d8'], abds['d1']], 4, (-1, -1)),\
			AD([abds['d32'], abds['d8'], abds['d1']], 4, (-1, -1)),\
			AD([abds['d8'], abds['d1']], 2, ("d8", "d1")),\
			AD([abds['d32'], abds['d16'], abds['d1']], 4, (-1, -1)),\
			AD([abds['d16'], abds['d1']], 2, ("d16", "d1")),\
			AD([abds['d32'], abds['d1']], 2, ("d32", "d1")),\
			\
			AD([abds['d1']], 0, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d8'], abds['d4'], abds['d2']], 5, (-1, -1)),\
			AD([abds['d16'], abds['d8'], abds['d4'], abds['d2']], 5, ("d16", "d8")),\
			AD([abds['d32'], abds['d8'], abds['d4'], abds['d2']], 5, (0, 0)),\
			AD([abds['d8'], abds['d4'], abds['d2']], 3, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d4'], abds['d2']], 5, (-1, -1)),\
			AD([abds['d16'], abds['d4'], abds['d2']], 4, ("d16", "d4")),\
			AD([abds['d32'], abds['d4'], abds['d2']], 4, ("d32", "d4")),\
			\
			AD([abds['d4'], abds['d2']], 1, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d8'], abds['d2']], 5, (-1, -1)),\
			AD([abds['d16'], abds['d8'], abds['d2']], 4, ("d16", "d8")),\
			AD([abds['d32'], abds['d8'], abds['d2']], 4, (-1, -1)),\
			AD([abds['d8'], abds['d2']], 2, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d2']], 4, (-1, -1)),\
			AD([abds['d16'], abds['d2']], 2, ("d16", "d2")),\
			AD([abds['d32'], abds['d2']], 2, ("d32", "d2")),\
			\
			AD([abds['d2']], 0, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d8'], abds['d4']], 5, (-1, -1)),\
			AD([abds['d16'], abds['d8'], abds['d4']], 3, ("d16", "d8")),\
			AD([abds['d32'], abds['d8'], abds['d4']], 4, (-1, -1)),\
			AD([abds['d8'], abds['d4']], 1, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d4']], 4, (-1, -1)),\
			AD([abds['d16'], abds['d4']], 2, ("d16", "d4")),\
			AD([abds['d32'], abds['d4']], 2, ("d32", "d4")),\
			\
			AD([abds['d4']], 0, (0, 0)),\
			AD([abds['d32'], abds['d16'], abds['d8']], 3, (-1, -1)),\
			AD([abds['d16'], abds['d8']], 1, (0, 0)),\
			AD([abds['d32'], abds['d8']], 2, ("d32", "d8")),\
			AD([abds['d8']], 0, (0, 0)),\
			AD([abds['d32'], abds['d16']], 1, (0, 0)),\
			AD([abds['d16']], 0, (0, 0)),\
			AD([abds['d32']], 0, (0, 0)),\
			\
			AD([abds['none']], 0, (0, 0))
		])
		
		self.triple_ratios = list([ \
			AD([abds['t16'], abds['t4'], abds['t2'], abds['t1']], 12, (-1, -1)),\
			AD([abds['t4'], abds['t2'], abds['t1']], 12, (0, 0)),\
			AD([abds['t8'], abds['t2'], abds['t1']], 8, ("t8", "d4")),\
			AD([abds['t16'], abds['t2'], abds['t1']], 8, ("t16", "d4")),\
			\
			AD([abds['t16'], abds['t8'], abds['t4'], abds['t1']], 14, (-1, -1)),\
			AD([abds['t8'], abds['t4'], abds['t1']], 12, ("t8", "d8")),\
			AD([abds['t4'], abds['t1']], 12, ("t8", "d4")),\
			AD([abds['t16'], abds['t8'], abds['t1']], 12, ("t16", "d4")),\
			\
			AD([abds['t16'], abds['t1']], 12, ("t16", "d8")),\
			AD([abds['t1']], 8, ("t16", "d8")),\
			AD([abds['t8'], abds['t4'], abds['t2']], 8, (0, 0)),\
			AD([abds['t16'], abds['t4'], abds['t2']], 8, ("t16", "d2")),\
			\
			AD([abds['t16'], abds['t8'], abds['t2']], 10, ("t16", "d2")),\
			AD([abds['t8'], abds['t2']], 6, ("none", "t8")),\
			AD([abds['t8'], abds['d4']], 3, (0, 0)),\
			AD([abds['t16'], abds['t8'], abds['t4']], 8, ("t16", "t8")),\
			\
			AD([abds['t16'], abds['t4']], 8, ("t16", "t8")),\
			AD([abds['t4']], 1, (0, 0)),\
			AD([abds['t8']], 1, (0, 0)),\
			AD([abds['t16']], 1, (0, 0))\
		])	
		
		self.all_ratios = (self.duple_ratios + self.triple_ratios)
		self.all_ratios.sort(key = lambda x: x.duration_ratio)
		for i, ratio in enumerate(self.all_ratios):
			ratio.one_hot_index = i
		self.all_ratios = np.array(self.all_ratios)
		# self.all_ratios.reverse() # order by duration in ascending order
		self.sz_compound_set = len(self.all_ratios)
		
		self.map = {2: {}, 4: {}, 8: {}} # indexed FIRST by denominator, THEN by nominator
		self.map[2][2] = self.AbstractDurationSet(2, 2, self.all_ratios, 2.0)
		self.map[2][3] = self.AbstractDurationSet(3, 2, self.all_ratios, 3.0)
		self.map[2][4] = self.AbstractDurationSet(4, 2, self.all_ratios, 4.0)
		self.map[4][2] = self.AbstractDurationSet(2, 4, self.all_ratios, 2.0)
		self.map[4][3] = self.AbstractDurationSet(3, 4, self.all_ratios, 3.0)
		self.map[4][4] = self.AbstractDurationSet(4, 4, self.all_ratios, 4.0)
		self.map[8][2] = self.AbstractDurationSet(2, 8, self.all_ratios, 2.0)
		self.map[8][3] = self.AbstractDurationSet(3, 8, self.all_ratios, 3.0)
		self.map[8][4] = self.AbstractDurationSet(4, 8, self.all_ratios, 4.0)

		
		self.map[4][6] = self.AbstractDurationSet(6, 4, self.all_ratios, 2.0) # 23
		self.map[4][9] = self.AbstractDurationSet(9, 4, self.all_ratios, 3.0) # 34
		self.map[4][12] = self.AbstractDurationSet(12, 4, self.all_ratios, 4.0) # 41
		self.map[8][6] = self.AbstractDurationSet(6, 8, self.all_ratios, 2.0)
		self.map[8][9] = self.AbstractDurationSet(9, 8, self.all_ratios, 3.0)
		self.map[8][12] = self.AbstractDurationSet(12, 8, self.all_ratios, 4.0)
		
	def get_duration_set(self, ts_num, ts_denom, tpb):
		if ts_denom in self.map:
			if (ts_num % 12) == 0:
				ts_num = 12
			elif (ts_num % 9) == 0:
				ts_num = 9
			elif (ts_num % 6) == 0:
				ts_num = 6
			elif (ts_num % 4) == 0:
				ts_num = 4
			elif (ts_num % 2) == 0:
				ts_num = 2
			if ts_num in self.map[ts_denom]:
				ads = self.map[ts_denom][ts_num]
				tpb *= ads.multiplier
				tpb = int(tpb)
				bds = {}
				# calculate athe values of the basic set once and for all
				for note in self.abds:
					bds[note] = self.abds[note].to_basic_duration(tpb)
				
				# return the duration set
				return self.DurationSet(tpb, bds, ads)
			else:
				raise ValueError("Illegal time signature numerator (", ts_num, " / ", ts_denom, ") only 2, 4, 6, 8, 9 and 12 allowed")
		else:
			raise ValueError("Illegal time signature denominator (", ts_num, " / ", ts_denom, ") only 2, 4 and 8 allowed")
