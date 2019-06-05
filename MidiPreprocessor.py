import mido, os
import numpy as np
from functools import reduce
import DurationFactory as df
from Events import Note, TimeSignature, Tempo, Program
from Instruments import Instruments
from utilities import print_divider

class MidiPreprocessor:

	def __init__(self):

		self.inst = Instruments()
		self.dur = df.DurationFactory()
		self.default_duration_set = self.dur.get_duration_set(4, 4, 480)
		self.default_common_duration_set = self.default_duration_set
		self.default_triple_duration_set = self.dur.get_duration_set(3, 4, 480)
		self.default_duple_duration_set = self.dur.get_duration_set(2, 4, 480)
		self.default_beat_units = ["d8"] # only basic units allowed
		self.default_bar_unit_duration = self.default_duration_set.bar_duration.duration
		self.default_beats_vector = []
		for unit in self.default_beat_units:
			self.default_beats_vector += [time for time in range(0, self.default_bar_unit_duration, self.default_duration_set.basic_set[unit].duration)]
		self.default_beats_vector = np.unique(np.array(self.default_beats_vector, dtype = np.int32)) # sorts and removes non-unique
		self.USE_DISTRIBUTED_DURATIONS = False
		self.NUM_BASIC_DURATIONS = self.dur.sz_basic_set
		self.NUM_COMPOUND_DURATIONS = self.dur.sz_compound_set
		self.NUM_INSTRUMENTS = self.inst.sz_instrument_set
		self.NUM_USED_PITCHES = 96
		self.NUM_SPECIAL_TOKENS = 1
		self.LOW_NOTE_THRESHOLD = 17 # inclusive
		self.HIGH_NOTE_THRESHOLD = 112 # inclusive
		self.NUM_PITCHES = self.HIGH_NOTE_THRESHOLD - self.LOW_NOTE_THRESHOLD + 1
		self.MAX_PITCHES = 128
		self.NUM_CHANNELS = 16
		self.NUM_BEATS = len(self.default_beats_vector)
		self.OCTAVE = 12
		self.P_INSTRUMENTS = False
		self.P_DURATIONS = False
		self.P_STREAMING = False
		self.P_TOKENIZATION = False
		self.P_ADJUSTMENT = 0
		self.P_FINALIZATION = False
		self.P_TO_DATA = False
		self.P_FROM_DATA = False
		self.P_TO_MIDI = False
		self.P_GENERATOR = False
		self.DEFAULT_MSPQ = 500000 # default tempo in microseconds per quarter note
		self.MAX_FAILED_ADJUSTMENTS = 0.05 # max ratio of failed adjustments in process_tokens
		self.FAST_THRESHOLD = 300000 # tempo in microseconds per quarter note under which the tempo is considered FAST
		self.SLOW_THRESHOLD = 900000 # same but over which the tempo is SLOW
		
		if self.USE_DISTRIBUTED_DURATIONS:
			self.num_durations = self.NUM_BASIC_DURATIONS
			self.default_durations = np.zeros([self.num_durations], dtype = np.int32)
			for i in range(self.num_durations):
				self.default_durations[i] = self.default_duration_set.inverse_mappings["bi" + str(i)].duration
		else:
			self.num_durations = self.NUM_COMPOUND_DURATIONS	
			self.default_durations = np.zeros([self.num_durations], dtype = np.int32)
			for i in range(self.num_durations):
				self.default_durations[i] = self.default_duration_set.inverse_mappings["li" + str(i)].duration
			
		# indices
		self.time_a = 0
		self.beats_a = self.time_a + self.num_durations
		self.inp_a = self.beats_a + self.NUM_BEATS
		self.dur_a = self.inp_a + self.NUM_PITCHES
		self.instr_a = self.dur_a + self.num_durations
		#			offset     beats	 pitch		duration	 instr        start + end    active notes      active instruments
		# layout: |---------|----------|-----------|------------|-------------|----------------|--------------------|-------------|
		#			
		
		self.final_tokens_a = self.instr_a + self.NUM_INSTRUMENTS
		self.act_notes_a = self.final_tokens_a + self.NUM_SPECIAL_TOKENS
		self.act_inst_a = self.act_notes_a + self.NUM_PITCHES
		
		self.START_TOKEN = self.final_tokens_a 
		self.END_TOKEN = self.START_TOKEN + 1
		
		# time since last event, the pitch, instrument and duration of the current input and a set of currently active pitches and instruments
		self.timestep_sz = self.num_durations + self.NUM_PITCHES + self.NUM_INSTRUMENTS + self.num_durations + self.NUM_PITCHES + self.NUM_INSTRUMENTS + self.NUM_BEATS + self.NUM_SPECIAL_TOKENS #last to for start and end
	
	def tracks_to_stream(self, midi, adjustments, factor, limit):
		p = self.P_STREAMING
		tokens = []
		tracks_no = 0	
		for i, track in enumerate(midi.tracks):
			tracks_no += 1
			ticks = 0
			program = 0
			if p:
				print(i, track[1])
			ex = [32, 33, 22] + list(range(2, 12))
			ex = []
			if i not in ex:
				for j, msg in enumerate(track):
					ticks += msg.time
					type = msg.dict()['type']
					if type == "program_change":
						program = int(msg.dict()['program'])
					# to shift all notes in the input a given interval, uncomment this
					#if type == "note_on" or type == "note_off":
					#	msg = msg.copy(note = msg.dict()["note"] + 6)							 	
					if type != "end_of_track":
						offset = 0
						for adj_tuple in adjustments:
							if ticks >= adj_tuple[0]:
								offset += adj_tuple[1]
							else:
								break
						tokens += [(i, round((ticks + offset) * factor), msg, program, -1)]		
		tokens.sort(key = lambda x: (x[1], int(not((x[2].dict()['type'] == "note_on" and int(x[2].dict()['velocity']) == 0) or x[2].dict()['type'] == "note_off")))) # sort by overall time
		# discard songs with ridiculously long pauses in them (probably due to malformatting)
		last = 0
		init = False
		for t in tokens:
			if t[2].dict()['type'] == "note_on":
				assert(not init or (t[1] - last) <= limit), "[STREAM]: ERROR - input offset between two note events too large, was " + str(t[1] - last) + " and limit is " + str(limit)
				init = True
				last = t[1]
		return tokens, tracks_no
		
	''' Tokenizes a MIDI file by going through all events and storing four types of events: tempo changes, instrument changes, time signature changes and note events (both on and off.
	The output is a list of events where each note event is represented by its starting time and how long it is in ticks. Some notes may be invalid but still returned as part of the 
	output list. No further adjustments are done to events at this time. All output events are timed with absolute timing (not relative as in the MIDI format itself). In the output
	all time signatures, instrument changes or tempo changes occur before possible notes starting at the same tick. Nevertheless, any original ordering within these three categories are
	preserved. Also, pitches starting at the same tick are sorted by pitch in ascending order. Order within Also returns a list of the indices in tokens which constitues new times 
	signatures so that further processing may be divided into chunks based on time signature.
	'''
	def tokenize(self, midi, inst, quarter, num_tracks, start = None, end = None, limit = None, internal_instrument = None, neglected_ts = []):
		p = self.P_TOKENIZATION
		#p = True
		tokens = [] # a list of message tokens
		last_started = [0] * num_tracks # holds when the last started note on a specific track was started
		time_signatures = [] # a list of the indices in tokens which constitutes new time signatures, if any
		initiated = np.zeros((self.NUM_CHANNELS, self.MAX_PITCHES), dtype='int') # a map over the channels and which pitches are active, e.g. has been started at a specific time (its values is not -1)
		initiated.fill(-1) # all notes are unused when we beging
		instruments = np.zeros((self.NUM_CHANNELS), dtype='int') # current instrument at each input channel, default is piano
		preceding_notes = [[] for _ in range(num_tracks)]
		
		stat_pitches = np.zeros((self.MAX_PITCHES), dtype = 'int')
		
		time = 0
		tempo = self.DEFAULT_MSPQ # default tempo corresponding to 120 bpm if no other tempo has been given, needed to convert events given in seconds to ticks
		
		def file_note(index, channel, pitch, track_number):
			tokens[index].duration = time - tokens[index].start
			
			# shorten this note if a new note started in the same track right before this one (probably held the old note slightly too long)
			if tokens[index].duration < quarter and ((time - last_started[track_number]) < (tokens[index].duration / 2)):
				tokens[index].duration -= (time - last_started[track_number])
				
			initiated[channel][pitch] = -1 # signal that there is no ongoing note for this pitch any more
			
			# save the just ended notes
			if len(preceding_notes[track_number]) == 0:
				# no notes bookmarked, add this one to the queue
				preceding_notes[track_number] += [tokens[index]]
			else:
				next = []
				for previous_note in preceding_notes[track_number]:
					previous_end_time = previous_note.start + previous_note.duration
					current_end_time = tokens[index].start + tokens[index].duration
					if current_end_time - previous_end_time < quarter / 8: # was 16, allows prolongation of notes that ended up to a 16th not before the current one
						next += [previous_note]
				next += [tokens[index]]
				preceding_notes[track_number] = next
				
			stat_pitches[pitch] += 1
			
		for token in midi: # setting up this iterator internally takes quite a long time because all tracks are merged into a single ordered iterator if events
			time = token[1]
			if (start is None or time > start) and (end is None or time < end) and (limit is None or len(tokens) < limit):
				if p:
					print("TOKEN: ", token[0], token[1], token[2], token[3])
				track_number = token[0]
				msg = token[2]
				instr = token[3]
				
				type = msg.dict()['type']
				if type == "note_off" or (type == "note_on" and int(msg.dict()['velocity']) == 0): # note off event
					pitch = int(msg.dict()['note'])
					while pitch <= self.LOW_NOTE_THRESHOLD:
						pitch += self.OCTAVE
					while pitch > self.HIGH_NOTE_THRESHOLD:
						pitch -= self.OCTAVE						
					channel = int(msg.dict()['channel'])
					index = initiated[channel][pitch]
					if index >= 0: # this note off event actually turns off a note, otherwise, we have an error and should ignore this event since it shuts off an unstarted note
						assert(tokens[index].channel == channel and tokens[index].pitch == pitch), "Inconsistency - Filed note has incorrect pitch or channel"
						file_note(index, channel, pitch, track_number)
						if p:
							print("MIDI note off event: ", msg)
							print("--->VALID END: ", tokens[index])	
					else:
						if p:
							print("--->INVALID END: ", msg)
				elif type == 'note_on':
					pitch = int(msg.dict()['note'])
					while pitch <= self.LOW_NOTE_THRESHOLD:
						pitch += self.OCTAVE
					while pitch > self.HIGH_NOTE_THRESHOLD:
						pitch -= self.OCTAVE					
					channel = int(msg.dict()['channel'])
					if instr == 0: # if regular piano, check if something else is registered in the channel bank
						instr = instruments[channel]
						if p:
							print("FOUND INSTR: ", instr)
					if channel != 9 and inst.is_valid_instrument(inst.get_internal_mapping(instr)) and (internal_instrument == None or internal_instrument == inst.get_internal_mapping(instr)): # ignore drum channel as well as instruments that are chosen to be ignored
						index = len(tokens)
						if p:
							print("Track", token[0], "start", token[1], "msg", token[2], "instr", token[3])
						# prolong the preceding notes in the same track if they ended relatively late compared to when this note starts (probably result of staccato)
						indices_previous_notes = preceding_notes[track_number]
						for previous_note in indices_previous_notes:
							if previous_note.duration < (quarter * 2):
								if abs(previous_note.start - last_started[track_number]) < (quarter / 8):
									# was 16, allows prolongation of notes that ended up to a 16th not before the current one
									# we only want to process notes in the last cluster of notes starter
									delta = time - (previous_note.start + previous_note.duration) 
									if delta < quarter:
										if p:
											print("Prolonged: ", previous_note, "to", previous_note.duration + delta)
											print(time, delta, previous_note.start, previous_note.duration)
										previous_note.duration += delta
									else:
										# prolong it for as much as possible depending on position
										for duration in [quarter, quarter / 2, quarter / 4, quarter / 3, quarter / 6]:
											mod = previous_note.start % duration
											if abs(mod - duration) < mod:
												mod -= duration
											if abs(mod) < (quarter / 16): # was 16
												if p:
													print("Prolonged by quantization: ", previous_note, "to", duration)
												previous_note.duration = duration
												break
										if p:
											print("NOT Prolonged: ", previous_note)
							else:
								if p:
									print("NOT Prolonged TOO LONG: ", previous_note)
						preceding_notes[track_number] = []
						# place this note slightly earlier if there is an even in the same track indicating that a note was JUST started
						if time - last_started[track_number] < (quarter / 16):
							tokens += [Note(last_started[track_number], pitch, channel, inst.get_internal_mapping(instr), inst, track_number, tempo)]
						else:
							tokens += [Note(time, pitch, channel, inst.get_internal_mapping(instr), inst, track_number, tempo)]
							last_started[track_number] = time
						if p:
							print("Added token to index ", len(tokens) - 1)
							print("MIDI note on event: ", msg)
							print("--->VALID START: ", tokens[len(tokens) - 1])
						while index > 0 and tokens[index - 1].start == tokens[index].start and isinstance(tokens[index - 1], Note) and (tokens[index].pitch < tokens[index -1].pitch):
							tokens[index - 1], tokens[index] = tokens[index], tokens[index - 1]
							if p:
								print("Switching places", index, index - 1)
								print("    -->", tokens[index])
								print("    -->", tokens[index - 1])
							if tokens[index].duration == 0:
								initiated[tokens[index].channel][tokens[index].pitch] = index # propagate
							index -= 1
						if initiated[channel][pitch] >= 0: # error need to invalidate a previous unstopped note (this note should be excluded from the entire processing)
							# or save the previous note, assuming its length is up to this note IF its length doesn't exceed some value
							if time - tokens[initiated[channel][pitch]].start < 4000:
								file_note(initiated[channel][pitch], channel, pitch, tokens[initiated[channel][pitch]].track)
							else:
								assert(tokens[initiated[channel][pitch]].channel == channel and tokens[initiated[channel][pitch]].pitch == pitch), "Inconsistency - Filed note has incorrect pitch or channel"
								tokens[initiated[channel][pitch]].invalidate()
						initiated[channel][pitch] = index # file the newly started note
					else:
						pass
				elif type == "set_tempo" or type == "program_change":				
					if type == "program_change":
						next_instrument = int(msg.dict()['program'])
						channel = int(msg.dict()['channel'])
						instruments[channel] = next_instrument
						#tokens += [Program(time, channel, next_instrument)] # save for testing only, need not be included in the final product
					else:
						tempo = int(msg.dict()['tempo']) # change tempo for all subsequent timing calculations				
				elif type == "time_signature":
					ts_num = int(msg.dict()['numerator'])
					ts_denom = int(msg.dict()['denominator'])
					if (ts_num, ts_denom) not in neglected_ts and (ts_num, '*') not in neglected_ts and ('*', ts_denom) not in neglected_ts:
						if p:
							print("MIDI time signature event: ", msg, time)
						index = len(tokens)
						tokens += [TimeSignature(time,ts_num, ts_denom)]
						while index > 0 and isinstance(tokens[index - 1], Note) and tokens[index - 1].start == tokens[index].start:
							tokens[index - 1], tokens[index] = tokens[index], tokens[index - 1]
							if tokens[index].duration == 0:
								initiated[tokens[index].channel][tokens[index].pitch] = index # propagate if unfinished
							index -= 1
						time_signatures += [index]	
						# do not allow active notes across the line of a time signature change
						for index_to_erase in initiated[initiated >= 0]: # note without an ending event
							tokens[index_to_erase].invalidate()					
						initiated.fill(-1) # all notes are unused when we beging
						for track_no in range(len(last_started)): # reset all places where a note might be tempted to be placed earlier into the previous time signature
							if last_started[track_no] < time:
								last_started[track_no] = 0
			elif limit is not None and len(tokens) >= limit:
				break
						
		# clean up unterminated notes and invalidate them in the output
		for i in initiated[initiated >= 0]: # note without an ending event
			tokens[i].invalidate()
		for i in range(1, len(tokens)):
			if p:
				print("TEST", tokens[i - 1], tokens[i])
			# remove duplicate notes (same note, same instrument, same starting time)
			if isinstance(tokens[i - 1], Note) and isinstance(tokens[i], Note) and tokens[i - 1] == tokens[i] and tokens[i - 1].is_valid():
				tokens[i].invalidate()
				if p:
					print("REMOVED!")
		return tokens, time_signatures, stat_pitches
		
	'''Processes ranges of unquantized note events in the same time signature and performs a sort of normalization. The starting point is the notion of a number of ticks per quarter beat,
	which is adjusted based on time signature: if the denominator in the time signature is anything else than a quarter beat, the beat unit used is divided in half (if 8) or double
	(2) since these beat units are the ones used in the music. Each time signature has a number of fractions of the base beat unit that are allowed when quantizing (or rounding) each duration
	to its closest allowed event. The allowed durations are duple-based or triplet-based. Furthermore, if a triplet-based time signature is used (6/8, 9/8 and 12/8), the beat unit is 
	multiplied with 1.5 to translate the processing into triplet-based meter. As an example, if an eight note is used as a beat unit, two eight notes in 6/8 could be replaced by three
	triplet eights instead. In that case, we would start working in duple rhythms in triplet time and thus allow for triplet rhyhms in triplet time. This would require us to have a lot
	of more possible durations and it would also be impractical since triplet rhythms in triplet time is uncommon. Instead, we CHOOSE to see an eights note as a dotted eight note and
	if it actually is only an eighth note, this is possible to express as well. The starting time and length of each duration is normalized and notes that are longer than a bar are kept
	that way with the rest being quantized as usual.'''
	def process_tokens(self, tokens, ds, tracks_no, triplet_ts, range_start, offset = 0): # ds = duration set
		p = self.P_ADJUSTMENT
		i = 0
		processed_valid_notes = 0 # keep track of how many valid notes are in this subrange, if zero, return an empty vector instead...
		eighth = int(ds.basic_set['d8'].duration)
		status = [(0, 0, 0, 0)] * tracks_no # for each track, indicate type (triplet, duplet or both) and when the last note ended
		failed_adjustments = 0
		successful_adjustments = 0

		# find the first actual note and adjust initial time offset to this point
		'''while i < len(tokens) and type(tokens[i]) is not Note:
			tokens[i].start = 0 # reset start times for intermediate events
			i += 1'''
		if len(tokens) > 0 and i < len(tokens): # there are items in the list and there is at least one note event
			
			''' Searches for a range of he most suitable, in terms of distance, quantized duration given a duration by means of binary search'''
			def get_quantization_candidates(duration, limit, durations):
				a, b = 0, len(durations) - 1
				while(b - a > 1):
					pivot = int(a + (b - a) / 2)
					if duration > durations[pivot].duration:
						a = pivot
					else:
						b = pivot
					if abs(duration - durations[a].duration) <= abs(duration - durations[b].duration):
						center = a
					else:
						center = b					
				if duration < limit.duration: # duration is shorter than a quarter, just return two values
					ret = [durations[a], durations[b]]
				else:
					a = center
					while a > 0 and abs(a - center) <= 2: # assume value at the lowest position
						a -= 1
					b = center + 1
					while b < len(durations) and abs(b - center) <= 2: # assume value at the lowest position
						b += 1
					ret = durations[a: b]
				return ret, durations[center]
					
			def sieve_candidates(duration_triple, candidates, starting_time):
				nonlocal successful_adjustments, failed_adjustments
				scores = [0] * len(candidates) # scores to minimize for the candidate to win
				best_index = -1
				adjustments = [0] * len(candidates)
				partial_duration = duration_triple[0]
				partial_forced_alignment = duration_triple[1]
				partial_start = duration_triple[2]
				
				for i, duration in enumerate(candidates):
					# account for SIGNED quantization: if quantization forces duration to be LONGER this is POSITIVE, if SHORTER this is NEGATIVE, this is scaled
					# by how common the note in question is (classes 0-4, translated into 1-5 for multiplicative reasons)
					initial_distance = duration.duration - partial_duration # pure distance
					
					if partial_forced_alignment == 0 or duration.duration == 0:
						# ignore if aligned by start of bar = foced alignment = 0 since this event always starts at the beginning of a bar line by default, no alignment needed
						# also ignore adjustment if it is recommended to erase this note since this is equivalent to no realignment
						adjustment = 0 # already 0 so unnecessary, only for clarity
						resulting_distance = 0
						
					elif partial_forced_alignment == 2: # duration MUST be aligned to the end of this bar
						if partial_duration > ds.bar_duration.duration:
							needed_start_in_current_bar = (ds.bar_duration.duration - (duration.duration - ds.bar_duration.duration)) 
							adjustment = needed_start_in_current_bar - ((partial_start - ds.bar_duration.duration) % ds.bar_duration.duration)
						else:
							needed_start_in_current_bar = (ds.bar_duration.duration - duration.duration) 
							adjustment =  needed_start_in_current_bar - (partial_start % ds.bar_duration.duration)
						# candidate duration start first MINUS actual start of this duration: adjustment is NEGATIVE if the chosen candidate forces alignment
						# EARLIER in time, effectively INCREASING the size of the desired duration, if the chosen candidate forces the actual duration
						# to start later, it effectively REDUCES the size of the chosen duration and the sign is POSITIVE
						
						# now, a POSITIVE quantization score indicates a LONGER note, and adding an adjustment that is POSITIVE indicates that the note should be delayed
						# in time further. This creates a note that both starts later than the original note AND is longer, which makes it ring even further longer afterin
						# the original note ended. if the alignment is NEGATIVE, the extra added duration is added to the beginning of the note, making the prolongation
						# of the note more likely since we are probably dealing with a note that was struck slightly too late. The opposite applies as well when shrunken and
						# delayed.
						
						resulting_distance = initial_distance + adjustment
						# distance is now updated with information that either corroborates the growing / shrinking or contradicts it
						# for example, if the note was first increased in size and then we ALSO have to adjust it to LATER in time, then both values will be positive
						# reflecting that we are farther away from the original note than after the first duration adjustment. On the other hand, if the note is increased
						# in size but moved earlier in time, then it will perhaps be a note that was pushed down slightly too late and so its likeliness is increased.
						
					elif partial_forced_alignment == 1: 
						# no particular alignment, place the different sub-partials (of the candidates) at their correct place and see which results in the 
						# smallest need for alignment, find the BEST adjustment
						base = duration.get_basic_component().duration # this is the largest sub-component
						# try to place it first
						adjustment = -(partial_start % base) # NEGATIVE, forces note to GROW in size (potentially keeping the same end time)
						if (base + adjustment) < abs(adjustment):
							adjustment += base # POSITIVE, forces note to SHRINK in size (potentially keeping the same end time)
						best = adjustment # keep the best so far
						
						if len(duration.partials) > 1:
							# now place each and one of the other notes before (single note) and see if this results in a better alignment for the largest note
							for j in range(len(duration.partials) - 1):
								start = partial_start + duration.partials[j].duration
								adjustment = -(start % base)
								if (base + adjustment) < abs(adjustment):
									adjustment += base
								if abs(adjustment) < abs(best):
									best = adjustment
									
							# now, if there are 3 partials, try to place the two smaller ones first
							if len(duration.partials) == 3:
								start = partial_start + duration.partials[0].duration + duration.partials[1].duration
								adjustment = -(start % base)
								if (base + adjustment) < abs(adjustment):
									adjustment += base
								if abs(adjustment) < abs(best):
									best = adjustment
						elif base >= ds.basic_set['t4'].duration: # this duration consists of only one partial, attempt to place it to the closest halve of it as well
							half = int(base / 2)
							start = partial_start
							adjustment = -(start % half)
							if (half + adjustment) < abs(adjustment):
								adjustment += half
							if abs(adjustment) < abs(best):
								best = adjustment

						adjustment = best
						resulting_distance = initial_distance + adjustment

					# process the actual score by the help of the variables initial_distance, resulting_distance and adjustment, the last being the most important
					if initial_distance < 0: # all cases where a note is being deleted totally ends up here
						# had to shrink the original duration to fit this suggestion
						if resulting_distance < initial_distance: # adjustment placed this earlier in time, potentially making it an even worse fit to the original
							distance_score = round((1.0 + (resulting_distance / initial_distance)) * (initial_distance / 2.0))
						else: # adjustment place this later in time, corroborating shortening the note, perhaps a note that was hit slightly to early?
							if resulting_distance > 0: # the adjustment made up for the entire decrease in duration and even more than that
								distance_score = round(initial_distance / 2.0) # still negative
								distance_score -= round((resulting_distance / initial_distance) * (initial_distance / 2.0)) # result is positive before minus sign
							elif duration.duration == 0:
								distance_score = initial_distance
							else: # the adjustment made up for part of the decrease in duration, but not all of it
								distance_score = round((1.0 + (resulting_distance / initial_distance)) * (initial_distance / 2.0))
					elif initial_distance > 0:
						# had to grow the original note to fit this suggestion
						if resulting_distance > initial_distance: # adjustment placed this suggestion even later in time, making it an even worse fit
							distance_score = round((1.0 + (resulting_distance / initial_distance)) * (initial_distance / 2.0))
						else: # adjustment placed this note earlier in time, perhaps making up for the increased duration, perhaps a note that was hit too late?
							if resulting_distance < 0: # the adjustment made up for the entire increase in note size and even more than that
								distance_score = round(initial_distance / 2.0)
								distance_score -= round((resulting_distance / initial_distance) * (initial_distance / 2.0)) # result is negative before minus sign
							else: # the adjustment made up for part of the increase in note size but note all of it
								distance_score = round((1.0 + (resulting_distance / initial_distance)) * (initial_distance / 2.0))
					else:
						# if initial distance is 0, there is nothing to scale and the only penalty is the adjustment term
						distance_score = 0
						
					if abs(adjustment) >= 1.5 * (ds.basic_set['d16'].duration): # tuneable parameter, the size which it is NEVER reasonable to move the start of a note
						scores[i] = 1000000 # rule this suggestion out of the picture
					elif starting_time + adjustment < 0:
						print("Event start,adjustment, range start", starting_time, adjustment, range_start) if p >= 2 else 0
						scores[i] = 1000000 # rule this suggestion out since it implies moving the event BEFORE the starting time of the current time signature start
					else:
						scores[i] = abs(distance_score * (duration.get_frequency_class() + 1)) + (duration.get_frequency_class() * 15)
						scores[i] += abs(adjustment * 4) # see above for explanation
						adjustments[i] = adjustment
						if best_index == -1 or scores[i] < scores[best_index]: # update the best found so far
							best_index = i
						elif scores[i] == scores[best_index]:
							if candidates[best_index].duration == 0:
								best_index = i
							elif candidates[i].is_basic() and not candidates[best_index].is_basic():
								best_index = i
							elif candidates[best_index].is_triplet_based() and not candidates[i].is_triplet_based():
								best_index = i
					print("       SCORE: ", scores[i], "(", initial_distance, resulting_distance, adjustment, ")", duration) if p >= 2 else 0
				if scores[best_index] == 1000000:
					failed_adjustments += 1
					return candidates[best_index], adjustments[best_index], False
				else:
					successful_adjustments += 1
					return candidates[best_index], adjustments[best_index], True
					
				
			''' Aligns start of events with the granularity chosen for the data representation, that is, equivalent to the smallest unit representable. Also quantizes the resulting notes'''
			def quantize_and_align(event):
				# process everything, both Notes, non-Notes and invalid notes.. we only have start notes at this time
				print("Event BEFORE adjustment to the current time signature:") if p >= 2 else 0
				print(event) if p >= 2 else 0
				event.start = event.start + offset - range_start # offset accounts for adjusment to current CHUNK and range_start to adjustment within current subrange
				print("Event AFTER adjustment to the current time signature:") if p >= 2 else 0
				print(event) if p >= 2 else 0
				assert(event.start >= 0), "Discrepancy, found an event with a negative starting time relative to the range start: " + str(event)
				to_process = []
				success = True # to determine whether to invalidate the event after processing or not (due to at least one sub partial being impossible to find a good candidate for)
				if type(event) is Note:
					if event.is_valid(): # only process valid notes
						duration = event.duration 
						saved_duration = 0 # excluded full bars
						out_duration = 0
						out_adjustment = 0
						if duration > ds.max_duration.duration: # longer than the longest permitted single note
							pre = ds.bar_duration.duration - (event.start % ds.bar_duration.duration) # exactly what is left until bar line
							duration -= pre
							saved_duration = int(duration / ds.bar_duration.duration) * ds.bar_duration.duration
							if (pre + ds.bar_duration.duration) <= ds.max_duration.duration: # the first chunk can contain a full bar as well
								saved_duration -= ds.bar_duration.duration
								pre += ds.bar_duration.duration
								duration -= ds.bar_duration.duration
							post = duration - saved_duration
							if saved_duration > 0.0 and ((post + ds.bar_duration.duration) <= ds.max_duration.duration): # the last chunk can include a full bar
								saved_duration -= ds.bar_duration.duration
								post += ds.bar_duration.duration
							to_process += [(pre, 2, event.start)] # must be aligned to bar line at right end
							to_process += [(post, 0, event.start + pre + saved_duration)] # must be aligned to bar line at left end
						else:
							to_process += [(duration, 1, event.start)] # no alignment to bar lines required
						print("------SAVED BARS: ", saved_duration / ds.bar_duration.duration) if p >= 2 else 0
						end_time = event.start
						last_partial_index = len(to_process) - 1
						for partial_index, duration_triple in enumerate(to_process):
							print("------PARTIAL DURATION: ", duration_triple) if p >= 2 else 0
							# use quarter note as a limit for when less margin for quantization and alignment is used
							
							# this is a heuristic system that affects what note values will be available when aligning and quantizing a note. The idea is that 
							# often, in a track, no matter if there is polyphony or not, all or groups of pitches follow the same note values and starts in
							# some relationship to notes that start at the same time or ends at the same time as when a note starts. We therefore, for each track
							# keep track of the previous note: if it is triplet, duplet or based on both, starting time, ending time and whether the preceding note
							# was triplet, duplet or both. We can then deduce stuff about the upcoming note, IF it turns out to start at the same time or at the
							# ending time of the previous note. If we are at a place in time where we can go both in duplet and triplet direction, we will do
							# so, otherwise we will use the info about predecing note to determine whether to choose from triplet or duplet note values. If there is
							# too much time that has passed, we will ignore the info and finally, when encountering multiple notes that starts at the same time but
							# ends at different times, we will keep the shortest. This is a heuristic.
						
							# separate indicates whether the next note is to be considered isolated from previous note or not
							if (event.start - status[event.track][1]) < (eighth / 4): # started at the same time as the recorded event
								separate = False
								status_index = 3
							elif (event.start - status[event.track][2]) < (eighth / 4): # started when the recorded event ended
								status_index = 0
								separate = False
							else:
								separate = True # open to suggestions
								
							if triplet_ts or (not separate and (status[event.track][status_index] == 1)): # triplets only
								if event.tempo < self.FAST_THRESHOLD:
									chosen_track = "TRIPLET_FAST"
									excludes = ["d32", "d16", "t16"]
								elif event.tempo > self.SLOW_THRESHOLD:
									excludes = ["d32", "d16"]
									chosen_track = "TRIPLET_SLOW"
								else:
									chosen_track = "TRIPLET_STANDARD"
									excludes = ["d32", "d16"]
							elif not separate and status[event.track][status_index] == 2:
								if event.tempo < self.FAST_THRESHOLD:
									excludes = ["t16", "t8", "t4", "t2", "t1", "d32", "d16"]
									chosen_track = "DUPLET_FAST"
								elif event.tempo > self.SLOW_THRESHOLD:
									excludes = ["t16", "t8", "t4", "t2", "t1"]
									chosen_track = "DUPLET_SLOW"
								else:
									excludes = ["t16", "t8", "t4", "t2", "t1", "d32"]
									chosen_track = "DUPLET_STANDARD"
								if event.instrument in [2, 4, 5, 6, 7, 8, 9]:
										chosen_track += " (NO_SHORTS)"
										excludes += ["d32", "t16"]
							else: # open to suggestions
								if event.tempo < self.FAST_THRESHOLD:
									chosen_track = "FAST"
									excludes = ["d32", "t16", "d16"]
								elif event.tempo > self.SLOW_THRESHOLD:
									chosen_track = "SLOW"
									excludes = []
								else:
									excludes = []
									chosen_track = "STANDARD"
							if event.instrument in [2, 4, 5, 6, 7, 8, 9]:
									chosen_track += " (NO_SHORTS)"
									excludes += ["d32", "t16"]
							if event.instrument in [9]:
									chosen_track += " (TIMPANI)"
									excludes += ["d16"]
							excludes += ["t16"]
							if event.tempo < self.SLOW_THRESHOLD:
								excludes += ["d32"]
							current_ds = ds.filter_durations(duration_triple[2], ds.basic_set['d4'].duration / 16, excludes)
							candidates, top_candidate = get_quantization_candidates(duration_triple[0], ds.basic_set['none'], current_ds)
							
							if p >= 2:
								for t in candidates:
									print("......CAND:", t, " with excluded notes ", excludes, "from strategy ", chosen_track)
								print("......TOP: ", top_candidate, top_candidate)
							# pick the best candidate
							best_duration, adjustment, successful_match = sieve_candidates(duration_triple, candidates, event.start)
							success = successful_match if success else success # only replace success if it is Ture, otherwise, we stick with the negative outcome
							print(":::::::WINNER:", best_duration, adjustment) if p >= 2 else 0
							if duration_triple[1] == 2: # first partial of several
								if best_duration.duration > 0:
									event.duration_partials += [best_duration]
								#if int(saved_duration / ds.bar_duration.duration) > 0: # should not be needed
								event.duration_partials += ([ds.bar_duration] * int(saved_duration / ds.bar_duration.duration))
							else: # applies to single partial durations and the last partial of a multiple partial duration
								if best_duration.duration > 0:
									event.duration_partials += [best_duration]
							if p >= 2:
								print(end_time)
								print(adjustment)
							if adjustment != 0:
								event.start += adjustment
								end_time += adjustment
								out_adjustment = adjustment
								#out_adjustment = adjustment # should only be non-zero, if at all, when processing one of the, at most two, partials
							out_duration += best_duration.duration
							end_time += best_duration.duration
							if partial_index == last_partial_index: # the info to be stored is only important in this case
								if out_duration > 0: # include notes with several partials where the last is 0 as well
									if (end_time - best_duration.duration) == status[event.track][1]: 
										# the processed event started at the same time as the last event on this track
										if end_time >= status[event.track][2]: 
											# the current event ends after the one already registered, keep the shorter
											# essentially, we already have what is recorded and may keep it as is
											pass
										else:
											# the newly found duration ends earlier than the one we had recorded from before
											if (end_time % eighth) == 0:
												status[event.track] = (0, end_time - best_duration.duration, end_time, status[event.track][3])
											else:
												if best_duration.is_triplet_based():
													status[event.track] = (1, end_time - best_duration.duration, end_time, status[event.track][3])
												else:
													status[event.track] = (2, end_time - best_duration.duration, end_time, status[event.track][3])									
									elif (end_time - best_duration.duration) == status[event.track][2]:
										# the processed event starts at the same time as the earlier ended
										if (end_time % eighth) == 0:
											status[event.track] = (0, end_time - best_duration.duration, end_time, status[event.track][0])
										else:
											if best_duration.is_triplet_based():
												status[event.track] = (1, end_time - best_duration.duration, end_time, status[event.track][0])
											else:
												status[event.track] = (2, end_time - best_duration.duration, end_time, status[event.track][0])
									else:
										# this note is not related to earlier notes, create new entries that do not depend on previous ones
										if (end_time % eighth) == 0:
											status[event.track] = (0, end_time - best_duration.duration, end_time, 0)
										else:
											if best_duration.is_triplet_based():
												status[event.track] = (1, end_time - best_duration.duration, end_time, 0)
											else:
												status[event.track] = (2, end_time - best_duration.duration, end_time, 0)									
								else:
									# this note was determined to be deleted
									# keep the record as it is of the last seen note, the distance to it will determine whether to use the supplied info or not anyways
									pass
						out_duration += saved_duration # restore the full bars saved
						
						# follow up
						if out_duration == 0 or not success: # this note was quantized to nothing
							event.invalidate()
						else:
							nonlocal processed_valid_notes
							processed_valid_notes += 1
							event.duration = out_duration
							
							print("Resulting adjustment: ", out_adjustment) if p >= 2 else 0
				event.start += range_start
				return event
				
			tokens = tokens[0: i] + list(map(quantize_and_align, list(tokens[i: len(tokens)])))
			tokens.sort(key = lambda x: (x.start, int(x.is_start()), x.pitch)) # sort first by time, then by start and end not and last by pitch in ascending order
			if successful_adjustments + failed_adjustments > 0:
				failed_ratio = failed_adjustments / (successful_adjustments + failed_adjustments)
			else:
				failed_ratio = 0.0
			assert(failed_ratio < self.MAX_FAILED_ADJUSTMENTS), "[PROC_TOK]: ERROR - " + str(round(100 * failed_ratio, 2)) + "% of processed tokens were not successfully matched, investigate if desirable, might be due to introduction of incorrect offset somewhere"
			if processed_valid_notes == 0:
				tokens = []
		else:
			tokens = []		
	
		return tokens

	# when multiple notes in the same track is sounding, try to make them unified into a single duration instead of several different
	def unify_tokens(self, tokens, unify_individual_tracks = True):
		total = len(tokens)
		time = 0
		i = 0
		while i < total:
			starts = {} # bookmark the notes starting on this time step by "track_number: [NOTES]"
			
			# collect all the notes that start on this time step
			while i < total and tokens[i].start == time:
				if tokens[i].is_valid():
					if unify_individual_tracks:
						track_no = tokens[i].track
					else:
						track_no = 0
					if track_no not in starts:
						starts[track_no] = []
					starts[track_no] += [i]
				i += 1
				
			# unify the notes found on this time step, if possible
			for key in starts:
				notes_no = len(starts[key]) # number of notes in this track starting at this time step
				if notes_no > 1: # only unify if we have at least two notes to compare
					durations = {} # bookmark the durations found by "duration (in ticks): (occurrences in this time step, the duration(s) objects making up this duration)"
					basics = {} # same as above but only for basic units since these are more common
					for token_index in starts[key]:
						# if they show coherence and one is sticking out, make them all the same length
						duration = tokens[token_index].duration
						if len(tokens[token_index].duration_partials) == 1 and tokens[token_index].duration_partials[0].is_basic():
							if duration not in basics:
								basics[duration] = (1, tokens[token_index].duration_partials)
							else:
								basics[duration] = (basics[duration][0], tokens[token_index].duration_partials)
						else:
							if duration not in durations:
								durations[duration] = (1, tokens[token_index].duration_partials)
							else:
								durations[duration] = (durations[duration][0], tokens[token_index].duration_partials)
								
					# all occurrences recorded, now go through the notes with this in mind
					durations = [(d, durations[d][0], durations[d][1]) for d in durations] # duration length and the number of occurrences and duration object(s)
					durations.sort(key = lambda x: x[1], reverse = True) # sort by number of occurrences, descending order
					basics = [(b, basics[b][0], basics[b][1]) for b in basics]
					basics.sort(key = lambda x: x[1], reverse = True)
					for token_index in starts[key]:
						done = False
						for basic in basics:
							if basic[1] >= (notes_no / 2) and abs(basic[0] - tokens[token_index].duration) < (tokens[token_index].duration / 4):
								# only replace a note if the found basic note is at least half the number of occurrences and the correction corresponds to less 
								# than a quarter of the original note
								tokens[token_index].duration = basic[0]
								tokens[token_index].duration_partials =  basic[2]
								done = True
								break
						if not done:
							for duration in durations:
								if duration[1] >= (notes_no / 2) and abs(duration[0] - tokens[token_index].duration) < (tokens[token_index].duration / 8):
									tokens[token_index].duration = duration[0]
									tokens[token_index].duration_partials =  duration[2]
									break
			if i < total:
				time = tokens[i].start
		return tokens
		
	# adds the final data before translation to data representation can take place, this include splitting long notes and inserting them as well as inserting end notes
	def finalize_representation(self, tokens, ds): # sets up the list with start tokens with end tokens as well so it will be easier to make the training matrices
		out_tokens = list([])
		cancelled_notes = 0
		start = 0
		end = 0
		# lists to verify that each included notes can actually be playr with a 16 (really 15) channel MIDI device
		out_instruments = [-1] * self.NUM_CHANNELS # holds the current internal instrument of a certain out channel
		out_hold = [0] * self.NUM_CHANNELS # holds the absolute time until which an outgoing channel is occupied until available for a new instrument
		out_hold[9] = float('inf') # never write to drum channel
		# holds a mapping from internal instrument number to channel number, -1 indicates that no channel has this instrument right now	, last index is always 
		out_reverse_instruments = [-1] * self.NUM_INSTRUMENTS
		
		
		index = 0 # the earliest index in the output list where it is possible to place the start of the next token
		for token in tokens:
			if type(token) is Note and token.is_valid(): # it is a note (should only be notes here anyways)
				
				current_time = token.start
				frontier = index
				track = out_reverse_instruments[token.instrument]
				
				# find a hypothetic channel for this event
				if track == -1:
					track = out_hold.index(min(out_hold)) # find the out channel that is available at the earliest
					if out_hold[track] <= current_time: # there is a channel available for instrument change to the needed instrument right now
						if out_instruments[track] >= 0: # this track has had an instrument assigned to it before
							out_reverse_instruments[out_instruments[track]] = -1 # this instrument is no longer associated with a channel
						out_instruments[track] = token.instrument
						out_reverse_instruments[token.instrument] = track
					else:
						track = -1
				
				# only use token if it passed the hypothetical channel test, otherwise, cancel the token without any distinction
				if track >= 0:
		
					out_hold[track] = current_time +  token.duration
					# set the length of the first (and perhaps only duration) that we will lay out
					for partial in token.duration_partials: # fill up with notes for as long as needed
						#print("it: ", next_duration)
						# insert new starting note
						start_note = Note(current_time, token.pitch, token.channel, token.instrument, self.inst, token.track, token.tempo)
						start_note.duration = partial.duration
						start_note.duration_partials = [partial]
						while frontier < len(out_tokens) and (current_time > out_tokens[frontier].start or (current_time == out_tokens[frontier].start and (out_tokens[frontier].is_end() or out_tokens[frontier].pitch < token.pitch))):
							frontier += 1
						out_tokens.insert(frontier, start_note)
						frontier += 1
						if current_time == token.start: # means that this is the start of the first duration and then index needs to be updated by where we insert this
							index = frontier
						end_time = current_time + partial.duration
						
						# insert end note
						end_note = Note(end_time, token.pitch, token.channel, token.instrument, self.inst, token.track, token.tempo)
						start_note.end = end_note # bookmark the end note so that we can invalidate it as well if we invalidate the start note
						while frontier < len(out_tokens) and ((end_time > out_tokens[frontier].start or (end_time == out_tokens[frontier].start and (out_tokens[frontier].is_end() and out_tokens[frontier].pitch < token.pitch)))):
							frontier += 1
						out_tokens.insert(frontier, end_note)
						frontier += 1
						
						current_time = end_time
				else:
					cancelled_notes += 1
		return out_tokens, cancelled_notes
							
	def to_data_representation(self, tokens, ds, last_event, milestone, active_notes, active_instruments):
		p = self.P_TO_DATA
		timestep_status = []
		skipped = 0
		
		output = np.zeros((max(len(tokens), 20), self.timestep_sz), dtype=np.uint8)
		if active_instruments is None:
			active_instruments = np.zeros((self.NUM_INSTRUMENTS), dtype="int")
		if active_notes is None:
			active_notes = np.zeros((self.NUM_PITCHES), dtype='int')
		i = 0
		
		skip = False
		# to keep the beats invariant of the current time signature, we alwaysmodel beats according to a thought 4/4 time signature with the same beats / quarter as the current one
		fictive_4_4_bar = ds.basic_set["d4"].duration * 4
		beats_vector = []
		for unit in self.default_beat_units:
			beats_vector += [time for time in range(0, fictive_4_4_bar, ds.basic_set[unit].duration)]
		beats_vector = np.unique(np.array(beats_vector, dtype = np.int32))
		assert(len(beats_vector) == len(self.default_beats_vector)), "Discrepancy! Temporary and default beats vector differ in length!"

		def next_timestep(i, output, tick, is_pause):
			nonlocal timestep_status
			timestep_status += [(int(is_pause), tick)]
			i += 1
			if output.shape[0] <= i:
				return i, np.concatenate((output, np.zeros((output.shape), dtype=np.uint8)))
			else:
				return i, output

		st = 0
		en = 0
		oog = 0
		oop = -1
		for token in tokens:
			if token.is_valid():
				if token.is_start():
					if token.start != oog:
						oop = -1
					else:
						if(token.pitch < oop):
							print("WRONG BAD SHIT!" + str(token.pitch) + " " + oop)
							exit()
						oop = token.pitch
					skip = False
					# construct the input vector for this time step
					if p:
						print("i: ", i, "DELTA: ", token.start - last_event, "TOKEN: ", token)
					delta = token.start - last_event
					if p:
						print("delta: ", delta, " token start: ", token.start, "max duration: ", ds.max_duration.duration)
					
					# lay out single pauses if the distance to the next note is too large to capture ina single time step
					while (token.start - last_event) > ds.max_duration.duration:
						to_next_barline = ds.bar_duration.duration - (last_event % ds.bar_duration.duration)
						if (to_next_barline + ds.bar_duration.duration) <= ds.max_duration.duration:
							to_next_barline += ds.bar_duration.duration
						if self.USE_DISTRIBUTED_DURATIONS:
							for basic_duration in ds.inverse_mappings["l" + str(to_next_barline)].partials:
								output[i][self.time_a + basic_duration.get_distributed_index()] = 1
						else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
							output[i][self.time_a + ds.inverse_mappings["l" + str(to_next_barline)].get_one_hot_index()] = 1
							output[i][self.dur_a + ds.inverse_mappings["l0"].get_one_hot_index()] = 1 # mark an empty event as a zero duration
						output[i][self.act_notes_a: self.act_notes_a + self.NUM_PITCHES] = (active_notes > 0).astype(int)
						output[i][self.act_inst_a: self.act_inst_a + self.NUM_INSTRUMENTS] = (active_instruments > 0).astype(int)
						last_event += to_next_barline
						current_beats_vector = ((last_event % ds.bar_duration.duration) == beats_vector).astype(np.uint8)
						assert(len(np.nonzero(current_beats_vector)[0]) <= 1), "1several 1s in beats vector, must be wrong " + str(current_beats_vector)
						output[i][self.beats_a: self.beats_a + self.NUM_BEATS] = current_beats_vector
						i, output = next_timestep(i, output, last_event, True)
					
					#process an actual note
					if (token.start - last_event) > 0:
						if "l" + str(token.start - last_event) not in ds.inverse_mappings:
							# need to divide this into two
							first_half_fill = ds.basic_set['d8'].duration - (last_event % ds.basic_set['d8'].duration)
							if "l" + str(first_half_fill) not in ds.inverse_mappings:
								skip = True
							elif "l" + str(token.start - (last_event + first_half_fill)) not in ds.inverse_mappings:
								skip = True
							else:
								if self.USE_DISTRIBUTED_DURATIONS:
									for basic_duration in ds.inverse_mappings["l" + str(first_half_fill)].partials:
										output[i][self.time_a + basic_duration.get_distributed_index()] = 1
								else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
									output[i][self.time_a + ds.inverse_mappings["l" + str(first_half_fill)].get_one_hot_index()] = 1
									output[i][self.dur_a + ds.inverse_mappings["l0"].get_one_hot_index()] = 1 # mark an empty event as a zero duration
								output[i][self.act_notes_a: self.act_notes_a + self.NUM_PITCHES] = (active_notes > 0).astype(int)
								output[i][self.act_inst_a: self.act_inst_a + self.NUM_INSTRUMENTS] = (active_instruments > 0).astype(int)
								last_event += first_half_fill
								current_beats_vector = ((last_event % ds.bar_duration.duration) == beats_vector).astype(np.uint8)
								assert(len(np.nonzero(current_beats_vector)[0]) <= 1), "2several 1s in beats vector, must be wrong " + str(current_beats_vector)
								output[i][self.beats_a: self.beats_a + self.NUM_BEATS] = current_beats_vector
								i, output = next_timestep(i, output, last_event, True)
								
					if not skip:
						st += 1
						if self.USE_DISTRIBUTED_DURATIONS:
							if (token.start - last_event) > 0:
								for basic_duration in ds.inverse_mappings["l" + str(token.start - last_event)].partials:
									output[i][self.time_a + basic_duration.get_distributed_index()] = 1
						else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
							output[i][self.time_a + ds.inverse_mappings["l" + str(token.start - last_event)].get_one_hot_index()] = 1
						output[i][self.inp_a + token.pitch - self.LOW_NOTE_THRESHOLD] = 1
						output[i][self.instr_a + token.instrument] = 1
						if self.USE_DISTRIBUTED_DURATIONS:
							for basic_duration in token.duration_partials[0].partials:
								output[i][self.dur_a + basic_duration.get_distributed_index()] = 1
						else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
							output[i][self.dur_a + token.duration_partials[0].get_one_hot_index()] = 1
						output[i][self.act_notes_a: self.act_notes_a + self.NUM_PITCHES] = (active_notes > 0).astype(int)
						output[i][self.act_inst_a: self.act_inst_a + self.NUM_INSTRUMENTS] = (active_instruments > 0).astype(int)
						current_beats_vector = ((token.start % ds.bar_duration.duration) == beats_vector).astype(np.uint8)
						assert(len(np.nonzero(current_beats_vector)[0]) <= 1), "3several 1s in beats vector, must be wrong " + str(current_beats_vector)
						output[i][self.beats_a: self.beats_a + self.NUM_BEATS] = current_beats_vector
						
						# alter state for next time step
						active_instruments[token.instrument] += 1
						active_notes[token.pitch - self.LOW_NOTE_THRESHOLD] += 1
						last_event = token.start
						i, output = next_timestep(i, output, last_event, False)
					else:
						skipped += 1
						token.end.invalidate()
					
				else: # token is an end token
					en += 1
					active_instruments[token.instrument] -= 1
					assert(active_instruments[token.instrument] >= 0), "Map over active instruments sank below 0"
					active_notes[token.pitch - self.LOW_NOTE_THRESHOLD] -= 1
					assert(active_notes[token.pitch - self.LOW_NOTE_THRESHOLD] >= 0), "Map over active pitches sank below 0"
					
		# lay out pauses to the end of the bar if necessary
		if milestone is not None:
			# lay out single pauses if the distance to the next note is too large to capture ina single time step
			while (milestone - last_event) > 0:
				left = milestone - last_event
				to_next_barline = ds.bar_duration.duration - (last_event % ds.bar_duration.duration)
				layout = min(left, to_next_barline)
				if (to_next_barline + ds.bar_duration.duration) <= ds.max_duration.duration:
					to_next_barline += ds.bar_duration.duration
				layout = min(layout, to_next_barline)
				if self.USE_DISTRIBUTED_DURATIONS:
					for basic_duration in ds.inverse_mappings["l" + str(layout)].partials:
						output[i][self.time_a + basic_duration.get_distributed_index()] = 1
				else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
					output[i][self.time_a + ds.inverse_mappings["l" + str(layout)].get_one_hot_index()] = 1
					output[i][self.dur_a + ds.inverse_mappings["l0"].get_one_hot_index()] = 1 # mark an empty event as a zero duration
				output[i][self.act_notes_a: self.act_notes_a + self.NUM_PITCHES] = (active_notes > 0).astype(int)
				output[i][self.act_inst_a: self.act_inst_a + self.NUM_INSTRUMENTS] = (active_instruments > 0).astype(int)
				last_event += layout
				current_beats_vector = ((last_event % ds.bar_duration.duration) == beats_vector).astype(np.uint8)
				assert(len(np.nonzero(current_beats_vector)[0]) <= 1), "4several 1s in beats vector, must be wrong " + str(current_beats_vector)
				output[i][self.beats_a: self.beats_a + self.NUM_BEATS] = current_beats_vector
				i, output = next_timestep(i, output, last_event, True)
		return output[: i], timestep_status, active_notes, active_instruments, skipped, beats_vector
		
	def from_data_representation(self, ranges, fill = False, default_duration = "d8"):
		p = self.P_FROM_DATA
		time = 0
		offset = 0
		num = -1
		denom = -1
		index = 0
		tokens = []
		for range in ranges:
			if p:
				print("Range length: ", len(range[0]))
			ds = range[2]
			current_num = ds.abstract_duration_set.ts_num
			current_denom = ds.abstract_duration_set.ts_denom
			if current_num != num or current_denom != denom:
				ts = TimeSignature(time, current_num, current_denom)
				while index < len(tokens) and tokens[index].start < time:
					if p:
						print(tokens[index])
					index += 1
				tokens.insert(index, ts)
				if p:
					print(tokens[index])
				index += 1
				num = current_num
				denom = current_denom
			for i, vec in enumerate(range[0]): # process all vectors in one sequence
				if vec[self.START_TOKEN] == 0: # and vec[self.END_TOKEN] == 0: # not a start or end token
					if self.USE_DISTRIBUTED_DURATIONS:
						for basic_duration in np.nonzero(vec[self.time_a: self.time_a + self.num_durations])[0]:
							offset += ds.inverse_mappings["bi" + str(basic_duration)].duration
					else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
						o_indexes = np.nonzero(vec[self.time_a: self.time_a + self.num_durations])[0]
						if len(o_indexes) == 0:
							if i > 0:
								offset += ds.basic_set[default_duration].duration # use a regular eighth note if note offset is supplied
						else:
							offset += ds.inverse_mappings["li" + str(o_indexes[0])].duration
					pitch = np.nonzero(vec[self.inp_a: self.inp_a + self.NUM_PITCHES])
					if len(pitch[0]) == 0: # only time progression, keep accumulating offset
						if p:
							print(vec)
							print("accumulated offset", offset, i)
						pass
					else:
						start = time + offset
						pitch = pitch[0][0] + self.LOW_NOTE_THRESHOLD
						i_indexes = np.nonzero(vec[self.instr_a: self.instr_a + self.NUM_INSTRUMENTS])[0]
						if len(i_indexes) == 0:
							instrument = 0
						else:
							instrument = i_indexes[0]
						if self.USE_DISTRIBUTED_DURATIONS:
							duration = 0
							for basic_duration in np.nonzero(vec[self.dur_a: self.dur_a + self.num_durations])[0]:
								duration += ds.inverse_mappings["bi" + str(basic_duration)].duration
						else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
							d_indexes = np.nonzero(vec[self.dur_a: self.dur_a + self.num_durations])[0]
							if len(d_indexes) == 0:
								duration = ds.basic_set[default_duration].duration # use a regular eighth note if note offset is supplied
							else:						
								duration = ds.inverse_mappings["li" + str(d_indexes[0])].duration
						add = Note(start, pitch, -1, instrument, self.inst)
						add.duration = duration
						# insert the created token
						while index < len(tokens) and (tokens[index].start < start or (tokens[index].start == start and tokens[index].is_end())):
							if p:
								print(tokens[index])
							index += 1
						tokens.insert(index, add)
						if p:
							print(tokens[index])
						index += 1
						end = start + duration
						end_add = Note(end, pitch, -1, instrument, self.inst)
						add.end = end_add
						frontier = index
						while frontier < len(tokens) and tokens[frontier].start < end:
							frontier += 1
						tokens.insert(frontier, end_add)
						time = start
						offset = 0
				else:
					if p:
						print("Found START TOKEN at position " + str(i) + ": ", vec[self.START_TOKEN])
			if fill:
				# needed when there is no offset filler at the end of a range to "fill out", because the first event in the next range will be with respect
				# to that range. However, when there is a filler, this works too because then we just reset the offset (that have been accumulated before getting
				# here) and then we start over... so fill could always be True and then it would work in all situations, no matter if there is a filler or not.
			
				# add the time left to the end of this chunk to the offset for the next event to output
				time = range[4]
				offset = 0 # since it's the time variable that dictates when the next reference point is, not the offset
		if p:
			while index < len(tokens):
				print(tokens[index])
				index += 1
		return tokens
			
	def midify(self, tokens, instr, tracks_num = 16, use_tracks = False, tpb = 480):
		p = self.P_TO_MIDI
		out = mido.MidiFile()
		instruments = [-1] * self.NUM_CHANNELS # holds the current instrument of a certain channel
		track_instruments = [-1] * tracks_num
		last_written = [0] * tracks_num # holds the absolute ticks when a certain channel was last written
		hold = [0] * self.NUM_CHANNELS # holds the farmost time for which a channel is occupied by an ongoing note
		hold[9] = float('inf')
		if not use_tracks:
			last_written[9] = hold[9]
		reverse_instruments = [-1] * self.NUM_INSTRUMENTS # holds a mapping from instrument number to channel number, -1 indicates that no channel has this instrument right now
		if p:
			print("\r\n starting MIDIFICATION!\r\n")
		num_tracks = tracks_num
		for i in range(num_tracks):
			add = mido.MidiTrack()
			out.tracks.append(add)
			
		for i, token in enumerate(tokens):
			#token.instrument = 0 # to make all instruments sound like piano
			increment = True
			if type(token) is TimeSignature or type(token) is Tempo:
				track = 0
			elif type(token) is Program:
				track = token.channel
			else:
				if use_tracks:
					track = token.track
					channel = reverse_instruments[token.instrument]
				else:
					track = channel = reverse_instruments[token.instrument]
		
			if type(token) is TimeSignature:
				out.tracks[track].append(mido.MetaMessage('time_signature', numerator=token.num, denominator=token.denom, time=token.start - last_written[track]))
			elif type(token) is Tempo:
				pass
			elif type(token) is Program:
				pass
			elif token.is_valid():
				if p:
					print(token, track)
				if token.is_start():
					if channel == -1:
						channel = hold.index(min(hold))
						if not use_tracks:
							track = channel
							track_instruments[track] = token.instrument
						if p:
							print("next", channel, instr.get_output_mapping(token.instrument))
						out.tracks[track].append(mido.Message('program_change', channel=channel, program=instr.get_output_mapping(token.instrument), time=0))
						if instruments[channel] >= 0: # this track has had an instrument assigned to it before
							reverse_instruments[instruments[channel]] = -1
						instruments[channel] = token.instrument
						reverse_instruments[token.instrument] = channel
					if track_instruments[track] != token.instrument:
						out.tracks[track].append(mido.Message('program_change', channel=channel, program=instr.get_output_mapping(token.instrument), time=0))
					if channel < 0:
						print(channel, token, track)
						exit()
					out.tracks[track].append(mido.Message('note_on', channel = channel, note=token.pitch, velocity=64, time=token.start - last_written[track]))
					
					# according to midi standard, it is ok to change program on a channel before a note is finished, it won't be affected, we reserve it for the sake of channels
					if p:
						pass
					if isinstance(token.start - last_written[track], float):
						print(token)
						print(last_written[track])
						exit()	
					hold[channel] = token.start + token.duration
				else: # token is an end token for a note
					out.tracks[track].append(mido.Message('note_on', channel = channel, note=token.pitch, velocity=0, time=token.start - last_written[track]))
			else:
				increment = False # invalid Note
			if increment:
				last_written[track] = token.start		
			
		out.ticks_per_beat = tpb
		return out
		
	def print_tokens(self, tokens, header, flag, limit = None):
		if flag:
			if limit is None:
				limit = len(tokens)
			else:
				limit = min(limit, len(tokens))
			print("\r\n(!!!) " + header + ":")
			for i, tok in enumerate(tokens):
				if i == limit:
					break
				print(tok)

	''' 
	Takes a data representation of ONE consecutive chunk of music (may contain several different time signatures with no interruption in between) and
	yields context, input pairs where the input is of length sequence_base * inp_length and the context is of length sequence_base * ctx_length. A stride of
	one sequence_base is always used, meaning that after a pair has been yielded, no matter the number of base units in context or input, the next yielded	
	tuple will always hold the same pattern as the one yielded before, only moved one base unit towards the end of the song. The first tuple yielded will have
	a context only consisting of a special start token, and an input consisting of the first inp_length sequence_bases of the first input tuple. The last
	yielded tuple will have the last inp_length sequence_bases of the last input tuple as input an the preceeding ctx_length sequence_bases as context. Thus,
	no end of song token is provided at this time.
	
	Parameters
	----------
	
	data_tuples: list of 6-tuples
		A list of 6-tuples representing an uninterrupted flow of music where each 6-tuple represents the range of a time signature. If a song has no changes 
		in time signatures, this list will only contain one 6-tuple and it will represent the entire input song (that may otherwise be divided into multiple 
		different chunks if an interruption in the form of an unsupported time signature should occur, or multiple ranges within a single chunk (here 6-tuples)
		if all time signatures are supported. Each 6-tuple is on the form (data_representation, statistics, duration_set, start_time, next, beats vector) 
		where data representation holds the actual data vectors. Statistics is a list of the same length as there are data vectors (time steps) and contains, 
		for each data vector, a tuple on the form (is pause_only, ticks at start) indicating whether the time step is a pause only (transport to next note 
		due to too long offset) and at what tick within the input chunk / song the vector starts. Duration_set is the duration_set used for the 6-tuple and 
		start_time and next indicates the starting tick and ending tick of the 6-tuple within the current chunk / song, as represented by this list of 
		6-tuples. Finally, the beats vectors indicates what beat milestones, in terms of ticks within a bar, that were used for this subrange of music.
	
	sequence_base: str
		A string indicating the unit of the sequence base used for generating context, input tuples. May be a note value (for example "d4" for duple time
		quarter note or "t8" for triplet time eight note) or "bar" indicating that the base unit used should be the somewhat differing length of a bar 
		according to the current time signature.
	
	inp_length: int
		The number of units used for the input part of the yielded tuple.
		
	ctx_length: int
		The number of units used for the context part of the yielded tuple.		
		
	Yields
	------
	
	(numpy array, numpy array)
		Tuples with a context and input of size ctx_length * sequence_base and inp_length * sequence_base respectively. No reference is held to the returned
		tuples and they are thus fresh copies with no need for further copying.
		
	'''
	def sequence_generator(self, data_tuples, sequence_base, ctx_length, inp_length):
		p = self.P_GENERATOR
		if p:
			print("[SEQ_GEN]: Initializing a run with", len(data_tuples), "range(s)")
		#assert att sequence_base delar såväl alla tuples som den totala längden
		
		def get_base(ds):
			if sequence_base == "bar": # unit is in terms of bars
				return ds.bar_duration.duration
			else: # note value
				return ds.basic_set[sequence_base].duration
				
		total_bases = 0
		for data_tuple in data_tuples:
			base = get_base(data_tuple[2])
			subrange = data_tuple[4] - data_tuple[3]
			assert(subrange >= base), "Illegal subrange: contains less than one unit of the chosen base"
			assert((subrange % base) == 0), "All ranges must divide the base exactly"
			total_bases += subrange // base
		assert(total_bases >= inp_length), "Not enough data to generate a single pattern"
		'''
		Takes an array of numpy arrays along with a start time and an end time that this window frame should represent, and construct a single numpy array
		with the pattern. Because the original vectors come from a continuous representation, the initial offset of the first vector might have to be adjusted
		to suit the current framing context (since the offset of the first vector should represent the offset to the beginning of the current partial window 
		(context or input) as opposed to the offset to the previous event. To determine the new offset, this function takes a reference tuple which is either
		the first tuple in the partial window that contains an active vector OR, if the entire partial window is silent, the last tuple in the partial window.
		The idea is that we need to use the duration set of this reference tuple to either encode a new offset for the initial vector or, in the absence of
		vectors, encode an empty offset positioned at the end of the partial window. In the case of the latter, we use the NEXT active vector that we may find
		which is often the next available vector in the same reference tuple. In unique cases, this is not possible and this happens at the end of a song.
		To see why, in to_data_representation, pause vectors are laid out between different time signatures but the last time signature range does not end with 
		such a pause since nothing ensues it. In this situation, we use the last tuple in the window to encode the pause only.
		
		Note that since this function only gets called once we know that either the start or the end point is within the actual time of the song (not dummies
		signalling that we are before the song starts of or after its end), we can rest assured that the reference tuple holds an actual data tuple in it. 
		
		Parameters
		----------
		
		w_tuple: 5-tuple of (start time, end time, data tuple, current base length, base indices)
			A reference window tuple that is used to either change the offset of the initial vector, if pattern contains any vectors, or determine how to encode
			an empty offset to place at the end of this context or input window, if pattern is empty and there is silence during th time of this partial window.
			The reference tuple is therefore in the first case, the first tuple in the input window that contains an actual vector, or the last tuple in the input
			window if no such vector exists.

		pattern: list of numpy arrays
			A list of numpy arrays with all the feature ranges gathered from data 6-tuples in the current window using base indices.
			
		start: int
			The starting time of the current framing to process.
		
		end: int
			The ending time of the current framing to process.
			
		Returns
		-------
		
		numpy array
			A 2-dimensional numpy array where the first dimension holds timesteps and the second holds features. The output array is a concatenation of
			patterns with initial offset adjusted to the input starting time, or an empty offset pause positioned at input ending time covering the full 
			length of the silent window.
			
		'''			
		def generate_pattern(w_tuple, pattern, start, end):
			ds = w_tuple[2][2]
			is_pause = False
			if len(pattern) == 0: # create a pause of the correct length, use the next active vector as a basis for this
				is_pause = True
				pattern = np.zeros((1, self.timestep_sz), dtype=np.uint8)
				if len(w_tuple[2][0]) > w_tuple[4][-1]: # use the last base index, which is exclusive so this is the next vector in the song
					# keep only the active instruments, ignore active pitches since we don't know how many have dies during the silence (heuristic approximation 
					#which maybe slightly incorrect but assumed to not have a major impact). # i might just as well keep them
					pattern[0][self.act_inst_a: self.act_inst_a + self.NUM_INSTRUMENTS] = \
						w_tuple[2][0][w_tuple[4][-1]][self.act_inst_a: self.act_inst_a + self.NUM_INSTRUMENTS]
					pattern[0][self.act_notes_a: self.act_notes_a + self.NUM_PITCHES] = \
						w_tuple[2][0][w_tuple[4][-1]][self.act_notes_a: self.act_notes_a + self.NUM_PITCHES]
				else:
					pass
				# take care of beats
				current_beats_vector = ((end % ds.bar_duration.duration) == w_tuple[2][5]).astype(np.uint8)
				assert(len(np.nonzero(current_beats_vector)[0]) <= 1), "generator: several 1s in beats vector, must be wrong" + str(current_beats_vector)
				pattern[0][self.beats_a: self.beats_a + self.NUM_BEATS] = current_beats_vector
				offset = end - start
			else: # just alter the initial vector so it reflects the correct offset
				pattern = np.concatenate(tuple(pattern), axis = 0) # concatenates all the vector ranges in pattern
				note_start = w_tuple[2][1][w_tuple[4][0]][1] # get the statistics for the first vector base index and extract its start time
				offset = note_start - start
				if p:
					print("[GENERATE_PATTERN]: altering offset of initial vector (note start of first vector according to statistics / start of window)", note_start, start)
				assert(offset >= 0), "Illegal (negative) offset detected"
				pattern[0][self.time_a: self.time_a + self.num_durations] = 0 # zero the current offset
			if self.USE_DISTRIBUTED_DURATIONS:
				if offset > 0: # if the offset is too long to express in a single event, there must have been some empty offsets laid out and included already
					for basic_duration in ds.inverse_mappings["l" + str(offset)].partials:
						pattern[0][self.time_a + basic_duration.get_distributed_index()] = 1
			else: # USE ONE HOT DURATIONS AND COMPOUND DURATIONS
				assert("l" + str(offset) in ds.inverse_mappings), "[GENERATE_PATTERN]: ERROR - offset key not found, perhaps due to bad choice of tpb " + str(ds.tpb)
				pattern[0][self.time_a + ds.inverse_mappings["l" + str(offset)].get_one_hot_index()] = 1
			if is_pause:
				pattern[0][self.dur_a + ds.inverse_mappings["l0"].get_one_hot_index()] = 1 # mark an empty event as a zero duration
			return pattern
		
		'''
		Takes a window (list) of 5-tuples in the same form as presented earlier (start time, end time, data 6-tuple, base unit length, base indices) where 
		each consective two indices in base indices corresponds to a mapping into the data vectors of the data tuple (remember that the data tuples ar on the
		form (data representation (vectors), statistics, duration set, start time, end time)) of a size corresponding to a sequence base, according to
		the surrounding function. All the 5-tuples along with their base indices represents one context / input pair and the role of this function is to
		extract these vectors as a separate data structure, adjust the initial offset of the first vector so that it is relative to the beginning of the 
		current time window, as opposed to relative to the previous event, which might have happened before the end of the previous time window and therefore
		give irrelevant information with respect to the curreng framing. Each 5-tuple has a start and ending time which indicates from what tuple a certain
		time range should be taken. Subsequent base indices in a tuple which are similar indicates silence during the entire framing, in which case a change of
		the offset on the first data vector is impossible. In this case, the start and ending time still indicates what tuple to use, when laying out pauses
		during this time. A negative starting time of a tuple indicates that a start token should be added and an ending time of a tuple that is greater than
		the ending time of the actual data tuple that it holds indicates an ensuing end of music token.
		
		Parameters
		----------
		
		start: int
			The starting time of the current framing to process.
		
		end: int
			The ending time of the current framing to process.
		
		window: list of 5-tuples
			A list of 5-tuples representing an uninterrupted flow of music where each 5-tuple represents the range of a time signature and the window has been
			sliced so that its entire content, as given by the base indices in each window tuple, corresponds to a window framing to be used for context or input
			pattern. Please note that in vectors the input data tuples, there might already be timesteps which only represents an offset. These happen when 
			the offset of an event is too large for it to be encoded in one event in which case emty offsets are lad out until this is possible. Another 
			scenario where empty offsets have been laid out by this MidiPreprocessor is at the end of a range in a song (e.g. end of a time signature). 
			Since, in the next time signature, there might be an offset to the start of the bar, a combination of offsets to the end of the bar in the old 
			time signature, and to the beginning of the bar in the new time signature, might be hard to process, we always terminate a range (time signature) 
			by laying out an empty offset to the actual end of its time. These events are rare, and even if they might have been solved differently, this is 
			how they are solved now. These events should ideally be removed in this method but they are not, since they are considered so rare and therefore, 
			they should not have a great impact on the output.
			
			Important to stress that all base indices in all input tuples belong to the range to generate, it is what they look like (consecutive indices are
			similar or different) and the nature of the start and end value (relative to the end value of the last tuple) that affects what we do.
			
		Returns
		-------
		
		numpy array
			A 2-dimensional numpy array where the first dimension holds timesteps and the second holds features.
			
		'''	
		def generate_window(start, end, window):
			# data_tuple: ([0] data_representation, [1] statistics, [2] duration_set, [3] start_time, [4] next, [5] beats vector)
			# statistics tuple: ([0] is pause only, [1] ticks count at start)
			# window tuple: ([0] start range, [1] end range, [2] data_tuple, [3] base, [4][base indices where len - 1 corresponds to the number of bases])
			if p:
				print("[GENERATE_WINDOW]: (start / end / length of window)", start, end, len(window))
			input_pattern = []
			ref_tuple = None
			
			# 1. Gather all the actual data vectors referred to by the sequence base indices in each window tuple, also determine the reference tuple, which
			# will be used to account for initial offset: if the range of actual vectors is empty, use the last window tuple, otherwise, use the tuple that
			# holds the first actual participating vector whose offset will be adjusted
			for w_tuple in window:
				if p:
					print("[GENERATE_WINDOW]: encountered tuple with starting time", w_tuple[0])
				if w_tuple[0] >= 0: # ignores the initial placeholder window tuple which has a negative starting time
					if p:
						print("[GENERATE_WINDOW]: encountered non-dummy tuple with base indices", w_tuple[4])
					if w_tuple[4][0] < w_tuple[4][-1]: # only add non-empty intervals
						input_pattern += [w_tuple[2][0][w_tuple[4][0]: w_tuple[4][-1]]]
						if ref_tuple is None:
							ref_tuple = w_tuple
			if ref_tuple is None: # there were no ranges to include, thus laying out a single pause using the info in the last tuple in the window is the right thing to do
				ref_tuple = window[-1]
				
			# 2. Generate the actual vectors
			if end > 0 and start < window[-1][2][4]: # otherwise only start or end token is needed
				input_pattern = [generate_pattern(ref_tuple, input_pattern, max(start, 0), end)]
			
			# 3. Add start or end token if needed
			if start < 0:
				start_token = np.zeros((1, self.timestep_sz), dtype=np.uint8)
				start_token[0][self.START_TOKEN] = 1
				if p:
					print("[GENERATE_WINDOW]: Generated start", start_token.shape)
				input_pattern = [start_token] + input_pattern
				
			return np.concatenate(input_pattern, axis = 0)

		# initialize
		dt_index = 0 # the index of the next input 5-tuple to add to the processing window
		segment_length = ctx_length + inp_length # this is the length of the entire segment under treatment, that is, both the context and the input
		# the first window should start before the song begins so that the first input contains the first sequence_base and the first context only holds a START_TOKEN
		win_start = -ctx_length
		win_end = 0 # window is expanded at the beginning of the loop and contracted at the end, we here simulate that the first expansion expands to the first unit
		# the current window of both context and input is a list of 5-tuples where each 5-tuple represents how much of an active input 5-tuple we are to use
		# window tuple: ([0] start range, [1] end range, [2] data_tuple, [3] base length, [4][base indices where len - 1 corresponds to the number of bases in this tuple])
		window = [(-ctx_length, 0, None, 1, [-1 for _ in range(ctx_length + 1)])] # store dummy values in the array to indicate the number of bases to take from the initial window tuple 
		bases_in_window = len(window[-1][4]) - 1
		pivot = -1 # the tick point where the input range ends and the target range starts
		wpi = 0 # window pivot index
		wpip = ctx_length - 1 # window pivot index pivot, that is, the index in the indices array of the pivot, initial value is the last index in the dummy window tuple
		t_end = data_tuples[-1][4] # the end of time, that is, when the context ends at this time, the input should contain the END_TOKEN and then we are done
		# loop and generate input, target pairs
		
		init_text = "[SEQ_GEN]: Init loop "
		while win_end < t_end: # used to be <= to include the end as context at which point the target was the end_of_song token which is now excluded
			if p:
				print_divider(init_text, "=")
			
			# 1. make sure the window contains the necessary components to produce the next pair
			# there will always be a 5-tuple in window, even though the number of vectors to take from it might be 0, but the window represents the current 
			# time even if there is silence at that time
			while bases_in_window < segment_length:
				if window[-1][2] is None or (len(window) > 0 and window[-1][1] == window[-1][2][4]):
					# either the window is empty or the last data_tuple in the window has already been fully covered so we have to add a new tuple
					if dt_index < len(data_tuples): # there are more tuples to process
						base = get_base(data_tuples[dt_index][2])
						
						if p:
							print("[SEQ_GEN]: Added tuple with base", base, "consisting of the time", data_tuples[dt_index][3], "-", data_tuples[dt_index][4])
						#assert((data_tuples[dt_index][4] - data_tuples[dt_index][3]) >= base), "Illegal chunk: contains less than one unit of the chosen base"
						#assert(((data_tuples[dt_index][4] - data_tuples[dt_index][3]) % base) == 0), "Illegal chunk: chosen base does not divide the range"
						# each window component tuple contains (start range, end range, data_tuples, base, [base indices where len - 1 corresponds to the number of bases])
						window += [(win_end, win_end, data_tuples[dt_index], base, [0])]
						dt_index += 1
				win_end = win_end + window[-1][3] # expand the frontier by the sequence_base according to the concerned window
				bases_in_window += 1
			
				# 2. Calculate the end index of the added base and add this index to the indices of the last window tuple
				base_index = window[-1][4][-1] # the index of the first non-included data vector from the last yielded pair
				while base_index < len(window[-1][2][1]) and \
					(window[-1][2][1][base_index][1] < win_end or \
					(window[-1][2][1][base_index][1] == win_end and window[-1][2][1][base_index][0])):
					# expand the last used final index so that the last interval contains the sequence_base over which we have expanded last
					base_index += 1		
				window[-1] = (window[-1][0], win_end, window[-1][2], window[-1][3], window[-1][4] + [base_index]) # add the final range to the window
			wpip += 1
			if wpi < 0 or wpip == len(window[wpi][4]): # the indices are EXCLUSIVE and so when wpip == len(window[wpi][4] -1, it is really the same as index 0 in the next wpi so at this point, we reset wpip to 1
				wpi += 1
				wpip = 1
			pivot += window[wpi][3] # last end time
			if p:
				print("[SEQ_GEN]: Before window extraction - current tuple pivot", wpi, ", base indices pivot", wpip, ", indices in the current tuple pivot", window[wpi][4])
				print("[SEQ_GEN]: Before window extraction - window size:", len(window), ", window start:", win_start, ", pivot time:", pivot, ", window end:", win_end, ", last window end: ", window[-1][1], ", total end:", t_end)

			# 3. Process the window and generate the actual output patterns, start by finding 
			orig_pivot_indices = window[wpi][4] # let's split up the pivot window indices temporarily so that the function calls get only relevant part of the window
			window[wpi] = (window[wpi][0], window[wpi][1], window[wpi][2], window[wpi][3], orig_pivot_indices[: wpip + 1])
			ctx = generate_window(win_start, pivot, window[: (wpi + 1) if wpip > 0 else wpi])
			window[wpi] = (window[wpi][0], window[wpi][1], window[wpi][2], window[wpi][3], orig_pivot_indices[wpip:]) # last tuple only keeps its last base range
			inp = generate_window(pivot, win_end, window[wpi if wpip < (len(orig_pivot_indices) - 1) else wpi + 1: ])
			window[wpi] = (window[wpi][0], window[wpi][1], window[wpi][2], window[wpi][3], orig_pivot_indices) # restore the window	
			
			if p:
				print("[GEN_SEQ]: shape of generated ctx:", ctx.shape, "and inp:", inp.shape)
			if inp is None or ctx is None:
				print("yyy")
				exit()
			else:
				yield (ctx, inp)
			
			# 4. update state variables for the next iteration
			win_start = window[0][0] + window[0][3]
			bases_in_window -= 1
			
			if p:
				print("[GEN_SEQ]: window pivot tuple / window pivot tuple base index BEFORE end-of-loop updating:", wpi, "/", wpip)
			window[0] = (window[0][0] + window[0][3], window[0][1], window[0][2], window[0][3], window[0][4][1:]) # pop the first index out of the set of indices
			if wpi == 0: # compensate for the fact that we just removed one base index, if the pivot base index refers to an index in the 0'th window where one base index was just removed
				wpip -= 1
			if window[0][0] == window[0][1]:
				# this tuple is exhausted, remove it
				window = window[1:]	
				wpi -= 1
			if p:
				print("[GEN_SEQ]: window pivot tuple / window pivot tuple base index AFTER end-of-loop updating:", wpi, "/", wpip)
			
	###############################################################################################################################################################

	def midi_to_tokens(self, filename, adjustments = [], range_start = None, range_stop = None, neglected_time_signatures = []):
		try:
			f = mido.MidiFile(filename)
		except:
			assert(False), "[TOK]: ERROR - Mido couldn't parse the file"
		
		orig_tpb = f.ticks_per_beat
		
		tpb = orig_tpb
		factor = 1.0
			
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################

		##############################################################################################################################################################
		# TOKENIZATION
		#
		# 1. RECORD TRACKS AND TURN TRACKS INTO A SINGLE STREAM OF MIDI EVENTS
		# 2. TOKENIZE THE STREAM OF EVENTS, THIS INCLUDES TURNING EACH EVENT INTO A NOTE TOKEN WITH A DURATION, A MAPPING TO AN INSTRUMENT AND MORE. SOME QUANTIZATION
		#	 TAKES PLACE WHEREBY TOKENS ARE PROLONGED TO THE NEXT TOKEN IN THE SAME TRACK (IF UNDER A CERTAIN THRESHOLD) SO COUNTER STACCATOS. NOTES ARE ALSO SHORTENED
		#	 WHEN SUSTAINED OVER A NEW NOTE IN THE SAME TRACK (ALSO UNDER A THRESHOLD).
		##############################################################################################################################################################

		print("[TOK]: (!!!) START PROCESSING of file '", filename, "' with ", tpb, " ticks per beat, (originally", orig_tpb, "tick per beat)")
		if self.P_INSTRUMENTS:
			print("\r\n(!!!) INSTRUMENT MAPPINGS:")
			for i in range(len(self.inst.original_names)):
				print(i, self.inst.original_names[i], " translates to (", self.inst.output_mappings[self.inst.internal_mappings[i]], ") ", self.inst.mapped_names[i])
		print("[TOK]: (!!!) STARTING STREAMING PROCESS:")
		tokens, tracks_no = self.tracks_to_stream(f, adjustments, factor, tpb * 4 * 3)
		print("      ->Input streamed into", len(tokens), "tokens over", tracks_no, "tracks")
		print("[TOK]: (!!!) STARTING TOKENIZATION PROCESS:")
		tokens, time_signatures, stat_pitches = self.tokenize(tokens, self.inst, tpb, tracks_no, range_start, range_stop, None, None, neglected_time_signatures) # 476160 (-160), 624960 (+480), 826560 (+480)
			
		if self.P_TOKENIZATION:
			print("\r\n[TOK]: (!!!) TOKENIZATION FINISHED: Printing pitch statistics for this song: ")
			for i, pitch in enumerate(stat_pitches):
				print("--> ", i, pitch)
				
			print("\r\n[TOK]: (!!!) TOKENIZATION FINISHED: Printing tokens: ")
			print("Length of tokens: ", len(tokens))
			print("Time signatures detected: ", len(time_signatures))
			
		assert(len(tokens) - len(time_signatures) > 0), "[TOK]: empty song with no notes"
		print("      ->Yielded", len(tokens), "tokens (note on and time signatures)")
		start_index = 0
		start_time = 0
		num = 4
		denom = 4
		add = "(DEF)"
		for ts in time_signatures:
			print("      ->Time signature @", start_time, ":", num, "/", denom, add + ": tokens", start_index,"(excl.)-", ts, "(excl.) (", max(ts-start_index-1, 0)," tokens)")
			add = ""
			num = tokens[ts].num
			denom = tokens[ts].denom
			start_index = ts
			start_time = tokens[ts].start
		
		print("      ->Time signature @", start_time, ":", num, "/", denom, add + ": tokens", start_index,"(excl.)-", len(tokens), "(excl.) (", max(len(tokens)-start_index-1, 0)," tokens)")
		
		return tokens, time_signatures, stat_pitches, tpb, tracks_no
		
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################		
	
	def tokens_to_processed_tokens(self, tokens, time_signatures, tpb, tracks_no):
		##############################################################################################################################################################
		# PROCESSING - PROCESS THE READ TOKENS
		#
		# The tokenization turned the input into an array of tokens for which we also have an array with the indices in this array, that corresponds to time 
		# signatures. Since we process tokens differently depending on time signature, we will now process each time signature range separately and in order. If a 
		# time signature is not valid (e.g. we cannot process it), then an interruption will occur in the input and the resulting chunks will be stored as 
		# independent songs in the output. After the initial step (process_tokens), only note onsets will be present in the processed tokens, same goes for the
		# second step (unify_tokens). After the final step (finalize_representation), the processed tokens will have separate onsets and offsets for each note.
		# All steps in this section uses tokens with an ABSOLUTE time stamp.
		#
		# 1. QUANTIZE AND ALIGN
		# 2. UNIFY SIMULTANEOUS EVENTS IN TRACKS
		# 3. CONVERT TO START AND END TOKENS THAT ARE DIRECTLY CONVERTIBLE TO DATA REPRESENTATION
		#
		# IMPORTANT:
		# The idea is that each chunk is a piece of contiguous music consisting of sub-ranges where each subrange is a separate time signature. Whenever we run
		# into a time signature that we can't process or a time signature that actually consists of time (not two changes at the same time) but is empty,
		# either as a result of processing tokens (all tokens were quantized to nothing) or as is, we consider it a break in the music and we don't see the
		# following music, if any, as contiguous with the first one. The reason for this book-keeping of times and ranges is two-fold. First of all keeping
		# track of the starting times and ending times of each sub range helps out in processing and the reason for keeping track of when the music starting after
		# a chunk should start is that we might connect chunks from DIFFERENT files to each other after this processing. Chunk in the same files will never be
		# connected again but if a call to midi_to_tokens is a list with several files, after each call to this function, we will try to connect the last chunk
		# of the first song to the first chunk of the second. Now, if either of these are empty, this won't result in anything useful but if they aren't, we might
		# have to update the starting times of all the tokens in the second chunk whereby we will use the chunk_offset data for this. Whenever we run into and empty
		# chunk, we reset the chunk_offset to 0. Even though we will never reattach chunks WITHIN a song, it is important that the automatic attachment of last / first
		# chunks in contiguous files work as intended: if the last chunk of A is empty and the first of B is not, attaching them won't result in any new tokens other
		# than those in B but we don't want to change the offsets of those, that's we we have to reset the chunk_offset. If the last chunk in A contains stuff, then
		# we might want to attach it to the first chunk in B and we then use the chunk offset of the former chunk to update all the event start in B's first chunk
		# before putting them together. Empty ranges will be removed in subsequent calls to processed_tokens_to_data.
		##############################################################################################################################################################
		p = self.P_ADJUSTMENT
		start = 0 # the index of the start token of the current range to process
		ds = self.dur.get_duration_set(4, 4, tpb) # initial time signature (default)
		triplet_ts = False # whether the current time signature is triplet based or not (6/8, 9/8, 12/8, 6/4, 9/4 and 12/4)
		processed_tokens = [] # holds a consecutive stream of tokens, possibly over different, consecutive time silence
		processed_chunks = [] # holds chunks of tokens that are consecutive streams, each chunk should be considered its own song since an interruption between chunks exist
		self.print_tokens(ds.durations, "CHANGING DURATION SET (4/4 " + str(tpb) + ")...", self.P_DURATIONS)
		total_cancelled = 0 # total cancelled notes due to no available channel for instrument to use
		interruption = False # whether to close the current chunk due to an interruption or not (interruption is an empty time frame or an invalid time signature)
		first = True # whether we are processing the first chunk of an input or not; only the first part should be stripped of initial silence, not consecutive parts
		offset = 0 # the initial offset in the first part of an input where silence has been removed (this is the amount of silence)
		chunk_offsets = [] # are only used in reality when we want to merge several input files into one, then the offset of each chunk is used as an offset to each event when merging
		i = -1 # index of the current time signature, -1 implies the initial default time signature
		# holds the positions in the current chunk, used to mark sub ranges, the offset holds the current offset between events an the start of the original file
		# needed for events to end up at the correct place in the current chunk
		current_start = 0
		ts_start = 0
		valid_ts = True
		start_time = 0
		end_time = 0
		last = False
		print("[PROC_TOK]: (!!!) STARTING QUANTIZATION, ALIGNMENT AND PURGING PROCESS:")
		
		# we keep track of two important variabels: the offset indicates what to add to a song for the song to end up correctly in the current chunk relative
		# to the original timing information which sees the entire song as one chunk. Offset get its values from three places: by removing initial silence in
		# the beginning of a chunk, by adding space to a subrange which ends with the final bar shorter than it is supposed to be (3 beats in a 4/4 sub range
		# for example) and whenever there is an interruption after which the next chunk restarts its time at 0 and the offset has to reflect this for the
		# events to come. The current_time variable stores the CURRENT TIME in the OUTPUT chunk, that is, if there is an interruption, it will restart at 0.
		# current start is used to mark each sub range with its correct start and ending position. Finally, the start_time indicates when, according to the input
		# time, the current time signature starts.. when we want to align the events to an output underlying grid, we need to subtract BOTH the offset to the current
		# chunk AND the offset to the current subrange since this will otherwise reflect previous time signatures in the same chunk. So, by adding the offset to an
		# event, we place it at its correct location in the current chunk but if we also want to make calculations based on the current time signature, we also need to
		# subtract the current_time variable which reflects the starting time fo the current time signature in the current chunk.
		while i < len(time_signatures):
			interruption = False
			if i >= 0:
				try:
					valid_ts = True
					# try to set up the next time signature
					start = time_signatures[i] + 1
					start_time = tokens[time_signatures[i]].start
					ts_start = start_time + offset
					next_ts = tokens[time_signatures[i]]
					next_ds = self.dur.get_duration_set(next_ts.num, next_ts.denom, tpb)
					print("[PROC_TOK]: New base duration:", next_ds.bar_duration.duration,"for time signature with num / denom", next_ts.num, "/", next_ts.denom) if p >= 1 else 0
					if next_ds.abstract_duration_set.ts_num in [6, 9, 12]:
						triplet_ts = True
					else:
						triplet_ts = False
					ds = next_ds
					self.print_tokens(ds.durations, "CHANGING DURATION SET (" + str(next_ts.num) + "/" + str(next_ts.denom) + " " + str(tpb) + ")...", self.P_DURATIONS)
				except ValueError as ae:
						valid_ts = False
						print("AssertionError: Illegal time signature detected, skipping subrange ...", ae.args)
			
			# find out time and index range
			if (i + 1) == len(time_signatures): # not more tokens are time signatures, set the end of the range to the last token
				end = len(tokens)
				end_time = tokens[-1].start
				last = True
			else:
				end = time_signatures[i + 1] # the current range ends with the next time signature found
				end_time = tokens[time_signatures[i + 1]].start	
			if first:
				current_start = 0
			else:
				current_start = start_time + offset
				print("[PROC_TOK]: Beginning of loop (ts " + str(i) +"), UPDATING current start to", current_start, "with starting time", start_time, "and offset", offset) if p >= 1 else 0
			
			if valid_ts: # an invalid time signature will make an interruption in the current input, store the remaining input as a separate song	
				# process  a range
				if start != end: # empty set of tokens, discriminated from the above since this case does not make an interruption in the music
					if first:
						offset = -start_time
						print("[PROC_TOK]: Beginning of loop (ts " + str(i) +"), RESETTING current start to", current_start, "with starting time", start_time, "and offset", offset) if p >= 1 else 0
						pre_silence = ((tokens[start].start + offset) - ((tokens[start].start + offset) % int(ds.bar_duration.duration)))
						offset -= pre_silence
						first = False # indicate that we no longer process the first time signature segment of a song
						print("[PROC_TOK]: Removing initial silence of size", pre_silence, "resulting in reinitialized offset", offset) if p >= 1 else 0
					processed = self.process_tokens(tokens[start: end], ds, tracks_no, triplet_ts, current_start, offset)
					if len(processed) > 0: # range might be empty after processing due to all notes being cancelled or only non-note events
						processed_tokens += [processed]
						# non-empty portion, continue with processing: unify, insert end notes, remove invalid notes and split long notes to bar long notes instead
						self.print_tokens(processed_tokens[-1], "PROCESSED TOKENS", self.P_ADJUSTMENT)
						processed_tokens[-1] = self.unify_tokens(processed_tokens[-1], True)
						processed_tokens[-1], cancelled = self.finalize_representation(processed_tokens[-1], ds)
						self.print_tokens(processed_tokens[-1], "PROCESSED TOKENS", self.P_FINALIZATION)
						processed_tokens[-1] = (ds, processed_tokens[-1], current_start) # store the duration set as well
						total_cancelled += cancelled
						print("[PROC_TOK]: Added successfully processed sub-range of size " + str(len(processed))) if p >= 1 else 0
					else:
						interruption = True # with a non-empty interruption
						print("[PROC_TOK]: Found a sub-range with all tokens cancelled causing an interruption") if p >= 1 else 0
				else: # 0 tokens in range, ignore if initial or if no time is in between
					if i >= 0 and start_time != end_time: 
						# can always ignore initial empty time signature, otherwise, only ignore time signatures that don't take any time
						interruption = True
						print("[PROC_TOK]: Found a time signature sub-range with non-zero time but no tokens causing an interruption") if p >= 1 else 0
						
			subrange_span = end_time - start_time	
			print("[PROC_TOK]: Original span of sub range is " + str(subrange_span) + " with end time " + str(end_time)) if p >= 1 else 0
			if valid_ts:
				if (subrange_span % ds.bar_duration.duration) != 0:
					# need the span to be a fixed number of bars
					bar_completion = ds.bar_duration.duration - (subrange_span % ds.bar_duration.duration)
					offset += bar_completion
					print("[PROC_TOK]: Adding bar completion " + str(bar_completion) + " to offset yielding " + str(offset)) if p >= 1 else 0
		
			if interruption or last or not valid_ts: # invalid or empty time signature causes an interruption in the music, store all the consecutive processed tokens together and start a new song for the remaining
				# we are interested in ALL interruptions, even initial interruptions, since the next chunk may otherwise be falsely linked to a previous file chunk
				if len(processed_tokens) > 0: # before the interruption, we had some data collected already
					processed_chunks += [processed_tokens]
					last_event_start = processed_tokens[-1][1][-1].start
					last_ds = processed_tokens[-1][0]
					last_subrange_span = last_event_start - processed_tokens[-1][-1]
					bar_completion = last_subrange_span % last_ds.bar_duration.duration
					if bar_completion != 0:
						last_event_start = last_event_start - bar_completion + last_ds.bar_duration.duration # let the ensuing start in the next measure
					chunk_offsets += [last_event_start]
					print("[PROC_TOK]: Filed NON-EMPTY chunk with " + str(len(processed_tokens)) + " sub-ranges ending at", last_event_start) if p >= 1 else 0
				else: # empty set, this chunk cannot be connected to this is ONLY relevant if this chunk is FIRST or LAST in the resulting chunks
					# of a song, otherwise it doesn't matter. Empty chunks are used when inputting multiple files together whereby the last chunk of the
					# first file will attempt to connect to the first chunk of the second (and so on). By having empty chunks, we show that there is an interruption
					# between the preceding music and the next. Otherwise empty chunks don't fill a role. Both empty chunks and empty subranges are removed in the
					# next function (processed_tokens_to_data)
					if i == -1 or last:
						processed_chunks += [processed_tokens]
	
						chunk_offsets += [0]
						print("[PROC_TOK]: Filed EMPTY chunk") if p >= 1 else 0
					else:
						print("[PROC_TOK]: Noticed EMPTY chunk") if p >= 1 else 0	
				processed_tokens = []
				first = True # the next chunk can well have its initial silence removed, at least, the same initial offset doesn't have to be used necessarily'
				# given that the offset consists of potential smaller adjustments due to unfinished bars and such, and larger parts due to the starting time
				# of a new chunk, current start already has the old offset included, we choose to keep it in here as well (as opposed to only using end_time since 
				# it's likely that the same smaller corrections applies throughout a file. This is a matter of taste.
			else:
				# the last time signature was valid and so duration_Set contains the time signature that just ended here, lets calculate the span of the
				# range, adjust the offset if it doesn't fill a full number of bars and update current_time accordingly (remember that if the last time signature
				# was invalid, we have to take its ending time for granted since we have no way of making sure that the most recent span amounts to a full number
				# of bars
				pass
			i += 1
			print_divider("[PROC_TOK]: End of loop ", "=") if p >= 1 else 0
		
		print("      ->Number of separate discontinuous chunks: ", len(processed_chunks))
		for i, chunk in enumerate(processed_chunks):
			print("      ->Chunk", i, "(" + str(len(chunk)), " ranges) with ensuing music starting after this chunk at time", chunk_offsets[i])
			for range_tuple in chunk:
				ds = range_tuple[0]
				tss = range_tuple[1]
				time_start = range_tuple[2]
				print("            ->", ds.abstract_duration_set.ts_num, "/", ds.abstract_duration_set.ts_denom, "@", time_start, "(", ds.tpb, "ticks / quarter,", ds.bar_duration.duration, "ticks / bar ): ", len(tss), "tokens")
				print("                  ->", tss[0])
				print("                  ->", tss[1])
				if len(tss) > 2:
					print("                  ->", tss[2])
				print("                  ...")
				if len(tss) > 2:
					print("                  ->", tss[-3])
				print("                  ->", tss[-2])
				print("                  ->", tss[-1])
		print("      ->Total cancelled notes due to channel clash: ", total_cancelled)
				
		return processed_chunks, total_cancelled, chunk_offsets
		
		##############################################################################################################################################################
		##############################################################################################################################################################
		##############################################################################################################################################################
		
	def processed_tokens_to_data(self, processed_chunks, offsets):
	
		##############################################################################################################################################################
		# TOKENS TO DATA REPRESENTATION - CONVERT THE TOKENS INTO THE DATA REPRESENTATION VECTORS
		#
		# The final step before the input has been converted to data representation uses a generator to take ranges from the input depending on the desired
		# size needed. The generator produces ranges until a 
		#
		# 1. TURN THE TOKENS INTO A DATA REPRESENTATION OF VECTORS WITH 1's and 0's
		##############################################################################################################################################################	
		print("[TO_DATA]: (!!!) STARTING CONVERSION TO DATA REPRESENTATION PROCESS:")
		output_data = [] # the data representation for all chunks of tokens from the previous step (non-consecutive)
		chunk_skipped = []
		for chunk_no, chunk in enumerate(processed_chunks): # these chunks are interrupted sets of one or several ranges of tokens from different time signatures
			data = [] # the data representation of a consecutive chunk of tokens
			ai = None
			an = None
			total_skipped = 0
			for i, range_tuple in enumerate(chunk): # each range tuple is a continuous range with one time signature
				duration_set = range_tuple[0] # the duration set used for this range
				subrange = range_tuple[1] # the actual range of processed tuples
				start_time = range_tuple[2]
				if i < (len(chunk) - 1):
					next = chunk[i + 1][2]
				else:
					next = None
				if len(subrange) > 0: # can be empty, this is where empty ranges disappear
					data_representation, statistics, an, ai, skipped, bv = self.to_data_representation(subrange, duration_set, start_time, next, an, ai)
					total_skipped += skipped
					if data_representation is None:
						print("Error")
						exit()
					# remember the last time step of this subrange along with the data representation of the subrange
					if next is None:
						next = offsets[chunk_no]
					data += [(data_representation, statistics, duration_set, start_time, next, bv)]
			if len(data) > 0:
				output_data += [data] # the output stores data representations of interrupted chunks separately
				chunk_skipped += [total_skipped]
		################################################################################################
		
		print("      ->Number of separate discontinuous data chunks: ", len(output_data))
		for i, data_chunk in enumerate(output_data):
			print("      ->Data chunk", i, "with", len(data_chunk), "subrange of data (", chunk_skipped[i], "notes skipped ):")
			for range_tuple in data_chunk:
				data = range_tuple[0]
				st = range_tuple[1]
				ds = range_tuple[2]
				print("            ->Time", range_tuple[3], "to", range_tuple[4], ":", len(data), "vectors of data")	

		return output_data, chunk_skipped
		
	def midi_to_data(self, filenames, sequence_length = 1, stride = 1):
	
		chunks = []
		pitch_statistics = np.zeros((1, self.MAX_PITCHES), dtype="int")
		for i, filename in enumerate(filenames):
			if isinstance(filename, tuple):
				tokens, time_signatures, stat_pitches, tpb, tracks_no = self.midi_to_tokens(filename[0], filename[1], filename[2], filename[3], filename[4])
			else:
				tokens, time_signatures, stat_pitches, tpb, tracks_no = self.midi_to_tokens(filename)
			pitch_statistics += stat_pitches
			processed_chunks, total_cancelled, chunk_offsets = self.tokens_to_processed_tokens(tokens, time_signatures, tpb, tracks_no)
			if i == 0:
				chunks = processed_chunks
				offsets = chunk_offsets
			else:
				for j, range_tuple in enumerate(processed_chunks[0]):
				# increase the starting time for all tokens in this first (possibly empty) chunk accordingly to the previous final range
					for token in range_tuple[1]:
						token.start += offsets[-1]
					processed_chunks[0][j] = (processed_chunks[0][j][0], processed_chunks[0][j][1], processed_chunks[0][j][2] + offsets[-1])
				chunk_offsets[0] += offsets[-1] # increase the chunk offsets for next iteration
				chunks = chunks[: -1] + [chunks[-1] + processed_chunks[0]] + processed_chunks[1:]
				offsets = offsets[:-1] + chunk_offsets					
		print("[MIDI_TO_DATA]: (!!!) RESULTS AFTER STEP 1 MERGING PROCESS:")
		print("      ->Number of separate discontinuous chunks: ", len(chunks))
		
		for i, chunk in enumerate(chunks):
			print("      ->Chunk", i, "with ensuing music starting after this chunk at time", offsets[i])
			for range_tuple in chunk:
				ds = range_tuple[0]
				tss = range_tuple[1]
				time_start = range_tuple[2]
				print("            ->", ds.abstract_duration_set.ts_num, "/", ds.abstract_duration_set.ts_denom, "@", time_start, "(", ds.tpb, "ticks / quarter,", ds.bar_duration.duration, "ticks / bar ): ", len(tss), "tokens")	
				print("                  ->", tss[0])
				print("                  ->", tss[1])
				if len(tss) > 2:
					print("                  ->", tss[2])
				print("                  ...")
				if len(tss) > 2:
					print("                  ->", tss[-3])
				print("                  ->", tss[-2])
				print("                  ->", tss[-1])
		for i in range(len(chunks)):
			j = 1
			while j < len(chunks[i]):
				if chunks[i][j - 1][0] == chunks[i][j][0]: # same time signature in both ranges, merge them
					chunks[i] = chunks[i][:(j - 1)] + [(chunks[i][j - 1][0], chunks[i][j - 1][1] + chunks[i][j][1], chunks[i][j - 1][2])] + chunks[i][(j + 1):]
					chunks[i][j - 1][1].sort(key = lambda x: (x.start, int(x.is_end()), x.pitch)) # sort in case the two segments overlap
				else:
					j += 1
		
		print("[MIDI_TO_DATA]: (!!!) RESULTS AFTER STEP 2 MERGING PROCESS:")
		print("      ->Number of separate discontinuous chunks: ", len(chunks))
		for i, chunk in enumerate(chunks):
			print("      ->Chunk", i, "with ensuing music starting after this chunk at time", offsets[i])
			for range_tuple in chunk:
				ds = range_tuple[0]
				tss = range_tuple[1]
				time_start = range_tuple[2]
				print("            ->", ds.abstract_duration_set.ts_num, "/", ds.abstract_duration_set.ts_denom, "@", time_start, "(", ds.tpb, "ticks / quarter,", ds.bar_duration.duration, "ticks / bar ): ", len(tss), "tokens")		
				print("                  ->", tss[0])
				print("                  ->", tss[1])
				if len(tss) > 2:
					print("                  ->", tss[2])
				print("                  ...")
				if len(tss) > 2:
					print("                  ->", tss[-3])
				print("                  ->", tss[-2])
				print("                  ->", tss[-1])					
				
		output_data, chunk_skipped = self.processed_tokens_to_data(chunks, offsets)	
		return output_data, len(chunks), pitch_statistics

	def data_to_midi(self, data, fill = False, default_duration = "d8"):
		# data contains one or several ARRAYS with TUPLES, where each tuple is on the form ... 
		# (data vectors, statistics about the vector times steps, the duration set used, the start time and end time of the chunk (NOT the time of first and last token necessarily))
		# instead of tuples, constructs maybe provided in the arrays such that the actual music vectors can be found at construct[0]
		# one array corresponds to one contiguous piece of music and will be seen as such
		files = []
		for chunk in data: # go through the arrays and feed each and one as a contiguous chunk of music to the from data representation method
			# concatenate all data vectors belonging to the same chunk
			tokens = self.from_data_representation(chunk, fill, default_duration)
			
			# if this chunk is a midification of a generated song, it will be 2/4, 3/4 or 4/4 and triplet signature, such as 6/8, 9/8 or 12/8 will be notated as triplets in these prior duple time signatures. However, when 
			# translating back from the intermediate data representation, other time signatures may occur and then we must translate the tpb back to its initial value. Remember that when altering the tpb, it is just a means
			# to translate the incoming pitches to the same intermediate data representation, and so we must translate it back. An entire chunk must have the same tpb, initially so it does not matter which range of a chunk we look
			# at, simply choose the first, translate its tpb back to midi standard, if necessary, and use it in the midification. Generated music 8and all music with 2, 3, 4/4 will pass by unnoticed.
			ds = chunk[0][2]
			tpb_to_use = ds.tpb
			multiplier = 1.0
			if ds.abstract_duration_set.ts_num % 3 != 0 or ds.abstract_duration_set.ts_num == 3: # duple time
				if ds.abstract_duration_set.ts_denom == 2:
					multiplier = 2.0 # half-note-based time signature
				elif ds.abstract_duration_set.ts_denom == 8:
					multiplier = 0.5 # eighth-based time signature
			else: # triplet time
				multiplier = 1.5 # the value of a duplet quarter will increase by 1.5, thus interpreting actual quarters as triplet quarters as desired
				if ds.abstract_duration_set.ts_denom == 4:
					multiplier *= 2.0 # quarter-based triplet time maps to half note based duple time
			tpb_to_use /= multiplier # divide by the multiplier instead of multiplying by it as we did initially when we first encountered a time signature where modification was necessary.
					
			out = self.midify(tokens, self.inst, tpb = int(tpb_to_use))
			files += [out]
		return files
