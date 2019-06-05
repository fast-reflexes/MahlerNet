class Instruments:

	def __init__(self):
		# maps internal representations of a subset of GM instruments to their 0-indexed actual numbers, 128 stands for ignored instruments
		self.output_mappings = [0, 16, 27, 33, 40, 41, 42, 43, 48, 47, 52, 56, 57, 58, 60, 61, 66, 69, 70, 71, 73, 80, 88, 128]
		# maps 0-indexed GM instruments to their internal mappings of a limited subset of the GM instruments, 23 stands for ignored instruments
		self.internal_mappings = [0] * 16 + [1] * 8 + [2] * 8 + [3] * 8 + list(range(4, 8)) + [23] + [2] + [23] + [9] + [8] * 4 + [10] * 3 + [8] + list(range(11, 14)) + [11] + [14] + \
						[15] * 3 + [16] * 4 + [17] * 2 + list(range(18, 20)) + [20] * 8 + [21] * 8 + [22] * 8 + [22] * 8 + [2] * 4 + [0] + [17] + [4] + [17] + [23] * 16 + [23]
		self.original_names = [\
			'Acoustic Grand Piano', 'Bright Acoustic Piano', 'Electric Grand Piano', 'Honky-tonk Piano', 'Electric Piano 1', 'Electric Piano 2', 'Harpsichord', 'Clavinet', 'Celesta', 'Glockenspiel',\
			'Music Box', 'Vibraphone', 'Marimba', 'Xylophone', 'Tubular Bells', 'Dulcimer', 'Drawbar Organ', 'Percussive Organ', 'Rock Organ', 'Church Organ',\
			'Reed Organ', 'Accordion', 'Harmonica', 'Tango Accordion', 'Acoustic Guitar (nylon)', 'Acoustic Guitar (steel)', 'Electric Guitar (jazz)', 'Electric Guitar (clean)', 'Electric Guitar (muted)', 'Overdriven Guitar',\
			'Distortion Guitar', 'Guitar Harmonics', 'Acoustic Bass', 'Electric Bass (finger)', 'Electric Bass (pick)', 'Fretless Bass', 'Slap Bass 1', 'Slap Bass 2', 'Synth Bass 1', 'Synth Bass 2',\
			'Violin', 'Viola', 'Cello', 'Contrabass', 'Tremolo Strings', 'Pizzicato Strings', 'Orchestral Harp', 'Timpani', 'String Ensemble 1', 'String Ensemble 2',\
			'Synth Strings 1', 'Synth Strings 2', 'Choir Aahs', 'Voice Oohs', 'Synth Choir', 'Orchestra Hit', 'Trumpet', 'Trombone', 'Tuba', 'Muted Trumpet',\
			'French Horn', 'Brass Section', 'Synth Brass 1', 'Synth Brass 2', 'Soprano Sax', 'Alto Sax', 'Tenor Sax', 'Baritone Sax', 'Oboe', 'English Horn',\
			'Bassoon', 'Clarinet', 'Piccolo', 'Flute', 'Recorder', 'Pan Flute', 'Blown bottle', 'Shakuhachi', 'Whistle', 'Ocarina',\
			'Lead 1 (square)', 'Lead 2 (sawtooth)', 'Lead 3 (calliope)', 'Lead 4 (chiff)', 'Lead 5 (charang)', 'Lead 6 (voice)', 'Lead 7 (fifths)', 'Lead 8 (bass + lead)', 'Pad 1 (new age)', 'Pad 2 (warm)',\
			'Pad 3 (polysynth)', 'Pad 4 (choir)', 'Pad 5 (bowed)', 'Pad 6 (metallic)', 'Pad 7 (halo)', 'Pad 8 (sweep)', 'FX 1 (rain)', 'FX 2 (soundtrack)', 'FX 3 (crystal)', 'FX 4 (atmosphere)',\
			'FX 5 (brightness)', 'FX 6 (goblins)', 'FX 7 (echoes)', 'FX 8 (sci-fi)', 'Sitar', 'Banjo', 'Shamisen', 'Koto', 'Kalimba', 'Bagpipe',\
			'Fiddle', 'Shanai', 'Tinkle Bell', 'Agogo', 'Steel Drums', 'Woodblock', 'Taiko Drum', 'Melodic Tom', 'Synth Drum', 'Reverse Cymbal',\
			'Guitar Fret Noise', 'Breath Noise', 'Seashore', 'Bird Tweet', 'Telephone Ring', 'Helicopter', 'Applause', 'Gunshot', 'IGNORED']
		self.mapped_names =  [self.original_names[self.output_mappings[i]] for i in self.internal_mappings]
		self.sz_instrument_set = len(self.output_mappings) - 1
		
	def get_internal_mapping(self, instrument):
		return self.internal_mappings[instrument]
		
	def get_output_mapping(self, internal_mapping):
		return self.output_mappings[internal_mapping]
		
	def is_valid_instrument(self, instrument):
		return self.output_mappings[instrument] < 128