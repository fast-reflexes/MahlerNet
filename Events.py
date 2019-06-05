
class Event:

	def __init__(self, start):
		self.start = start

	def __str__(self):
		return "Event @ " + str(self.start)
		
class TimeSignature(Event):
	
	def __init__(self, start, num = 4, denom = 4):
		Event.__init__(self, start)
		self.num = num
		self.denom = denom
		
	def __str__(self):
		return "TimeSignature @ " + str(self.start) + ": " + str(self.num) + " / " + str(self.denom)		
		
class Tempo(Event):
	
	def __init__(self, start, mspq = 50000):
		Event.__init__(self, start)
		self.mspq = mspq
		
	def __str__(self):
		return "Tempo @ " + str(self.start) + ": " + str(self.mspq) + " ms / quarter note"

class ChannelEvent(Event):

	def __init__(self, start, channel):
		Event.__init__(self, start)
		self.channel = channel
		
	def __str__(self):
		return "ChannelEvent @ " + str(self.start) + " on channel " + str(self.channel)		
		
class Note(ChannelEvent):
	
	def __init__(self, start, pitch, channel, instrument, instrument_map, track = None, tempo = None):
		ChannelEvent.__init__(self, start, channel)
		self.pitch = pitch
		self.channel = channel
		self.end = -1 # the index of the end event of this note, if it is a starting note that is
		self.duration = 0 # duration of 0 means it is an end of a note, > 0 means it's a start of a note and -1 means that the note is invalid
		self.duration_partials = list([])
		self.instrument = instrument
		self.inst = instrument_map
		self.pitch_map = ["A", "Bb", "B", "C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab"]
		self.track = track
		self.tempo = tempo
		
	def __str__(self):
		if self.is_valid():
			if self.is_start():
				return "Valid start Note (" + str(self.inst.original_names[self.inst.get_output_mapping(self.instrument)]) + ") @ " + str(self.start) + ": pitch " + str(self.pitch) + "(" + self.to_note_name() + "), duration " + str(self.duration) + ", track " + str(self.track) + ", tempo " + str(self.tempo)
			else:
				return "End Note (" + str(self.inst.original_names[self.inst.get_output_mapping(self.instrument)]) + ") @ " + str(self.start) + ": pitch " + str(self.pitch) + "(" + self.to_note_name()  + "), track " + str(self.track) + ", tempo " + str(self.tempo)
		else:
			return "Invalid start (" + str(self.inst.original_names[self.inst.get_output_mapping(self.instrument)]) + ") Note @ " + str(self.start) + " on channel " + str(self.channel) + ": pitch " + str(self.pitch) + "(" + self.to_note_name() + "), duration " + str(self.duration) + ", track " + str(self.track) + ", tempo " + str(self.tempo)
		
	def __eq__(self, ob):
		return self.start == ob.start and self.pitch == ob.pitch and self.duration == ob.duration and self.instrument == ob.instrument
		
	def to_note_name(self):
		start = -2
		diff = self.pitch - (-3)
		octaves = int(diff / 12)
		octave = str(start + octaves)
		note_name = self.pitch_map[diff % 12] #chr(ord('A') + (diff  % 12))
		return note_name + octave
		
	def is_start(self):
		return self.duration > 0
	
	def is_end(self):
		return self.duration == 0
		
	def is_valid(self):
		return self.duration != -1
		
	def invalidate(self):
		self.duration = -1
		
	def invalidate_short_note(self, tokens, threshold):
		if self.is_start() and self.duration < threshold:
			self.invalidate()

class Program(ChannelEvent):
	
	def __init__(self, start, channel, program):
		ChannelEvent.__init__(self, start, channel)
		self.program = program
		
	def __str__(self):
		return "Program @ " + str(self.start) + " on channel " + str(self.channel) +": " + str(self.program)