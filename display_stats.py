import matplotlib.pyplot as plt
import sys, os, pickle, os.path, math, itertools
import numpy as np

X_TICKS_MAX = 20
X_POINTS_MAX = 20

DEFAULT_COLORS = ["b", "g", "r", "c", "m", "y", "k", "b--"]

def curves(data, labels, title, xlabel, ylabel):
		step = 1
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		max_x = max(list(map(lambda a: len(a), data)))
		max_y = max(list(map(lambda a: max(a), data)))
		plt.xlim(0, max_x + 1)
		plt.ylim(0, max_y + (max_y / 20.0))
		plt.title(title)
		plots = []
		for i, (vector, label) in enumerate(zip(data, labels)):
			plots += [plt.plot(np.arange(1, len(vector) + 1, 1), vector, DEFAULT_COLORS[i % len(DEFAULT_COLORS)], label = label)]
		x_range = np.arange(1, max_x + 1, step)
		while len(x_range) > X_TICKS_MAX:
			step += 1
			x_range = np.arange(1, max_x + (step - (max_x % step)) + 1, step)
		plt.xticks(x_range)
		plt.legend(tuple(list(map(lambda a: a[0], plots))), tuple(labels))
		plt.grid()
		plt.show()
			
def read(args):
	mode = args[0]
	rest = args[1:]
	names = rest[::2]
	paths = rest[1::2]
	assert(len(names) == len(paths)), "must supply the same number of labels and paths to statistics files"
	num = len(names)
	stats = {"epoch_l": [0] * num, "step_l": [0] * num, "epoch_p_r": [0] * num, "step_p_r": [0] * num, "epoch_d": [0] * num, "step_d": [0] * num}
	for i, path in enumerate(paths):
		with open(os.path.join(path, "records", "stats"), mode = "rb") as stats_f:
			(stats["epoch_l"][i], stats["step_l"][i], stats["epoch_p_r"][i], stats["step_p_r"][i], stats["epoch_d"][i], stats["step_d"][i]) = pickle.load(stats_f)
	return mode, stats, names
	
if __name__ == "__main__":
	if len(sys.argv) < 3:
		print("[STATS]: please supply a full path to the statistics file followed by \"l\", \"d\" or \"p\" only for loss, distance and precision metrics")
		exit()
	else:
		mode, stats, labels = read(sys.argv[1:])
		
		if mode == "l":
			stats = list(itertools.chain.from_iterable([[(stat[0], label + "_TR_EPOCH_LOSS")] if np.max(stat[1]) == 0.0 else [(stat[0], label + "_TR_EPOCH_LOSS"), (stat[1], label + "_VA_EPOCH_LOSS")] for (stat, label) in zip(stats["epoch_l"], labels)]))
			for (stat, label) in stats:
				print(label, list(zip(range(1, len(stat)), stat)))
			data, labels = zip(*stats)
			curves(data, labels, 'Avg. loss / epoch', "epochs", "loss")
		elif mode == "p":
			key = "p"
			for key in ["o", "d", "p", "i"]:
				cat_stats = list(itertools.chain.from_iterable([[(stat[0], label + "_TR_EPOCH_REC%_" + key)] if np.max(stat[1]) == 0.0 else [(stat[0], label + "_TR_EPOCH_REC%_" + key), (stat[1], label + "_VA_EPOCH_REC%_" + key)] for (stat, label) in zip([s[key] for s in stats["epoch_p_r"]], labels)]))
				for (stat, label) in cat_stats:
					print(label, list(zip(range(1, len(stat)), stat)))
					print("MAX_EPOCH_P for key " + key + ": " + str(np.max(stat)))
				data, labels_to_use = zip(*cat_stats)
				curves(data, labels_to_use, 'Avg. recon. perc. / epoch', "epochs", "%")
		elif mode == "d":
			for key in ["o", "d", "p"]:
				print("EPOCH DST", [(l, label) for (l, label) in zip(list(map(lambda a: a[key][0], stats["epoch_d"])), labels)])
				print("MIN EPOCH DST", [(l, label) for (l, label) in zip(list(map(lambda a: min(a[key][0]), stats["epoch_d"])), labels)])
				curves(list(map(lambda a: a[key][0], stats["epoch_d"])), labels, "Dist. / epoch", "epoch", "classes")
		elif mode == "ls":
			curves(list(map(lambda a: a[0], stats["step_l"])), labels, 'Avg. loss / step', "steps", "loss")
