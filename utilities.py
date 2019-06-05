import os

def print_divider(text, divider_symbol):
	if len(divider_symbol) != 1:
		divider_symbol = "-"
	rows, columns = os.popen('stty size', 'r').read().split()
	fill = "".join([divider_symbol] * (int(columns) - len(text) - 1))
	print(text + fill)
	