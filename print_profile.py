import sys

import pstats

def main(n, col):
	p = pstats.Stats('profile.txt')
	p.sort_stats(col).print_stats(n)

if __name__ == '__main__':
	n = sys.argv[1] if len(sys.argv) > 1 else 10
	col = sys.argv[2] if len(sys.argv) > 2 else 'cumulative'
	main(int(n), col)
