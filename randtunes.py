import random
import sys

notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G',
         'a', 'b', 'c', 'd', 'e', 'f', 'g']

f = open(sys.argv[1], 'r')
lengths = []
for line in f:
    lengths.append(len(line) - 1)

for l in lengths:
    tune = ''
    for i in range(0, l/2):
        tune = tune + notes[random.randint(0, 13)]
    print "%s%s" % (tune, tune)
