import re
import sys
from IPython import embed 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

inp = sys.argv[1]

with open(inp) as f:
    log = f.readlines()
def output(i):
    return "     Train net output #{}: (.*?) = (\d+\.\d+)[ |\n]".format(i)

iteration = " Iteration (\d+), loss = "
loss = " Iteration \d+, loss = (\d+\.\d+)\n"
output(0)

def topd(pattern, log, matchname = True):
    prog = re.compile(pattern)
    if matchname:
        name = None
    matched = []
    for line in log:
        match = prog.findall(line)
        if len(match) != 0:
            if matchname:
                if name is None:
                    name = match[0][0]
                matched.append(match[0][-1])
            else:
                matched.append(match[0])
    if matchname:
        return name, pd.Series(matched, dtype=float)
    else:
        return pd.Series(matched, dtype=float)

iter = topd(iteration, log, matchname=False)

l = topd(loss, log, matchname=False)
d = {'loss':l}
for i in range(100):
    o = topd(output(i), log)
    if o[0] is None:
        break
    d[o[0]] = o[1]

df = pd.DataFrame(data=d)
df.index = iter

pd.rolling_mean(df, 128).plot()
plt.savefig(inp+'.png')
plt.show()