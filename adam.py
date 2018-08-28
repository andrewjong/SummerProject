import time
from joblib import Parallel, delayed
import itertools

if __name__ == '__main__':
    functions = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions.append((x,y,z,w))
    functions2 = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions2.append((x,y,z,w))
    functions3 = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions3.append((x,y,z,w))
    functions4 = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions4.append((x,y,z,w))
    functions5 = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions5.append((x,y,z,w))
    functions6 = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions6.append((x,y,z,w))
    functions7 = []
    for x in [0,1]:
        for y in [0,1]:
            for z in [0,1]:
                for w in [0,1]:
                    functions7.append((x,y,z,w))
    def meme(input):
        function, function2 = input
        final = set()
        for function3 in functions3:
            for function4 in functions4:
                for function5 in functions5:
                    for function6 in functions6:
                        for function7 in functions7:
                            tree_function = []
                            for x in [0,1]:
                                for y in [0,1]:
                                    for z in [0,1]:
                                        for w in [0,1]:
                                            for x2 in [0,1]:
                                                for y2 in [0,1]:
                                                    for z2 in [0,1]:
                                                        for w2 in [0,1]:
                                                            tree_function.append(function7[function[function2[x+y*2] + function3[z+w*2]] + 2*function4[function5[x2+y2*2] + function6[z2+w2*2]]])
                            final.add(tuple(tree_function))
        return len(set(final))
    labels = Parallel(n_jobs=-1,backend="multiprocessing")(map(delayed(meme), itertools.product(functions,functions2)))
    print(sum(labels))
