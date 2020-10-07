import sys

if __name__ == '__main__':
    f = open(sys.argv[1])
    lines = f.readlines()
    f.close()

    pos_samples = []
    neg_samples = []

    for line in lines:
        splitted = [l for l in line.split('\t')]
        if float(splitted[3]) > 3.0:
            pos_samples.append(str({'question':splitted[0],'source':splitted[1],'answer':splitted[2]}) + '\n')
        else:
            neg_samples.append(str({'question':splitted[0],'source':splitted[1],'answer':splitted[2]}) + '\n')

    
    pos = open(sys.argv[2],'w')
    neg = open(sys.argv[3],'w')
    
    
    pos.writelines(pos_samples)
    neg.writelines(neg_samples)
    
    pos.close()
    neg.close()
