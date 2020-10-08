import sys,os

if __name__ == '__main__':
    files = os.listdir(sys.argv[1])
    lines = []

    for filename in files:
        try:
            file_ = open(sys.argv[1] + '/' + filename)
            lines.extend(file_.readlines())
            file_.close()
        except UnicodeDecodeError:
            print (filename)

    pos_samples = []
    neg_samples = []

    for line in lines:
        try:
            splitted = [l for l in line.split('\t')]
            if float(splitted[3]) > 2.8:
                pos_samples.append(str({'question':splitted[0],'source':splitted[1],'answer':splitted[2]}) + '\n')
            elif float(splitted[3]) < 1.00:
                neg_samples.append(str({'question':splitted[0],'source':splitted[1],'answer':splitted[2]}) + '\n')
        except UnicodeDecodeError:
            print (line)

    
    pos = open(sys.argv[2],'w')
    neg = open(sys.argv[3],'w')
    
    
    pos.writelines(pos_samples)
    neg.writelines(neg_samples)
    
    pos.close()
    neg.close()
