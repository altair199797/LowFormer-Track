

def to_file(string, path):
    with open(path,"w") as write_file:
        write_file.write(string)