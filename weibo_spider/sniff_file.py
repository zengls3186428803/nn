import os

def sniff_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return False
    return True

def sniff_file(path):
    if not os.path.exists(path):
        return False
    else:
        return True
