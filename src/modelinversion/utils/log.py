import sys

class Tee(object):
    """A workaround method to print in console and write to log file
    """
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout
        sys.stdout = self
    def __del__(self):
        sys.stdout = self.stdout
        self.file.close()
    def write(self, data):
        if not '...' in data:
            self.file.write(data)
        self.stdout.write(data)
    def flush(self):
        self.file.flush()
