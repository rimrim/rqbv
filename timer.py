import time
class Timer(object):
    """a snippet for timing measurments"""
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __enter__(self):
        """docstring for __enter__"""
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000
        if self.verbose:
            print('elapsed time: {:f} ms'.format(self.msecs))
            
        
        
