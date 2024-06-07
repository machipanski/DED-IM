import time


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        # print('Início de [%s]' % self.name, )
        print(self.name)

    def __exit__(self, type, value, traceback):
        print("Total de [" + str(self.name) + "] : " + str((time.time() - self.tstart)))
        # print('[%s]' % self.name,)
        # print('Elapsed: %s' % (time.time() - self.tstart))
