import time


class Timer(object):
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()
        # print('Início de [%s]' % self.name, )
        print("Starting -> " + str(self.name))

    def __exit__(self, type, value, traceback):
        print(
            "finnished -> "
            + str(self.name)
            + " - Total time: "
            + str((time.time() - self.tstart))
        )
        # print('[%s]' % self.name,)
        # print('Elapsed: %s' % (time.time() - self.tstart))
