"""
Cache in files
"""
import gzip
import os
import pickle
import sys
from functools import lru_cache, wraps



    
def LOCK_EX():
    return 0
        
def lockf(fd, operation, length=0, start=0, whence=0):
    pass

class FileSystemMutex:
    """
    Mutual exclusion of different **processes** using the file system
    """

    def __init__(self, filename):
        self.handle = None
        self.filename = filename

    def acquire(self):
        """
        Locks the mutex
        if it is already locked, it waits (blocking function)
        """
        self.handle = open(self.filename, "w")
        lockf(self.handle, LOCK_EX)
        self.handle.write("{}\n".format(os.getpid()))
        self.handle.flush()

    def release(self):
        """
        Unlock the mutex
        """
        if self.handle is None:
            raise RuntimeError()
        lockf(self.handle, LOCK_EX)
        self.handle.close()
        self.handle = None

    def __enter__(self):
        self.acquire()

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()


def cached_dirpklgz(dirname, maxsize=128):
    """
    Cache a function with a directory

    :param dirname: the directory path
    :param maxsize: maximum size of the RAM cache (there is no limit for the directory cache)
    """

    def decorator(func):
        """
        The actual decorator
        """

        @lru_cache(maxsize=maxsize)
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            The wrapper of the function
            """
            try:
                os.makedirs(dirname)
            except FileExistsError:
                pass

            indexfile = os.path.join(dirname, "index.pkl")
            mutexfile = os.path.join(dirname, "mutex")

            with FileSystemMutex(mutexfile):
                try:
                    with open(indexfile, "rb") as file:
                        index = pickle.load(file)
                except FileNotFoundError:
                    index = {}

                key = (args, frozenset(kwargs), func.__defaults__)

                try:
                    filename = index[key]
                except KeyError:
                    index[key] = filename = "{}.pkl.gz".format(len(index))
                    with open(indexfile, "wb") as file:
                        pickle.dump(index, file)

            filepath = os.path.join(dirname, filename)

            try:
                with FileSystemMutex(mutexfile):
                    with gzip.open(filepath, "rb") as file:
                        result = pickle.load(file)
            except FileNotFoundError:
                print("compute {}... ".format(filename), end="")
                sys.stdout.flush()
                result = func(*args, **kwargs)
                print("save {}... ".format(filename), end="")
                sys.stdout.flush()
                with FileSystemMutex(mutexfile):
                    with gzip.open(filepath, "wb") as file:
                        pickle.dump(result, file)
                print("done")
            return result

        return wrapper

    return decorator
