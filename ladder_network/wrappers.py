import functools


class Registry(object):
    """Wrapper to register various python functions."""
    def __init__(self, collection):
        self.collection = collection
        self.init_map = {}

    def register(self, name):
        def init_wrapper(func):
            self.init_map[name] = func
            return func
        return init_wrapper

    def get(self, name, default=None):
        func = self.init_map.get(name, default)
        if func is None:
            raise ValueError('`{}` not a valid {}.'.format(name, self.collection))
        return func


class TFCollectionRegistry(Registry):
    def register(self, name):
        def init_wrapper(func):
            self.init_map[name] = func

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                output = func(*args, **kwargs)
                tf.add_to_collection(self.collection, output)
                return output
            return func
        return init_wrapper
