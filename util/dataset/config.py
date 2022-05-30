class ConfigFun:
    def __init__(self, fun, config):
        self.fun = fun
        self.config = config

    def __config__(self):
        return self.config

    def __call__(self, *args, **kwargs):
        return self.fun(*args, **kwargs)

def fun(config):
    def do_wrap(fun):
        return ConfigFun(fun, config)
    return do_wrap