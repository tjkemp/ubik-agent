import inspect


def get_methods(cls, excluded_methods=['cli']):

    methods = [
        item for item in dir(cls) if
        not item.startswith('_') and
        item.islower() and
        inspect.ismethod(getattr(cls, item))]

    for method in excluded_methods:
        methods.remove(method)

    methods_and_args = {}
    for method in methods:
        func = getattr(cls, method)
        sig = inspect.signature(func)

        arguments = []
        for param in sig.parameters.values():
            param_name = param.name

            param_default = None if param.default is inspect.Parameter.empty else param.default
            kw_arg = True if param.default is not inspect.Parameter.empty else False

            if param.annotation is not inspect.Parameter.empty:
                param_type = param.annotation
            elif type(param_default) is int:
                param_type = int
            elif type(param_default) is bool:
                param_type = bool
            else:
                param_type = None

            # print(inspect.getdoc(func))
            param_doc = None
            arguments.append((param_name, kw_arg, param_default, param_type, param_doc))
        methods_and_args[method] = arguments

    return methods_and_args
