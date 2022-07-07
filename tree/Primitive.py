class Primitive:
    def __init__(self, name, return_type, parameter_types, function):
        self.__name = name
        self.__arity = len(parameter_types)
        self.__return_type = return_type
        self.__parameter_types = parameter_types
        self.__function = function

    def __str__(self):
        return f"{self.__name}{[parameter for parameter in self.__parameter_types]} --> {self.__return_type}"

    def __call__(self, *args):
        return self.__function(*args)

    def __len__(self):
        return self.__arity

    def name(self):
        return self.__name

    def return_type(self):
        return self.__return_type

    def parameter_types(self):
        return self.__parameter_types
