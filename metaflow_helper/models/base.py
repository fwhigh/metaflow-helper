class BaseModel:
    def __init__(self):
        pass

    def _validate_init_kwargs(self):
        try:
            self.mode
        except NameError as e:
            print('You must make mode an init kwarg')
            raise e
        try:
            self.iterations
        except NameError as e:
            print('You must make iterations an init kwarg')
            raise e
        try:
            self.input_dim
        except NameError as e:
            print('You must make input_dim an init kwarg')
            raise e

    def _validate_fit_kwargs(self):
        pass
