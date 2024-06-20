import deglib_cpp


class ResultSet:
    def __init__(self, result_set_cpp: deglib_cpp.ResultSet):
        if not isinstance(result_set_cpp, deglib_cpp.ResultSet):
            raise TypeError('result_set_cpp must be of type deglib_cpp.ResultSet')
        self.result_set_cpp = result_set_cpp

    def top(self) -> :
