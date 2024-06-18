from deglib_cpp import Mt19937 as Mt19937cpp


class Mt19937:
    def __init__(self, seed: int = 5489):
        self.mt19937_cpp = Mt19937cpp(seed)

    def _to_cpp(self) -> Mt19937cpp:
        return self.mt19937_cpp


