class Sentinel:
    def __repr__(self) -> str:
        return f"{type(self).__name__}()"

    def __eq__(self, value: object) -> bool:
        return isinstance(value, type(self)) or value is type(self)
