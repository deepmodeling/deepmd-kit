# utils/dotdict.py

class DotDict(dict):
    """A dict subclass that allows attribute-style access."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{name}'")

    def __setattr__(self, name: str, value):
        self[name] = value

    def __delattr__(self, name: str):
        try:
            del self[name]
        except KeyError:
            raise AttributeError(f"'DotDict' has no attribute '{name}'")
