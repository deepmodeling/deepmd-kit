class GraphTooLargeError(Exception):
    pass

class GraphWithoutTensorError(Exception):
    pass

class OutOfMemoryError(Exception):
    """This error is caused by out-of-memory (OOM)."""