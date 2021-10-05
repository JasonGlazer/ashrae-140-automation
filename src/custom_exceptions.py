from logger import Logger


class CustomException(Logger, Exception):
    """
    Custom Exceptions used to write outputs to logger and
    indicate program-specific issues
    """
    def __init__(self, msg=''):
        super().__init__()
        self.msg = msg
        self.logger.error(msg)
        return

    def __str__(self):
        return self.msg


class ASHRAE140ProcessingError(CustomException):
    """
    General file processing errors
    """
    pass


class ASHRAE140FileNotFoundError(CustomException, FileNotFoundError):
    pass


class ASHRAE140TypeError(CustomException, TypeError):
    pass
