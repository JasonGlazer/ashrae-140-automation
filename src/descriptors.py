import pathlib
from src.custom_exceptions import ASHRAE140FileNotFoundError

root_directory = pathlib.Path(__file__).parent.parent.resolve()


class VerifyInputFile:
    """
    Verify that a valid file is passed to the class
    """
    def __get__(self, obj, owner):
        file_location = obj._file_location
        return file_location

    def __set__(self, obj, value):
        print('value', value)
        print('root', root_directory)
        if root_directory.joinpath(value).is_file():
            obj._file_location = root_directory.joinpath(value)
        else:
            obj._file_location = None
            raise ASHRAE140FileNotFoundError('Input file not found {}'.format(value))
        return
