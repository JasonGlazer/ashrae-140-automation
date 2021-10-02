from logger import Logger


class InputProcessor(Logger):
    """
    Verify Input file and create a cleansed epJSON file for data visualizations
    """

    def __init__(self, logger_level="WARNING", logger_name="console_only_logger"):
        super().__init__(logger_level=logger_level, logger_name=logger_name)
        return


if __name__ == "__main__":
    ip = InputProcessor(logger_level='DEBUG')
    ip.logger.info('Test')