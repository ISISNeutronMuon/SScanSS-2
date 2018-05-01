import json
import logging
import logging.config
import os
import sys
from sscanss.ui import ui


def setup_logging(file_path='logging.json', default_level=logging.ERROR):
    """
    Configure of logging file handler from json file.

    :param file_path: path to log configuration file, defaults to 'logging.json'
    :type file_path: str
    :param default_level: log verbosity level, defaults to logging.ERROR
    :type default_level: int
    """
    try:
        log_path = os.path.join(os.environ['APPDATA'], 'SScanSS 2', 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        with open(file_path, 'rt') as config_file:
            config = json.load(config_file)

            filename = os.path.join(log_path, config['handlers']['info_file_handler']['filename'])
            config['handlers']['info_file_handler']['filename'] = filename

            filename = os.path.join(log_path, config['handlers']['error_file_handler']['filename'])
            config['handlers']['error_file_handler']['filename'] = filename

            logging.config.dictConfig(config)
    except (IOError, ValueError, AttributeError, ImportError):
        logging.basicConfig(level=default_level)
        logging.exception("Could not initialize logging with %s", file_path)


def main():
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        logger.info('Started the application...')
        ui.execute()
    except IOError:
        logger.exception('Could not read/write file !')
    except:
        logger.exception('An unknown error occurred!')

    sys.exit(1)


if __name__ == '__main__':
    main()
