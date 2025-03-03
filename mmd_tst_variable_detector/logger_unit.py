from logging import getLogger, StreamHandler, DEBUG, INFO, Formatter
# logger = getLogger(__package__)
formatter = Formatter('[%(asctime)s] p%(process)s {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s','%m-%d %H:%M:%S')

handler = StreamHandler()
handler.setLevel(INFO)
handler.setFormatter(formatter)
# logger.setLevel(INFO)
# logger.addHandler(handler)
# logger.propagate = False