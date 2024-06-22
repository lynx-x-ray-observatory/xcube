import logging

xcubeLogger = logging.getLogger("xcube")

ufstring = "%(name)-3s : [%(levelname)-9s] %(asctime)s %(message)s"
cfstring = "%(name)-3s : [%(levelname)-18s] %(asctime)s %(message)s"

xcube_sh = logging.StreamHandler()
# create formatter and add it to the handlers
formatter = logging.Formatter(ufstring)
xcube_sh.setFormatter(formatter)
# add the handler to the logger
xcubeLogger.addHandler(xcube_sh)
xcubeLogger.setLevel("INFO")
xcubeLogger.propagate = False

mylog = xcubeLogger
