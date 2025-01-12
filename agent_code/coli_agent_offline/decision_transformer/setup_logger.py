import logging

dt_logger = logging.getLogger("dt_logger")
dt_logger.setLevel(logging.INFO)
handler = logging.FileHandler("logs/coli_agent_offline_train.log", mode="w")
handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s %(levelname)-8s %(name)s:%(lineno)-3s - %(funcName)s()   %(message)s"
)
handler.setFormatter(formatter)
dt_logger.addHandler(handler)
