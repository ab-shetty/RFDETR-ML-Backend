import os
import logging
import logging.config

log_level = os.getenv("LOG_LEVEL", "INFO")

logging.config.dictConfig(
    {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "[%(asctime)s] [%(levelname)s] [%(name)s::%(funcName)s::%(lineno)d] %(message)s"
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": log_level,
                "stream": "ext://sys.stdout",
                "formatter": "standard",
            }
        },
        "root": {
            "level": log_level,
            "handlers": ["console"],
            "propagate": True,
        },
    }
)

from label_studio_ml.api import init_app
from model import RFDETR

app = init_app(model_class=RFDETR)
