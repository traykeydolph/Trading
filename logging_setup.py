import logging, os, sys, json, time

class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        base = {
            "ts": time.strftime('%Y-%m-%dT%H:%M:%S', time.gmtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            base["exception"] = self.formatException(record.exc_info)
        return json.dumps(base)

def configure_logging():
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    mode = os.getenv("LOG_FORMAT", "plain")
    root = logging.getLogger()
    root.setLevel(level)
    # Clear default handlers so repeated calls don't duplicate
    root.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    if mode == "json":
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s %(name)s: %(message)s'))
    root.addHandler(handler)
    logging.getLogger(__name__).debug("Logging configured", extra={"format": mode})
