from datetime import datetime


def local_timezone():
    return datetime.utcnow().astimezone().tzinfo


def datetime_local():
    return datetime.now(local_timezone())
