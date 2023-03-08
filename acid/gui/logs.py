from datetime import datetime


class Logger:
    _entries = []
    _has_fresh_data = False

    def __init__(self, max_entries=255):
        self.max_entries = max_entries

    @property
    def has_fresh_data(self):
        return self._has_fresh_data

    def get_entries(self):
        return self._entries

    def get_entries_reversed(self):
        yield from reversed(self._entries)

    def mark_consumed(self):
        self._has_fresh_data = False

    def append(self, msg, stdout_only=False, timestamp=None):
        if timestamp is None:
            timestamp = datetime.now()
        msg = f"{timestamp.strftime('%H:%M:%S')} | {msg}"
        if not stdout_only:
            if len(self._entries) == self.max_entries:
                self._entries.pop(0)
            self._entries.append(msg)
            self._has_fresh_data = True
        print(msg)
        return msg
