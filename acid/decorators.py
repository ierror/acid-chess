from functools import cache


class classproperty(property):
    def __get__(self, owner_self, owner_cls):
        return self.fget(owner_cls)


def cached_class_attr(f):
    return classmethod(property(cache(f)))
