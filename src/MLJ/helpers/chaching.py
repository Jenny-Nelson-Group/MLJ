# src/MLJ/helpers/caching.py
#####################################################################################
# MLJ Package
#
# Helper module that handles cache invalidation.
# Author: Jolanda S Müller, Tim Rein, Imperial College London
# Copyright (c) 2025, Imperial College London, BSD 3-Clause License
# Date: November 2025
#####################################################################################

import weakref
from functools import cached_property
from functools import lru_cache

class read_only_cached_property(cached_property):
    """
    A read-only version of @cached_property.

    Computes the attribute value once and stores it in the instance dictionary.
    Unlike the standard cached_property, this implementation prevents manual
    overwriting by raising an AttributeError on assignment.
    """
    def __set__(self, instance, value):
        raise AttributeError(
            "This property is read-only. To recalculate, update the 'transition' attribute."
        )

class ReactiveModule:
    """
    Mixin that:
      - clears cached_property values when public attributes change
      - bubbles changes up via callbacks
      - auto-links to child objects that support add_callback/remove_callback
    """
    def __init__(self, *args, **kwargs):
        # We call super to maintain MRO, though often this is the end of the chain
        super().__init__(*args, **kwargs)
        self._initialised = False

    def start_caching(self):
        """Once initialised is set to true, changes lead to cache clearing."""
        self._initialised = True

    @classmethod
    @lru_cache(maxsize=None)
    def _cached_property_names(cls) -> set[str]:
        """Find the names of the cached_properties in the class."""
        names: set[str] = set()
        for base in cls.__mro__:
            for name, attr in base.__dict__.items():
                if isinstance(attr, cached_property):
                    names.add(name)
        return names

    def _clear_cache(self) -> None:
        """Clear all cached properties, so that they will be recomputed."""
        for name in self.__class__._cached_property_names():
            self.__dict__.pop(name, None)

    @property
    def _callbacks(self):
        """Lazy-init the WeakSet so it always exists when accessed."""
        if '_on_change_callbacks' not in self.__dict__:
            # Use WeakSet to prevent memory leaks/zombie references
            self.__dict__['_on_change_callbacks'] = weakref.WeakSet()
        return self.__dict__['_on_change_callbacks']

    def add_callback(self, callback):
        self._callbacks.add(callback)

    def remove_callback(self, callback):
        self._callbacks.discard(callback)

    def _on_attribute_change(self):
        """Clears local cache and notifies all parents in the hierarchy."""
        self._clear_cache()
        for callback in list(self._callbacks):
            callback()

    def __setattr__(self, name, value):
        # 1. Lifecycle Management: Disconnect old objects, link new ones
        links = self.__dict__.setdefault('_attribute_links', {})

        if name in links:
            old_obj, old_link = links.pop(name)
            if hasattr(old_obj, 'remove_callback'):
                old_obj.remove_callback(old_link)

        if value is not None and hasattr(value, 'add_callback'):
            handler = self._on_attribute_change
            value.add_callback(handler)
            links[name] = (value, handler)

        # 2. Perform the assignment
        super().__setattr__(name, value)

        # 3. Trigger: If a public attribute is set after __init__, nuke and bubble
        if not name.startswith('_') and getattr(self, '_initialised', False):
            self._on_attribute_change()
