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
    A mixin that enables automatic cache invalidation and event bubbling.

    This class transforms standard objects into "reactive" components. It
    automatically detects when public attributes are modified and performs
    three key actions:
    1. **Invalidation**: Clears all local `functools.cached_property` values.
    2. **Propagation**: Triggers registered callbacks to notify observers.
    3. **Linking**: Automatically subscribes to changes in child objects that
       implement the callback interface, creating a reactive tree.
    """
    _initialised = False

    def start_caching(self):
        """
        Enables the reactivity gate.

        Call this method at the end of your subclass `__init__`. Once called,
        any further updates to public attributes will trigger the
        invalidation and bubbling logic.
        """
        self._initialised = True

    @classmethod
    @lru_cache(maxsize=None)
    def _cached_property_names(cls) -> set[str]:
        """
        Introspects the class hierarchy to find all cached properties.

        This is a class-level optimization. By scanning the MRO (Method
        Resolution Order) once, we identify which attribute names correspond
        to `cached_property` descriptors so they can be cleared later.

        Returns:
            set[str]: A set of attribute names decorated with @cached_property.
        """
        names: set[str] = set()
        for base in cls.__mro__:
            for name, attr in base.__dict__.items():
                if isinstance(attr, cached_property):
                    names.add(name)
        return names

    def _clear_cache(self) -> None:
        """
        Nukes the local cache of all identified cached properties.

        Standard `cached_property` stores its result in the instance `__dict__`.
        Deleting the key from `__dict__` forces the property to re-evaluate
        the next time it is accessed.
        """
        for name in self.__class__._cached_property_names():
            self.__dict__.pop(name, None)

    @property
    def _callbacks(self):
        """
        A WeakSet of callables to be executed on attribute changes.

        We use a `WeakSet` to ensure that if a parent/observer is garbage
        collected, this object doesn't keep it alive (preventing memory leaks).
        """
        if '_on_change_callbacks' not in self.__dict__:
            # Use WeakSet to prevent memory leaks/zombie references
            self.__dict__['_on_change_callbacks'] = weakref.WeakSet()
        return self.__dict__['_on_change_callbacks']

    def add_callback(self, callback):
        """Registers a listener to be notified of any internal state changes."""
        self._callbacks.add(callback)

    def remove_callback(self, callback):
        """Unregisters a listener."""
        self._callbacks.discard(callback)

    def _on_attribute_change(self):
        """
        The core reactive trigger.

        Clears local caches first, then bubbles the notification to all
        registered observers (usually parent objects in a hierarchy).
        """
        self._clear_cache()
        for callback in list(self._callbacks):
            callback()

    def __setattr__(self, name, value):
        """
        Overrides attribute assignment to manage the reactive lifecycle.

        Logic flow:
        1. **Unlink**: If the attribute previously held a reactive child,
           unsubscribe from that child's changes.
        2. **Link**: If the new value is reactive (has `add_callback`),
           subscribe to it so its changes bubble through this object.
        3. **Set**: Assign the value using the superclass method.
        4. **Trigger**: If the attribute is public and the object is
           fully initialized, fire the change logic.
        """
        # Update links before the assignment to ensure the callback in the 'old' object is cleared
        self._update_attribute_links(name, value)

        # 2. Perform the assignment
        super().__setattr__(name, value)

        # Trigger invalidation if we are 'live' and the attribute is public
        # Public attributes are defined as those not starting with '_'
        if not name.startswith('_') and self._initialised:
            self._on_attribute_change()

    def _update_attribute_links(self, name: str, new_obj: any) -> None:
        """
        Manages the observer relationships between this object and its children.

        This handles the 'handshake'—unsubscribing from the old value and
        subscribing to the new value if it supports the callback interface.
        """
        # 1. get the _attribute_links dictionary if it exists or create it
        links = self.__dict__.setdefault('_attribute_links', {})

        # 2. Remove my callback from previous object,
        # and remove it from links
        if name in links:
            old_obj, old_callback = links.pop(name)
            if hasattr(old_obj, 'remove_callback'):
                old_obj.remove_callback(old_callback)

        # 2. Add my callback to the new subscribed objec,
        # and store reference to the bound method
        if new_obj is not None and hasattr(new_obj, 'add_callback'):
            callback = self._on_attribute_change
            new_obj.add_callback(callback)
            links[name] = (new_obj, callback)
