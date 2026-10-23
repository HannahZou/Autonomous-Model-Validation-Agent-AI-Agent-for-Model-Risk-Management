"""A minimal stand-in for the streamlit API, enough to execute app/streamlit_app.py.

It lets the test suite drive the app's code paths (render, accept, sign off,
generate report) without a browser or the streamlit server. `press` names the
buttons that return True on this run, either by label or as (form_key, label).
"""

from __future__ import annotations

import contextlib
import types


class Stop(Exception):
    pass


class Rerun(Exception):
    pass


class _Container:
    def __init__(self, app: "FakeStreamlit"):
        self._app = app

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __getattr__(self, name):
        return getattr(self._app, name)


class FakeStreamlit:
    def __init__(self, inputs: dict | None = None, press: set | None = None):
        self.inputs = inputs or {}
        self.press = press or set()
        self.session_state: dict = {}
        self.calls: list[tuple[str, tuple]] = []
        self.errors: list[str] = []
        self._form: str | None = None
        self.sidebar = _Container(self)

    # display -------------------------------------------------------------
    def _log(self, name, *args):
        self.calls.append((name, args))

    def __getattr__(self, name):
        # title, header, subheader, caption, markdown, write, info, success, warning,
        # metric, progress, json, dataframe, divider, set_page_config, ...
        return lambda *a, **k: self._log(name, *a)

    def error(self, msg, *a, **k):
        self.errors.append(str(msg))
        self._log("error", msg)

    # layout ----------------------------------------------------------------
    def columns(self, spec, **k):
        n = spec if isinstance(spec, int) else len(spec)
        return [_Container(self) for _ in range(n)]

    def tabs(self, labels):
        return [_Container(self) for _ in labels]

    def expander(self, label, **k):
        self._log("expander", label)
        return _Container(self)

    def popover(self, label, **k):
        return _Container(self)

    @contextlib.contextmanager
    def form(self, key, **k):
        self._form = key
        try:
            yield _Container(self)
        finally:
            self._form = None

    @contextlib.contextmanager
    def spinner(self, *a, **k):
        yield

    # inputs ----------------------------------------------------------------
    def _value(self, label, key, default):
        for k in (key, label):
            if k is not None and k in self.inputs:
                return self.inputs[k]
        return default

    def selectbox(self, label, options, index=0, key=None, **k):
        return self._value(label, key, options[index] if options else None)

    def multiselect(self, label, options, default=None, key=None, **k):
        return self._value(label, key, list(default or []))

    def text_input(self, label, value="", key=None, **k):
        v = self._value(label, key, value)
        if key:
            self.session_state[key] = v
        return v

    def text_area(self, label, value="", key=None, **k):
        return self._value(label, key, value)

    def toggle(self, label, value=False, **k):
        return self._value(label, k.get("key"), value)

    def checkbox(self, label, value=False, **k):
        return self._value(label, k.get("key"), value)

    def button(self, label, key=None, **k):
        if k.get("disabled"):
            return False
        return label in self.press or (key is not None and key in self.press)

    def form_submit_button(self, label, **k):
        return (self._form, label) in self.press

    def download_button(self, label, data, **k):
        self._log("download_button", label, k.get("file_name"))

    # control flow --------------------------------------------------------------
    def stop(self):
        raise Stop

    def rerun(self):
        raise Rerun


def as_module(fake: FakeStreamlit) -> types.ModuleType:
    mod = types.ModuleType("streamlit")
    for name in dir(fake):
        if not name.startswith("__"):
            setattr(mod, name, getattr(fake, name))
    mod.__getattr__ = fake.__getattr__  # anything not defined explicitly is a no-op display call
    return mod
