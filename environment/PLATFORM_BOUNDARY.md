# Environment boundary

The bundled dependency runtime targets CPython 3.12 on macOS arm64 and permits offline execution without package installation. A Python interpreter is not bundled. Other platforms require a compatible environment using the versions in `requirements-lock.json`; equivalent execution outside the tested target is not claimed.

All experiment timestamps are explicit UTC values. The analysis does not read or depend on the host time zone, locale, user name, host name, or operating-system patch level.
