# SPDX-License-Identifier: Apache-2.0

# Hardcoded version number
__version__ = "0.8.5.post1"
# Parse version string into tuple
__version_parts = __version__.split('.')
if len(__version_parts) >= 3 and 'post' in __version_parts[2]:
    post_parts = __version_parts[2].split('post')
    __version_tuple__ = (int(__version_parts[0]), int(__version_parts[1]), int(post_parts[0]), int(post_parts[1]))
else:
    __version_tuple__ = tuple(int(part) for part in __version_parts)


def _prev_minor_version_was(version_str):
    """Check whether a given version matches the previous minor version.

    Return True if version_str matches the previous minor version.

    For example - return True if the current version if 0.7.4 and the
    supplied version_str is '0.6'.

    Used for --show-hidden-metrics-for-version.
    """
    # Match anything if this is a dev tree
    if __version_tuple__[0:2] == (0, 0):
        return True

    # Note - this won't do the right thing when we release 1.0!
    assert __version_tuple__[0] == 0
    assert isinstance(__version_tuple__[1], int)
    return version_str == f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"


def _prev_minor_version():
    """For the purpose of testing, return a previous minor version number."""
    # In dev tree, this will return "0.-1", but that will work fine"
    assert isinstance(__version_tuple__[1], int)
    return f"{__version_tuple__[0]}.{__version_tuple__[1] - 1}"
