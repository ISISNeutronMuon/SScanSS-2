import re

VERSION_REGEX = re.compile(r"(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
                           r"(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)"
                           r"(?:\.(?:0|[1-9]\d*|\d *[a-zA-Z-][0-9a-zA-Z-]*))*))?"
                           r"(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?")


class Version:
    """This class represents semantic versioning number

    :param major: major version
    :type major: int
    :param minor: minor version
    :type minor: int
    :param patch: path version
    :type patch: int
    :param pre_release: pre release tag
    :type pre_release: Optional[str]
    :param build: build number
    :type build: Optional[str]
    """
    def __init__(self, major, minor, patch, pre_release=None, build=None):
        self.major = major
        self.minor = minor
        self.patch = patch
        self.pre_release = pre_release
        self.build = build

    @classmethod
    def parse(cls, value):
        """Creates equivalent Version object from the string representation

        :param value: version
        :type value: str
        :return: version
        :rtype: Version
        """
        try:
            version = list(VERSION_REGEX.match(value.replace(' ', '')).groups())
            version[:3] = map(int, version[:3])
        except AttributeError as e:
            raise ValueError(f'Could not parse "{value}" to a semantic version') from e

        return cls(*version)

    def __eq__(self, other):
        if not isinstance(other, Version):
            return False

        return (self.major == other.major and self.minor == other.minor and self.patch == other.patch
                and self.pre_release == other.pre_release and self.build == other.build)

    def __str__(self):
        version = f'{self.major}.{self.minor}.{self.patch}'
        if self.pre_release is not None:
            version = f'{version}-{self.pre_release}'
        if self.build is not None:
            version = f'{version}+{self.build}'

        return version

    def __repr__(self):
        return f'Version({self.major}, {self.minor}, {self.patch}, {self.pre_release}, {self.build})'


__version__ = Version(1, 1, 0)
__editor_version__ = Version(1, 1, 0)
