#! /usr/bin/env sh

# Use: ./bump_version.sh a.b.c  bumps the version to a.b.c

poetry version $1
sed -i "s/__version__ = \".*\"/__version__ = \""$1"\"/g" phileas/__init__.py
