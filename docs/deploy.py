import os
import shutil
import sys

sys.path.insert(0, os.path.abspath(".."))
from sscanss.__version import __version__

ref = "master"
version = str(__version__)
if len(sys.argv) > 1 and sys.argv[1].strip().endswith(version):
    ref = version

DOCS_PATH = os.path.abspath(os.path.dirname(__file__))
BUILD_PATH = os.path.join(DOCS_PATH, "_build", "html")
WEB_PATH = os.path.join(DOCS_PATH, "_web", ref)

if os.path.isdir(WEB_PATH):
    shutil.rmtree(WEB_PATH, ignore_errors=True)

shutil.copytree(BUILD_PATH, WEB_PATH, ignore=shutil.ignore_patterns(".buildinfo", "objects.inv", ".doctrees"))

if ref == version:
    INDEX_FILE = os.path.join(DOCS_PATH, "_web", "index.html")
    data = [
        "<!DOCTYPE html>\n",
        "<html>\n",
        "  <head>\n",
        f"    <title>Redirecting to https://isisneutronmuon.github.io/SScanSS-2/{ref}/</title>\n",
        '    <meta charset="utf-8">\n',
        f'    <meta http-equiv="refresh" content="0; URL=https://isisneutronmuon.github.io/SScanSS-2/{ref}/index.html">\n',
        f'    <link rel="canonical" href="https://isisneutronmuon.github.io/SScanSS-2/{ref}/index.html">\n',
        "  </head>\n",
        "</html>",
    ]

    with open(INDEX_FILE, "w") as index_file:
        index_file.writelines(data)
