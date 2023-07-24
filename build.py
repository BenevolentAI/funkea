import os
import shutil
import subprocess
from pathlib import Path
from typing import Final

from funkea.core.utils import version

_BUILD_TOOL: Final[str] = "sbt"
_SCALA_NAME: Final[str] = "spark-udf"
_HOME: Final[Path] = Path(".").resolve()
_SCALA_PROJECT: Final[Path] = _HOME / "scala" / _SCALA_NAME
_SCALA_VERSION: Final[str] = "2.12"
_SCALA_FILENAME: Final[str] = f"{_SCALA_NAME}_{_SCALA_VERSION}-{version.__version__}.jar"
_RESOURCE_DIR: Final[Path] = _HOME / "funkea" / "core" / "resources"
_BUILD_DEST: Final[Path] = _RESOURCE_DIR / _SCALA_FILENAME
_FORCE_BUILD: Final[bool] = os.environ.get("FORCE_BUILD", "false") == "true"


def compile_jar():
    if shutil.which(_BUILD_TOOL) is None:
        raise RuntimeError(
            f"`{_BUILD_TOOL}` required for building `funkea` from source. Please see "
            "https://www.scala-lang.org/download/ for installation details or make sure "
            "that the tool is available in your PATH."
        )

    subprocess.run([_BUILD_TOOL, "clean", "package"], cwd=_SCALA_PROJECT, check=True, env={**os.environ, "VERSION": version.__version__})
    # we remove the old jars to avoid packaging deprecated versions
    for file in _RESOURCE_DIR.glob(f"{_SCALA_NAME}_*.jar"):
        file.unlink()
    shutil.copy(
        str(_SCALA_PROJECT / "target" / f"scala-{_SCALA_VERSION}" / _SCALA_FILENAME),
        str(_BUILD_DEST),
    )


def main():
    if not _FORCE_BUILD and _BUILD_DEST.exists():
        return
    compile_jar()


if __name__ == "__main__":
    main()
