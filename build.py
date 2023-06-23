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


def compile_jar():
    if shutil.which(_BUILD_TOOL) is None:
        raise RuntimeError(
            f"`{_BUILD_TOOL}` required for building `funkea` from source. Please see "
            "https://www.scala-lang.org/download/ for installation details or make sure "
            "that the tool is available in your PATH."
        )

    subprocess.run([_BUILD_TOOL, "clean", "package"], cwd=_SCALA_PROJECT, check=True)
    filename: str = f"{_SCALA_NAME}_{_SCALA_VERSION}-{version.__version__}.jar"
    shutil.copy(
        str(_SCALA_PROJECT / "target" / f"scala-{_SCALA_VERSION}" / filename),
        str(_HOME / "funkea" / "core" / "resources" / filename),
    )


if __name__ == "__main__":
    compile_jar()
