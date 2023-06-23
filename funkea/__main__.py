"""Simple CLI for managing the file-registry."""
import distutils.util
import os
import sys

import fire
import rich
from rich import markdown

from funkea.core.utils import files


def _interactive() -> dict[str, str | None]:
    paths: dict[str, str | None] = {}
    for name, field in files.FileRegistry.schema()["properties"].items():
        rich.print(
            markdown.Markdown(
                "# " + name + "\n## " + field["title"] + "\n\n" + field["description"]
            )
        )
        path: str = input("Path: ")
        paths[name] = path or None
    return paths


def _from_stem(stem: str) -> dict[str, str]:
    paths: dict[str, str] = {}
    for name in files.FileRegistry.schema()["properties"].keys():
        # we make sure here that all keys in the file registry have the name of the files
        paths[name] = os.path.join(stem, name)
    return paths


def _store(fr: files.FileRegistry) -> None:
    out = fr.json(indent=4)

    rich.print()
    rich.print(f"[bold red]Generated registry[/bold red] (at {files.FILE_REGISTRY}):")
    rich.print_json(out)
    with open(files.FILE_REGISTRY, "w") as file:
        file.write(out)


class CLI:
    """funkea command-line interface.

    This CLI is for convenient setting the file-registry values. Use this tool to set the default
    filepaths for various, frequently used datasets.
    """

    @staticmethod
    def init(*, from_stem: str | None = None) -> None:
        """Initialises the file-registry.

        This command can be used in two modes: (1) interactive, where the user will be asked for
        each filepath in turn (each of which can be left blank); and (2) in one go, if the stem of
        the filepaths is provided (this assumes that all files are in the same parent directory
        *and* have the appropriate names).
        If the user is using the data provided by BAI, it is recommended to use (2).

        Args:
            from_stem: Optional filepath stem. If this is provided, the command runs in mode (2).
                This assumes that all files have the correct names and are within the directory
                at the end of the stem. If left unspecified, the interactive session is launched.
                (``default=None``)
        """
        res: dict[str, str | None] | dict[str, str]
        if from_stem is not None:
            res = _from_stem(from_stem)
        else:
            res = _interactive()
        fr = files.FileRegistry.parse_obj(res)
        _store(fr)

    @staticmethod
    def set(name: str, path: str | None = None) -> None:
        """Sets a specific filepath in the registry.

        Args:
            name: Name of the registry path.
            path: Path to the file.
        """
        fr = files.FileRegistry.default().dict()
        if path is None:
            path = input("Path: ")
        fr[name] = path
        _store(files.FileRegistry.parse_obj(fr))

    @staticmethod
    def reset(yes: bool = False):
        """Resets the file-registry.

        This restores the original file-registry file; that is, each field in the JSON file will be
        ``null``. Since this is a deleterious action, the command will ask for confirmation.

        Args:
            yes: Whether to proceed. If False, a prompt will be raised asking whether to proceed.
                (``default=False``)
        """
        rich.print("[bold red]WARNING[/bold red]: resetting file registry.")
        sure = yes or distutils.util.strtobool(input("Proceed [y/n]: "))
        if sure:
            # mypy somehow does not realise the arguments to FileRegistry are optional
            _store(files.FileRegistry())  # type: ignore[call-arg]


def main():
    try:
        fire.Fire(CLI, name="funkea-cli")
    except KeyboardInterrupt:
        sys.exit(1)


if __name__ == "__main__":
    main()
