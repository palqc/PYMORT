# src/pymort/cli.py
import typer
from importlib.metadata import version as _pkg_version, PackageNotFoundError

app = typer.Typer(help="PYMORT – Longevity bond & mortality toolkit")

@app.command("version")
def version_cmd() -> None:
    """Show installed package version."""
    try:
        print(_pkg_version("pymort"))
    except PackageNotFoundError:
        print("0.0.dev")

@app.command("echo")
def echo_cmd(msg: str) -> None:
    """Echo a message."""
    print(msg)

if __name__ == "__main__":
    app()