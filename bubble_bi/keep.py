"""Not losing your work.

Colab disconnects. It disconnects after ninety minutes of a two-hour run, and it does
not care what you were in the middle of. So every trained model is written to disk the
moment it is finished, and re-running the notebook picks up what is already there
instead of training it again.

On Colab, point `data_dir` at your Google Drive and the models outlive the session:

    SETTINGS = dict(..., data_dir="/content/drive/MyDrive/bubble_bi")
"""

from __future__ import annotations

from pathlib import Path

import torch


def _folder(settings: dict) -> Path:
    place = Path(settings["data_dir"]) / "models"
    place.mkdir(parents=True, exist_ok=True)
    return place


def save(model, name: str, settings: dict, **notes) -> Path:
    """Write a trained model to disk, with the settings it was trained under."""
    path = _folder(settings) / f"{name}.pt"
    torch.save(
        {
            "weights": model.state_dict(),
            "settings": settings,
            "notes": notes,
        },
        path,
    )
    return path


def load(model, name: str, settings: dict, quiet: bool = False):
    """Put a saved model back, or return None if there is nothing saved.

    ⚠️ Refuses to load weights trained under settings that would not fit — a model
    trained with a 512-word vocabulary cannot be poured into a 256-word one, and doing
    it silently would produce a model that is subtly, invisibly wrong.
    """
    path = _folder(settings) / f"{name}.pt"
    if not path.exists():
        return None

    saved = torch.load(path, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(saved["weights"])
    except RuntimeError as why:
        raise RuntimeError(
            f"The saved '{name}' does not fit the model you have built — the settings "
            f"must have changed since it was trained.\n\n{why}\n\n"
            f"Delete {path} to train it again from scratch."
        ) from None

    if not quiet:
        notes = saved.get("notes", {})
        said = ", ".join(f"{k} {v}" for k, v in notes.items())
        print(f"↩︎  loaded a trained '{name}' from {path}"
              + (f"  ({said})" if said else ""))
    return model


def trained(name: str, settings: dict) -> bool:
    return (_folder(settings) / f"{name}.pt").exists()


def forget(name: str, settings: dict) -> None:
    """Throw a saved model away, so the next run trains it fresh."""
    path = _folder(settings) / f"{name}.pt"
    if path.exists():
        path.unlink()
        print(f"🗑  deleted {path} — it will be trained again")
