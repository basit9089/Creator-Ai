"""Asset provider contract.

Kept abstract so a future phase can add remote or generated assets
without touching the Video Producer: it only ever sees this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.agents.video_producer.schemas import MediaAsset


class AssetProvider(ABC):
    """Resolves asset references into real (or honestly missing) files."""

    name: str = "asset-provider"

    @abstractmethod
    def resolve(self, reference: str) -> MediaAsset:
        """Resolve ``reference`` to a :class:`MediaAsset`.

        A reference that does not exist must come back with
        ``status=MISSING`` — never as a fabricated asset.
        """

    @abstractmethod
    def root(self) -> Path:
        """The directory this provider is allowed to read from."""
