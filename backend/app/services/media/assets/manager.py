"""Asset manager: directories, resolution and honest status reporting.

The manager owns the media directory layout (``images/``, ``video/``,
``audio/``, ``thumbnails/``) and answers the one question the renderer
cares about: *are the assets a scene asked for actually available?*
"""

from __future__ import annotations

from pathlib import Path

from app.agents.video_producer.schemas import AssetStatus, MediaAsset, Scene
from app.services.media.assets.base import AssetProvider
from app.services.media.assets.local import LocalAssetProvider

#: Subdirectories created under the media root.
MEDIA_SUBDIRS = ("images", "video", "audio", "thumbnails")


class AssetManager:
    """Coordinates asset resolution for a production run."""

    def __init__(self, provider: AssetProvider | None = None, *, base_dir: Path | None = None) -> None:
        self._base = Path(base_dir) if base_dir else Path.cwd() / "media"
        self._provider = provider or LocalAssetProvider(self._base)

    @property
    def base_dir(self) -> Path:
        return self._provider.root()

    def ensure_directories(self) -> None:
        """Create the standard media layout if missing."""
        for name in MEDIA_SUBDIRS:
            (self.base_dir / name).mkdir(parents=True, exist_ok=True)

    def resolve(self, reference: str) -> MediaAsset:
        """Resolve one reference through the underlying provider."""
        return self._provider.resolve(reference)

    def resolve_scene_assets(self, scenes: list[Scene]) -> tuple[list[Scene], str]:
        """Resolve every scene's asset references.

        A requirement carrying a concrete ``path`` goes to the
        provider; a requirement that only carries a ``prompt`` is a
        production instruction with no file behind it yet, so it stays
        ``PLANNED``. Either way the scene is *not* available until a
        real file resolves, which is the honest answer Phase 7 must
        give: nothing has supplied image assets yet.

        Returns the updated scenes plus an overall asset status
        (``"none"`` when no scene asked for assets, otherwise
        ``"available"`` or ``"missing"``).
        """
        updated: list[Scene] = []
        any_required = False
        all_available = True

        for scene in scenes:
            if scene.asset_requirements:
                any_required = True
            resolved: list[MediaAsset] = []
            for requirement in scene.asset_requirements:
                media_type = str(requirement.get("media_type", ""))
                reference = requirement.get("path")
                if not reference:
                    # Planned, not present: no path was ever supplied.
                    resolved.append(
                        MediaAsset(
                            asset_id=f"{scene.scene_id}_{media_type or 'asset'}",
                            media_type=media_type,
                            status=AssetStatus.PLANNED,
                        )
                    )
                    all_available = False
                    continue
                try:
                    asset = self.resolve(str(reference))
                except Exception:  # noqa: BLE001 - a bad ref must not kill the run
                    asset = MediaAsset(
                        asset_id=f"{scene.scene_id}_{media_type or 'asset'}",
                        media_type=media_type,
                        status=AssetStatus.MISSING,
                    )
                resolved.append(asset)
                if asset.status is not AssetStatus.AVAILABLE:
                    all_available = False
            updated.append(scene.model_copy(update={"asset_references": tuple(resolved)}))

        if not any_required:
            overall = "none"
        else:
            overall = "available" if all_available else "missing"
        return updated, overall
