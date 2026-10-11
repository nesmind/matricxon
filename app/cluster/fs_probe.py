import re
from pathlib import Path

_SHARED_TYPES = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "ceph", "glusterfs", "lustre", "9p", "afs"}
_OCTAL = re.compile(r"\\([0-7]{3})")


class FilesystemProbe:
    """Filesystem type of a path (from the mount table) and whether it is a network/shared one."""

    def __init__(self, mountinfo: Path = Path("/proc/self/mountinfo")) -> None:
        self._mountinfo = mountinfo

    def fstype(self, path: Path) -> str | None:
        target = str(path.resolve())
        best: tuple[int, str] | None = None
        try:
            lines = self._mountinfo.read_text().splitlines()
        except OSError:
            return None
        for line in lines:
            left, _, right = line.partition(" - ")
            fields = left.split()
            if len(fields) < 5 or not right:
                continue
            mount = _OCTAL.sub(lambda m: chr(int(m.group(1), 8)), fields[4])
            inside = target == mount or target.startswith(mount.rstrip("/") + "/")
            if inside and (best is None or len(mount) >= best[0]):
                best = (len(mount), right.split()[0])
        return best[1] if best else None

    def is_shared(self, path: Path) -> bool:
        kind = self.fstype(path) or ""
        return kind in _SHARED_TYPES or (kind.startswith("fuse.") and kind != "fuse.portal")
