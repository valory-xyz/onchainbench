"""Small helpers fork-connect and its interim intent gate share: the ABI cache, the EIP-1967 slot, number parsing."""

from __future__ import annotations

import json
import threading
from pathlib import Path

ABI_DIR = Path(__file__).resolve().parent / "abis"
EIP1967_IMPL = "0x360894a13ba1a3210667c828492db98dca3e2076cc3735a920a3ca505d382bbc"


JSON_SAFE_INT = 2**53  # past this a JSON number is not exact in a JavaScript client: 9e18 can arrive as 9e15


def _int(v) -> int:
    """An amount (wei, base units) from an integer, a decimal string or a 0x hex string, ASCII digits only (the
    broker's rule). A float is taken only when it is a whole number a JSON client holds exactly; larger amounts
    must come as a decimal string, so a rounded number is refused rather than sent."""
    from onchainbench.connect.policy import _int as amount

    if v is None or v == "":
        return 0
    if isinstance(v, float):
        if not v.is_integer() or abs(v) > JSON_SAFE_INT:
            raise ValueError(f"amount {v!r} is not an exact integer: send large amounts as a decimal string")
        v = int(v)
    return amount(v)


# ---- Anvil's fork RPC cache ---------------------------------------------------------------------
# Foundry >= 1.9 names a fork's cache ~/.foundry/cache/rpc/<chain>/<block>/storage-<keccak(fork URL)>.json and keeps
# the fork URL in the cache's identity (its source id: a cache copied to another URL's file name is thrown away), so a
# relay on a random port made every fork start cold. Anvil gets a fixed relay URL per chain instead: one relay per
# chain in each process (evm.shared_relay), on relay_port; a process that finds the port taken uses a free one and
# starts cold. ForkCache copies the file from the <chain>/<block>/storage.json layout older Foundry used, which CI
# saves, and merges it back there.
#
# Every fork of a chain starts from the same saved file and Anvil writes its whole cache when it stops, so the saved
# file is the union of what the forks fetched (union_cache), not the last one's. The forks are pinned to a block and
# the cache holds state fetched from upstream at that block, which never changes: a union loses nothing and adds
# nothing a single fork couldn't have fetched. Local state (a trial's transactions, anvil_set*) stays in Anvil's own
# layer and is never written, with one exception: Foundry before 1.9 also writes the hashes of blocks it mined
# itself, so the union keeps no block hash above the fork block.
# relay port = base + chain id % 1000 (ONCHAINBENCH_RELAY_PORT_BASE overrides). 29000-29999 is below every OS's
# ephemeral range (Linux 32768-60999, macOS and Windows 49152-65535), so no outgoing connection is ever given a
# chain's port (and the fork then started cold); the bench's other servers bind port 0 (an ephemeral port)
RELAY_PORT_BASE = 29000
CACHE_CHAIN_NAMES = {  # alloy-chains' names, which Foundry uses for the cache directory
    1: "mainnet",
    10: "optimism",
    100: "xdai",
    137: "polygon",
    8453: "base",
    34443: "mode",
    42161: "arbitrum",
    42220: "celo",
}
STABLE_CACHE = "storage.json"
_DIR_LOCKS: dict = {}  # cache directory -> this process's lock for it
_DIR_LOCKS_GUARD = threading.Lock()


def relay_port(chain_id: int) -> int:
    import os

    return int(os.environ.get("ONCHAINBENCH_RELAY_PORT_BASE") or RELAY_PORT_BASE) + chain_id % 1000


def rpc_cache_root() -> Path:
    return Path.home() / ".foundry" / "cache" / "rpc"


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"  # Foundry >= 1.9 writes the cache zstd-compressed; older Foundry plain JSON


def read_cache(f: Path) -> tuple[dict, bool]:
    """A cache file's JSON, and whether it was compressed. ValueError if it can't be read (Anvil killed while
    writing it, or not a cache at all)."""
    raw = f.read_bytes()
    compressed = raw.startswith(ZSTD_MAGIC)
    if compressed:
        import zstandard

        try:
            raw = zstandard.ZstdDecompressor().stream_reader(raw).read()
        except zstandard.ZstdError as e:
            raise ValueError(f"unreadable zstd cache: {e}") from e
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("not a fork cache: the top level is not an object")
    return data, compressed


def write_cache(f: Path, data: dict, compressed: bool) -> None:
    """Whole or not at all (a temporary file, then a rename): a reader never sees half a file. Compressed as the
    Foundry that wrote the newest file reads it."""
    raw = json.dumps(data).encode()
    if compressed:
        import zstandard

        raw = zstandard.ZstdCompressor().compress(raw)
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_name(f.name + ".tmp")
    try:
        tmp.write_bytes(raw)
        tmp.replace(f)
    finally:
        tmp.unlink(missing_ok=True)


def _obj(v) -> dict:
    return v if isinstance(v, dict) else {}


def _block_number(key) -> int | None:
    try:
        return int(str(key), 0)
    except ValueError:
        return None


def union_cache(saved: dict, new: dict, block: int) -> dict:
    """The union of two caches of one fork (chain, block): every account, storage slot and block hash either holds.
    Fetched at a pinned block, an entry can't legitimately differ between the two; if it does, the saved one stays,
    so the file every fork starts from doesn't change under them. The rest (meta: the block, the fork's source)
    comes from the new file, which the running Foundry wrote. Block hashes above the fork block are blocks a fork
    mined itself (Foundry before 1.9 caches them): never kept."""
    out = {**saved, **new}
    out["accounts"] = {**_obj(new.get("accounts")), **_obj(saved.get("accounts"))}
    storage = {a: dict(_obj(s)) for a, s in _obj(new.get("storage")).items()}
    for a, slots in _obj(saved.get("storage")).items():
        storage[a] = {**storage.get(a, {}), **_obj(slots)}
    out["storage"] = storage
    hashes = {**_obj(new.get("block_hashes")), **_obj(saved.get("block_hashes"))}
    out["block_hashes"] = {k: h for k, h in hashes.items() if (n := _block_number(k)) is not None and n <= block}
    return out


class ForkCache:
    """The cache handoff for one fork (chain, block) behind a fixed relay URL: `warm()` before Anvil starts,
    `keep(stop)` to stop it. Both hold a lock on the block's directory, so concurrent forks of one chain never
    write the file at once; each Anvil writes its whole cache when it exits, and keep() adds it to the saved one."""

    def __init__(self, chain_id: int, block: int, fork_url: str, root: Path | None = None):
        from eth_utils import keccak

        self.block = block
        self.dir = (root or rpc_cache_root()) / CACHE_CHAIN_NAMES.get(chain_id, str(chain_id)) / str(block)
        self.hashed = self.dir / f"storage-{keccak(fork_url.encode()).hex()}.json"  # Foundry >= 1.9's name
        self.stable = self.dir / STABLE_CACHE  # older Foundry's name, and the one CI saves

    def _locked(self):
        """This process's lock for the directory, and a file lock for other processes where there is flock (not on
        Windows: there only forks of one run are kept apart)."""
        import contextlib

        self.dir.mkdir(parents=True, exist_ok=True)
        with _DIR_LOCKS_GUARD:
            local = _DIR_LOCKS.setdefault(str(self.dir), threading.Lock())
        try:
            import fcntl
        except ImportError:  # pragma: no cover - Windows
            fcntl = None

        @contextlib.contextmanager
        def lock():
            with local, open(self.dir / ".onchainbench.lock", "w") as f:
                if fcntl is not None:
                    fcntl.flock(f, fcntl.LOCK_EX)
                yield  # the file lock goes with the file

        return lock()

    @staticmethod
    def _copy(src: Path, dst: Path) -> None:
        """Whole or not at all: a reader never sees half a file."""
        import shutil

        tmp = dst.with_name(dst.name + ".tmp")
        shutil.copy2(src, tmp)
        tmp.replace(dst)

    def warm(self) -> bool:
        """Put the saved cache where this Anvil looks, if it is newer than what is there. True if a cache is there."""
        with self._locked():
            if self.stable.is_file() and (
                not self.hashed.is_file() or self.stable.stat().st_mtime > self.hashed.stat().st_mtime
            ):
                self._copy(self.stable, self.hashed)
            return self.hashed.is_file() or self.stable.is_file()

    def keep(self, stop) -> None:
        """Stop Anvil under the lock, then add the cache it wrote as it exited to the one saved under the stable
        name. Foundry >= 1.9 writes the hashed file; older Foundry writes the stable one itself, over what the other
        forks saved, so that is read before the stop. A file Anvil didn't write is skipped; so is one it was killed
        while writing (the saved state stays). Never raises into the trial."""
        with self._locked():
            before = {f: _stamp(f) for f in (self.hashed, self.stable)}
            saved = _load(self.stable)
            stop()
            written = [f for f in (self.hashed, self.stable) if _stamp(f) not in (None, before[f])]
            if written:
                merge_caches(self.stable, [saved, *(_load(f) for f in written)], self.block)


def _stamp(f: Path) -> tuple | None:
    """What changes when a file is written: its time, and its size and inode too (a coarse clock, as on Windows,
    can give a rewrite the same time)."""
    try:
        st = f.stat()
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size, st.st_ino


def _load(f: Path) -> tuple[dict, bool] | None:
    """read_cache, or None if the file is missing or can't be read."""
    try:
        return read_cache(f)
    except (OSError, ValueError):
        return None


def merge_caches(dst: Path, caches: list, block: int) -> bool:
    """Write the union of `caches` (read_cache results, oldest first; None for one that couldn't be read, skipped)
    to dst, for a fork at `block`, in the newest one's format. An earlier entry wins a conflict (union_cache).
    Never raises; returns whether dst was written (not if no cache could be read, or the write failed: dst is
    only ever replaced whole)."""
    readable = [c for c in caches if c is not None]
    if not readable:
        return False
    data: dict = {}
    for c, _ in readable:
        data = union_cache(data, c, block)
    try:
        write_cache(dst, data, readable[-1][1])
    except OSError:  # a full disk, say
        return False
    return True


def merge_into(dst: Path, src: Path, block: int) -> bool:
    """Add the fork cache file src to dst (created if missing), for a fork at `block`; for merging the caches
    CI's shards saved. A src that can't be read is skipped (dst stays as it was); a dst that can't be read is
    replaced by src's state. Never raises; returns whether dst was written."""
    new = _load(src)
    return new is not None and merge_caches(dst, [_load(dst), new], block)


def load_abi(chain: str, address: str) -> list | None:
    p = ABI_DIR / chain / f"{address.lower()}.json"
    return json.loads(p.read_text()) if p.is_file() else None


def _abi_type(i: dict) -> str:
    t = i["type"]
    if t.startswith("tuple"):
        return "(" + ",".join(_abi_type(c) for c in i.get("components", [])) + ")" + t[5:]
    return t
