"""Why a prompt reused no cached tokens - the reason in the "re-reading the entire chat" log."""

import torch

from app.runtime.cache_store import StoredEntry
from app.runtime.prompt_slot import CacheSlot


class CacheMissDiagnosis:
    @staticmethod
    def ram(
        slots: list[CacheSlot], prompt_ids: list[int], key: tuple[int, torch.dtype], tag: str
    ) -> str:
        """Why no RAM slot supplied a prefix."""
        if not slots:
            return "RAM holds no conversations"
        parts: list[str] = []
        for slot in slots:
            if tag and slot.tag != tag:
                continue
            if slot.key[1] != key[1]:
                parts.append(f"this chat's RAM cache uses {slot.key[1]}, now {key[1]}")
                continue
            common, usable = slot.match(prompt_ids)
            if common == 0:
                parts.append("this chat's RAM cache shares no leading tokens with the prompt")
            elif usable == 0:
                parts.append(
                    f"hybrid model: no recurrent snapshot at or before the {common} shared tokens"
                )
        return (
            "; ".join(dict.fromkeys(parts))
            or f"none of the {len(slots)} RAM slots belongs to or matches this chat"
        )

    @staticmethod
    def disk(
        entries: list[StoredEntry],
        wanted: list[str],
        block_tokens: int,
        model_id: str,
        dtype: torch.dtype,
        tag: str,
        enabled: bool,
    ) -> str:
        """Why no stored file supplied a prefix."""
        if not enabled:
            return "the disk cache is off"
        if not entries:
            return "the disk cache is empty (nothing was saved, or the files expired)"
        mine = [e for e in entries if tag and e.tag == tag]
        if tag and not mine:
            return f"no file for this chat on disk ({len(entries)} from other chats)"
        parts: list[str] = []
        best = -1
        for entry in mine or entries:
            if entry.model_id != model_id:
                parts.append("saved for a different model file")
            elif entry.dtype_name != str(dtype):
                parts.append(f"saved with {entry.dtype_name}, now {dtype}")
            else:
                shared = 0
                for mine_hash, theirs in zip(wanted, entry.blocks, strict=False):
                    if mine_hash != theirs:
                        break
                    shared += 1
                best = max(best, shared)
        if best >= 0:
            parts.append(
                f"closest saved chat shares {best * block_tokens} tokens ({best} whole "
                f"{block_tokens}-token blocks) - its start differs from this prompt"
            )
        return "; ".join(dict.fromkeys(parts))
