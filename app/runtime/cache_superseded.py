from collections.abc import Iterable
from typing import Protocol


class _StoredEntry(Protocol):
    model_id: str
    length: int
    blocks: list[str]
    tag: str


class SupersededCopies:
    """Older saved copies of a chat that a newly saved one fully covers.

    Each time a chat leaves RAM a new file is written, and an older one stayed until it expired
    (one chat showed as several "saved chats"). A copy is superseded when it is the same chat and
    model and its tokens are a prefix of the new one's, in either precision: the newest state is the
    one worth restoring. Another branch of the chat (an edited message) has a different prefix and
    is kept.
    """

    @staticmethod
    def find(
        new_name: str, new: _StoredEntry, stored: Iterable[tuple[str, _StoredEntry]]
    ) -> list[str]:
        if not new.tag:
            return []
        return [
            name
            for name, old in stored
            if name != new_name
            and old.tag == new.tag
            and old.model_id == new.model_id
            and old.length <= new.length
            and old.blocks == new.blocks[: len(old.blocks)]
        ]

    @staticmethod
    def chat_count(stored: Iterable[_StoredEntry]) -> int:
        """Distinct chats with a saved copy (files without a chat tag count one each)."""
        tags: set[str] = set()
        untagged = 0
        for entry in stored:
            if entry.tag:
                tags.add(entry.tag)
            else:
                untagged += 1
        return len(tags) + untagged
