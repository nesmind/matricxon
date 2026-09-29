"""SpecialTokenTextFilter (app/runtime/special_token_filter.py) - the general, model-agnostic
safety net for a leaked `<|...|>`-shaped special token in a live chat stream, added 2026-09-29
after a real DictaLM-3.0-1.7B-Thinking chat produced a visible `<|im_start|>assistant` in its
reply. Complementary to app.runtime.chat_stop_tokens's own targeted ChatML-EOS fix - this catches
a leak regardless of *why* generation didn't stop before it.
"""

from app.runtime.special_token_filter import SpecialTokenTextFilter


def test_plain_text_passes_through_unchanged() -> None:
    f = SpecialTokenTextFilter()
    assert f.feed("Hello, world!") == "Hello, world!"
    assert f.flush() == ""


def test_a_whole_tag_in_one_chunk_is_stripped() -> None:
    f = SpecialTokenTextFilter()
    assert f.feed("<|im_start|>assistant\n") == "assistant\n"


def test_a_tag_between_real_text_is_stripped_and_the_rest_kept() -> None:
    f = SpecialTokenTextFilter()
    assert f.feed("before<|im_end|>after") == "beforeafter"


def test_a_tag_split_across_two_chunks_is_still_stripped() -> None:
    f = SpecialTokenTextFilter()
    first = f.feed("Hello <|im")
    second = f.feed("_end|> bye")
    assert first + second == "Hello  bye"


def test_a_lone_pipe_bracket_with_no_closing_tag_is_kept_verbatim_at_flush() -> None:
    """Text before an unresolved `<|` is emitted right away (no reason to hold it back); only
    the ambiguous `<|` tail itself is buffered until flush()."""
    f = SpecialTokenTextFilter()
    assert f.feed("cost <|") == "cost "
    assert f.flush() == "<|"


def test_text_that_looks_like_a_tag_opener_but_isnt_is_not_dropped() -> None:
    f = SpecialTokenTextFilter()
    # A space breaks the real tag shape - "<|" here is never going to close, so it's ordinary
    # text (matricxon's own docs use "a<|b" nowhere for real, but a user typing raw pipes must
    # not silently lose characters).
    assert f.feed("a<|b calc") == "a<|b calc"


def test_a_tag_shaped_span_that_never_closes_gives_up_after_a_bounded_length() -> None:
    f = SpecialTokenTextFilter()
    long_text = "<|" + "a" * 100 + " done"
    assert f.feed(long_text) == long_text


def test_multiple_real_tags_in_one_chunk_are_all_stripped() -> None:
    f = SpecialTokenTextFilter()
    assert f.feed("<|im_start|>assistant<|im_end|>") == "assistant"


def test_flush_with_nothing_pending_returns_empty_string() -> None:
    f = SpecialTokenTextFilter()
    f.feed("plain text")
    assert f.flush() == ""
