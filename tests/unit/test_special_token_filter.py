"""SpecialTokenTextFilter (app/runtime/special_token_filter.py): a model-agnostic safety net for a
leaked `<|...|>` special token in a chat stream (seen as `<|im_start|>assistant` from
a DictaLM model). Complements chat_stop_tokens' ChatML-EOS fix."""

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
    """Text before an unresolved `<|` is emitted at once; only the ambiguous tail is buffered until
    flush."""
    f = SpecialTokenTextFilter()
    assert f.feed("cost <|") == "cost "
    assert f.flush() == "<|"


def test_text_that_looks_like_a_tag_opener_but_isnt_is_not_dropped() -> None:
    f = SpecialTokenTextFilter()
    # A space breaks the tag shape, so "<|" is ordinary text; raw pipes typed by a user must not
    # lose characters.
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
