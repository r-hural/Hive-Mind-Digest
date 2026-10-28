from pipeline.links import kind_noun, locator_label, safe_url, source_link
from pipeline.schemas import Anchor

A = Anchor(char_start=0, char_end=10, t_start=754.0, score=100, segment=4, page=3)


def test_links_per_kind():
    assert source_link({"kind": "youtube", "video_id": "aaaaaaaaaaa", "url": "u"}, A, "q") == \
        "https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s"
    assert source_link({"video_id": "aaaaaaaaaaa", "url": "u"}, A, "q") == \
        "https://www.youtube.com/watch?v=aaaaaaaaaaa&t=754s"   # a row without `kind` is YouTube
    assert source_link({"kind": "video", "url": "https://vimeo.com/1"}, A, "q") == "https://vimeo.com/1#t=754s"
    assert source_link({"kind": "video", "url": "https://ted.com/talks/x"}, A, "q") == "https://ted.com/talks/x"
    assert source_link({"kind": "pdf", "url": "https://e.org/p.pdf"}, A, "q") == "https://e.org/p.pdf#page=3"
    nopage = Anchor(char_start=0, char_end=10, t_start=0.0, score=100)
    assert source_link({"kind": "pdf", "url": "https://e.org/p.pdf"}, nopage, "q") == "https://e.org/p.pdf"
    assert source_link({"kind": "text", "url": "file:///C:/n.txt"}, A, "q") == "file:///C:/n.txt"
    assert source_link({"kind": "media", "url": "file:///C:/a.mp3"}, A, "q") == "file:///C:/a.mp3"
    # text fragments: the first 8 words, percent-encoded, with "-", "&" and "," encoded too
    web = source_link({"kind": "web", "url": "https://e.org/a"}, A, "Small changes, compound - over months & years ok")
    assert web == "https://e.org/a#:~:text=Small%20changes%2C%20compound%20%2D%20over%20months%20%26%20years"


def test_text_fragment_skips_extracted_list_and_table_markup():
    # Page text is extracted with list bullets and table pipes (trafilatura txt output); a browser
    # never sees those characters and cannot match across table cells, so the fragment drops a
    # leading bullet and stays inside the first non-empty cell.
    web = {"kind": "web", "url": "https://e.org/a"}
    assert source_link(web, A, "- 3 had unrealistically high modeled asymptotes (above 42)") == \
        "https://e.org/a#:~:text=3%20had%20unrealistically%20high%20modeled%20asymptotes%20%28above%2042%29"
    assert source_link(web, A, "| Time to 95% of asymptote | 66 days | 18-254 |") == \
        "https://e.org/a#:~:text=Time%20to%2095%25%20of%20asymptote"
    assert source_link(web, A, "|Exercise| 13 | 91 days |") == "https://e.org/a#:~:text=Exercise"
    assert source_link(web, A, "* one  two") == "https://e.org/a#:~:text=one%20two"
    assert source_link(web, A, "| - |") == "https://e.org/a"     # nothing matchable: the bare URL


def test_urls_never_break_markdown_link_syntax():
    assert safe_url("file:///C:/My Docs/a (1).pdf") == "file:///C:/My%20Docs/a%20%281%29.pdf"
    assert source_link({"kind": "text", "url": "file:///C:/a (1).txt"}, A, "q") == "file:///C:/a%20%281%29.txt"
    assert safe_url("https://e.org/a|b<c>d e") == "https://e.org/a%7Cb%3Cc%3Ed%20e"
    assert safe_url("https://e.org/caf\u00e9") == "https://e.org/caf%C3%A9"
    kept = "https://u@e.org:8080/p;x=1/a%20b?q=1&r=a+b,c*d~e!f$g'h#frag[0]"
    assert safe_url(kept) == kept                     # URL syntax and existing escapes are left alone


def test_locator_labels():
    assert locator_label({"kind": "youtube"}, A) == "12:34"
    assert locator_label({}, A) == "12:34"
    assert locator_label({"kind": "pdf"}, A) == "p. 3"
    assert locator_label({"kind": "web"}, A) == "\u00b6 5"
    plain = Anchor(char_start=0, char_end=10, t_start=0.0, score=100)
    assert locator_label({"kind": "pdf"}, plain) == "PDF"
    assert locator_label({"kind": "web"}, plain) == "text"
    assert locator_label({"kind": "text"}, plain) == "text"


def test_kind_nouns():
    assert (kind_noun("youtube", 1), kind_noun("web", 2), kind_noun("pdf", 2)) == ("video", "articles", "PDFs")
    assert (kind_noun("video", 2), kind_noun("media", 1), kind_noun("text", 3)) == ("videos", "recording", "documents")


def test_claims_hash_of_old_analyses_is_unchanged():
    from pipeline.schemas import Analysis, claims_hash
    from tests.test_output import _analysis
    assert claims_hash(Analysis.model_validate(_analysis(754.0))) == \
        "085b9ee14f1023d97a0d6ab69fffe21460eb8571b9576c1089273f5708717984"
    # a new anchor field set changes the hash (it is real content), unset fields do not
    data = _analysis(754.0)
    data["key_claims"][0]["anchor"]["segment"] = 2
    assert claims_hash(Analysis.model_validate(data)) != claims_hash(Analysis.model_validate(_analysis(754.0)))
