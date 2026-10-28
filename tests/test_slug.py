from pipeline.slug import slugify, unique_slug


def test_basic():
    assert slugify("Acme SEO in 2026!") == "acme-seo-in-2026"


def test_unicode_folds_to_ascii():
    assert slugify("Café Résumé") == "cafe-resume"


def test_length_capped_without_trailing_dash():
    s = slugify("a" * 39 + " b" + "c" * 20)
    assert len(s) <= 40 and not s.endswith("-")


def test_windows_reserved_names_are_suffixed():
    assert slugify("CON") == "con-nb"
    assert slugify("lpt1") == "lpt1-nb"


def test_empty_becomes_notebook():
    s = slugify("!!!")
    assert s.startswith("notebook-") and len(s) == 17


def test_non_latin_topics_get_distinct_stable_slugs():
    a = slugify("机器学习")
    b = slugify("Машинное обучение")
    assert a != b
    assert slugify("机器学习") == a


def test_truncate_before_reserved_check():
    assert slugify("connect", max_len=3) == "con-nb"


def test_unique_slug_never_returns_bare_reserved_name():
    assert unique_slug("con", {"con"}) == "con-2"


def test_unique_slug_appends_counter():
    assert unique_slug("x", {"x", "x-2"}) == "x-3"
    assert unique_slug("y", {"x"}) == "y"
