from src.trie_index import Trie, benchmark_lookup_methods


def _sample_trie():
    t = Trie()
    t.insert("घर", {"freq": 100})
    t.insert("घरमा", {"freq": 50})
    t.insert("घरहरू", {"freq": 40})
    t.insert("साथी", {"freq": 70})
    return t


def test_insert_and_exists():
    t = _sample_trie()
    assert t.exists("घर") is True
    assert t.exists("घरमा") is True
    assert t.exists("साथी") is True
    assert t.exists("नदी") is False
    assert t.exists("घ") is False  # prefix only, not a complete entry


def test_count():
    t = _sample_trie()
    assert t.count() == 4
    t.insert("घर", {"freq": 999})  # re-inserting an existing key must not increase count
    assert t.count() == 4


def test_metadata_retrieval():
    t = _sample_trie()
    assert t.get_metadata("घर") == {"freq": 100}
    assert t.get_metadata("घरमा") == {"freq": 50}
    assert t.get_metadata("नदी") is None


def test_prefix_matches_ordering():
    t = _sample_trie()
    matches = t.prefix_matches("घरमाथि")
    assert matches == ["घर", "घरमा"]


def test_prefix_matches_no_match():
    t = _sample_trie()
    assert t.prefix_matches("नदीमा") == []


def test_longest_prefix_match():
    t = _sample_trie()
    assert t.longest_prefix_match("घरमाथि") == "घरमा"
    assert t.longest_prefix_match("नदी") is None


def test_starts_with():
    t = _sample_trie()
    assert t.starts_with("घ") is True
    assert t.starts_with("घर") is True
    assert t.starts_with("न") is False


def test_empty_key_insert_is_noop():
    t = Trie()
    t.insert("", {"x": 1})
    assert t.count() == 0
    assert t.exists("") is False


def test_all_keys():
    t = _sample_trie()
    assert set(t.all_keys()) == {"घर", "घरमा", "घरहरू", "साथी"}


def test_benchmark_runs_and_returns_expected_shape():
    vocab = ["घर", "घरमा", "साथी", "किताब", "पानी"] * 20
    queries = ["घर", "साथी", "नदी", "घरमाथि"] * 10
    result = benchmark_lookup_methods(vocab, queries, repeats=1)
    assert set(result.keys()) == {"exact", "prefix"}
    for mode in ("exact", "prefix"):
        for method in ("trie", "dict", "linear"):
            assert result[mode][method] >= 0.0
