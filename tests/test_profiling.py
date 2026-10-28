import pytest

from analyst import config
from analyst.dictionary import DictionarySearch, load_dictionary_json, rule_based_dictionary
from analyst.profiling import DataLoadError, load_dataframe, profile_dataframe, profile_to_text


def test_demo_csv_loads_with_real_dates_and_profile():
    df = load_dataframe(config.DEMO_CSV)
    assert str(df["Order Date"].dtype).startswith("datetime64")
    text = profile_to_text(profile_dataframe(df))
    assert "'Order Date' (datetime)" in text and "'Customer Segment'" in text and "missing 57" in text
    assert "ORD-00001" not in text            # no raw rows in the prompt text


def test_empty_and_header_only_files_are_rejected():
    with pytest.raises(DataLoadError):
        load_dataframe(b"   ")
    with pytest.raises(DataLoadError):
        load_dataframe(b"a,b,c\n")


def test_too_large_file_is_rejected():
    with pytest.raises(DataLoadError):
        load_dataframe(b"a,b\n" + b"1,2\n" * 100, max_bytes=50)


def test_non_utf8_file_is_still_read():
    df = load_dataframe("이름,값\n홍길동,1\n".encode("cp949"))
    assert len(df) == 1 and list(df.columns) == ["이름", "값"]


def test_dictionary_json_and_search():
    df = load_dataframe(config.DEMO_CSV)
    entries = load_dictionary_json(config.DEMO_DICTIONARY, profile_dataframe(df))
    assert len(entries) == 10 and entries[7]["dtype"] == "number"
    top = DictionarySearch(entries)("what does the discount mean")
    assert top[0]["column"] == "Discount"
    assert len(rule_based_dictionary(profile_dataframe(df))) == 10


def test_binary_file_renamed_to_csv_is_rejected():
    with pytest.raises(DataLoadError):
        load_dataframe(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" * 50)


def test_semicolon_decimal_comma_and_report_lines_above_the_header():
    raw = "Sales report\nExported 2024-01-31\n\nRegion;Sales;Date\nNorth;1234,50;2024-01-05\nSouth;-10,25;2024-01-06\n"
    df = load_dataframe(raw.encode("utf-8"))
    assert list(df.columns) == ["Region", "Sales", "Date"] and df["Sales"].tolist() == [1234.5, -10.25]
    assert str(df["Date"].dtype).startswith("datetime64")


def test_currency_text_becomes_numbers():
    df = load_dataframe(b'Item,Price,Code\nA,"$1,234.56",001\nB,-$12.30,002\nC,$5.00,003\n')
    assert df["Price"].tolist() == [1234.56, -12.3, 5.0] and df["Code"].tolist() == [1, 2, 3]


def test_profile_flags_numbers_stored_as_text_next_to_placeholders():
    df = load_dataframe(b"Total\n4.0\n12.0\nERROR\n10.0\n20.0\n9.0\n")
    text = profile_to_text(profile_dataframe(df))
    assert "stored as TEXT" in text and "'ERROR'" in text
