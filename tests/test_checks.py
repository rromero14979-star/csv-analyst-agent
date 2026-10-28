from analyst.checks import extract_numbers, unsupported_numbers


def test_extract_numbers_handles_commas_and_decimals():
    vals = [(v, d) for v, d, _ in extract_numbers("Revenue was 1,234,567.89 in 2024 (-3.5%).")]
    assert (1234567.89, 2) in vals and (2024.0, 0) in vals and (-3.5, 1) in vals


def test_rounded_number_is_supported():
    assert unsupported_numbers("Total revenue was 4,523.49.", "result\n4523.4876") == []


def test_percentage_forms_are_supported():
    assert unsupported_numbers("About 15.3% of orders.", "0.1530") == []
    assert unsupported_numbers("The rate is 0.153.", "15.30") == []


def test_invented_number_is_flagged():
    assert unsupported_numbers("Revenue was 999,999.", "total 4523.4876") == ["999,999"]


def test_numbers_from_the_question_are_ignored():
    assert unsupported_numbers("In 2024 the West led.", "West 123.4", "Which region led in 2024?") == []


def test_question_percentage_restated_as_fraction_is_ignored():
    q = "What percentage of orders had a discount of at least 20%?"
    assert unsupported_numbers("Orders with a discount of at least 0.2: 30.5%.", "30.5", q) == []


def test_percentage_range_is_not_read_as_a_negative_number():
    assert [r for _, _, r in extract_numbers("rates of 2%-6%")] == ["2", "6"]
    assert unsupported_numbers("Typical rates are 2%-6% [1].", "[1] site: between 2% and 6%") == []


def test_hundred_in_percentage_wording_is_not_flagged():
    assert unsupported_numbers("18.72% of lines (0.1872 x 100) lose money.", "0.1872") == []
