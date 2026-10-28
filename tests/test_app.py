"""UI smoke tests with Streamlit's AppTest (no network; graph replaced by fakes)."""
from streamlit.testing.v1 import AppTest

from analyst import factory
from analyst.builder import build_graph
from analyst.dictionary import DictionarySearch
from tests.fakes import fake_nodes


def patch(monkeypatch):
    monkeypatch.setattr(factory, "prepare_dataset", lambda key, df, dictionary=None, **k: type("Ctx", (), {"dictionary": dictionary or [], "df": df})())
    monkeypatch.setattr(factory, "create_graph", lambda *a, **k: build_graph(fake_nodes()))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")


def test_app_renders_and_answers(monkeypatch):
    patch(monkeypatch)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert not at.exception
    at.chat_input[0].set_value("Which region had the highest revenue?").run()
    assert not at.exception
    assert [m.name for m in at.chat_message] == ["user", "assistant"]


def test_app_missing_keys_asks_for_the_users_own_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    assert any("Enter your OpenAI API key" in i.value for i in at.info)
    at.chat_input[0].set_value("total revenue?").run()
    assert not at.exception
    assert any("Add your OpenAI API key" in w.value for w in at.warning)


def test_app_users_openai_key_without_tavily_turns_web_search_off(monkeypatch):
    patch(monkeypatch)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    seen = {}
    monkeypatch.setattr(factory, "create_graph", lambda *a, **k: seen.update(k) or build_graph(fake_nodes()))
    at = AppTest.from_file("app.py", default_timeout=30).run()
    at.text_input[0].input("sk-user").run()
    at.chat_input[0].set_value("total revenue?").run()
    assert not at.exception and not at.error and seen["web_enabled"] is False
    assert any("Web search is off" in c.value for c in at.caption)


def test_app_example_button_runs_a_question(monkeypatch):
    patch(monkeypatch)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    [b for b in at.button if b.label.startswith("S1")][0].click().run()
    assert not at.exception and len(at.chat_message) == 2


def test_app_history_renders_only_the_messages(monkeypatch):
    patch(monkeypatch)
    at = AppTest.from_file("app.py", default_timeout=30).run()
    at.chat_input[0].set_value("Which region had the highest revenue?").run()
    at.run()  # rerun: the history loop renders the conversation again
    user = at.chat_message[0]
    assert [e.value for e in user.markdown] == ["Which region had the highest revenue?"]
    assert not any("None" in str(e.value) for e in at.chat_message[1].markdown)
