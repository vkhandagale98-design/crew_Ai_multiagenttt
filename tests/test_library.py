import pytest
from prototype import library_app as lib

@pytest.fixture(autouse=True)
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(lib, "DB_PATH", tmp_path / "test.db")
    lib.initialize_database()

def test_add_book():
    lib.add_book("Python", "Author", 2)
    assert lib.list_books()[0]["available_copies"] == 2

def test_register_student():
    lib.register_student("A Student", "a@example.com")
    assert lib.list_students()[0]["email"] == "a@example.com"

def test_issue_and_return():
    lib.add_book("SQL", "Writer")
    lib.register_student("B Student", "b@example.com")
    lib.issue_book(1, 1)
    assert lib.list_books()[0]["available_copies"] == 0
    issue_id = lib.list_issues()[0]["id"]
    lib.return_book(issue_id)
    assert lib.list_books()[0]["available_copies"] == 1

def test_unavailable_book_rejected():
    lib.add_book("Only copy", "Writer")
    lib.register_student("C Student", "c@example.com")
    lib.issue_book(1, 1)
    with pytest.raises(ValueError, match="No available copies"):
        lib.issue_book(1, 1)

def test_double_return_rejected():
    lib.add_book("Book", "Writer")
    lib.register_student("D Student", "d@example.com")
    lib.issue_book(1, 1)
    issue_id = lib.list_issues()[0]["id"]
    lib.return_book(issue_id)
    with pytest.raises(ValueError, match="already returned"):
        lib.return_book(issue_id)
