"""SQLite library prototype. Run: python prototype/library_app.py"""
import sqlite3
from pathlib import Path
DB_PATH = Path(__file__).with_name("library.db")

def connect():
    db = sqlite3.connect(DB_PATH)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    return db

def initialize_database():
    with connect() as db:
        db.executescript("""
        CREATE TABLE IF NOT EXISTS books(id INTEGER PRIMARY KEY, title TEXT NOT NULL,
          author TEXT NOT NULL, total_copies INTEGER NOT NULL CHECK(total_copies>0),
          available_copies INTEGER NOT NULL CHECK(available_copies>=0));
        CREATE TABLE IF NOT EXISTS students(id INTEGER PRIMARY KEY, name TEXT NOT NULL,
          email TEXT NOT NULL UNIQUE);
        CREATE TABLE IF NOT EXISTS issues(id INTEGER PRIMARY KEY, book_id INTEGER NOT NULL REFERENCES books(id),
          student_id INTEGER NOT NULL REFERENCES students(id), issue_date TEXT NOT NULL DEFAULT CURRENT_DATE,
          return_date TEXT);
        """)

def add_book(title, author, copies=1):
    if not title.strip() or not author.strip() or not isinstance(copies, int) or copies < 1:
        raise ValueError("Valid title, author and positive copy count are required.")
    with connect() as db:
        db.execute("INSERT INTO books(title,author,total_copies,available_copies) VALUES(?,?,?,?)",
                   (title.strip(), author.strip(), copies, copies))

def list_books():
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM books ORDER BY title")]

def register_student(name, email):
    if not name.strip() or "@" not in email:
        raise ValueError("Valid student name and email are required.")
    with connect() as db:
        db.execute("INSERT INTO students(name,email) VALUES(?,?)", (name.strip(), email.strip().lower()))

def list_students():
    with connect() as db:
        return [dict(r) for r in db.execute("SELECT * FROM students ORDER BY name")]

def issue_book(book_id, student_id):
    with connect() as db:
        book = db.execute("SELECT available_copies FROM books WHERE id=?", (book_id,)).fetchone()
        student = db.execute("SELECT id FROM students WHERE id=?", (student_id,)).fetchone()
        if book is None or student is None:
            raise ValueError("Book or student not found.")
        if book["available_copies"] < 1:
            raise ValueError("No available copies.")
        db.execute("UPDATE books SET available_copies=available_copies-1 WHERE id=?", (book_id,))
        db.execute("INSERT INTO issues(book_id,student_id) VALUES(?,?)", (book_id, student_id))

def return_book(issue_id):
    with connect() as db:
        issue = db.execute("SELECT book_id,return_date FROM issues WHERE id=?", (issue_id,)).fetchone()
        if issue is None:
            raise ValueError("Issue record not found.")
        if issue["return_date"] is not None:
            raise ValueError("Book already returned.")
        db.execute("UPDATE issues SET return_date=CURRENT_DATE WHERE id=?", (issue_id,))
        db.execute("UPDATE books SET available_copies=available_copies+1 WHERE id=?", (issue["book_id"],))

def list_issues():
    with connect() as db:
        return [dict(r) for r in db.execute("""SELECT i.id,b.title AS book,s.name AS student,
          i.issue_date,i.return_date FROM issues i JOIN books b ON b.id=i.book_id
          JOIN students s ON s.id=i.student_id ORDER BY i.id DESC""")]

if __name__ == "__main__":
    initialize_database()
    print("Library database initialized. Import this module to use the library functions.")
