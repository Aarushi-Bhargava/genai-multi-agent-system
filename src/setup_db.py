"""
setup_db.py
Creates a SQLite database with fake customer, ticket, and order data.
Run this ONCE before starting the app.
"""

import sqlite3
import random
from pathlib import Path
from faker import Faker

# Where the database file will be saved
DB_PATH = Path(__file__).parent.parent / "data" / "customers.db"
DB_PATH.parent.mkdir(exist_ok=True)

# If a database already exists, delete it so we start fresh
if DB_PATH.exists():
    DB_PATH.unlink()

# Connect to SQLite — this creates the file if it doesn't exist
conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

# 1. Create the three tables
cur.executescript("""
CREATE TABLE customers (
    customer_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT UNIQUE NOT NULL,
    plan          TEXT NOT NULL,           -- 'Free', 'Pro', 'Enterprise'
    signup_date   TEXT NOT NULL,
    country       TEXT NOT NULL
);

CREATE TABLE tickets (
    ticket_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL,
    subject       TEXT NOT NULL,
    description   TEXT NOT NULL,
    status        TEXT NOT NULL,           -- 'open', 'in_progress', 'resolved', 'closed'
    priority      TEXT NOT NULL,           -- 'low', 'medium', 'high', 'urgent'
    created_at    TEXT NOT NULL,
    resolution    TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE orders (
    order_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id   INTEGER NOT NULL,
    product       TEXT NOT NULL,
    amount_usd    REAL NOT NULL,
    order_date    TEXT NOT NULL,
    status        TEXT NOT NULL,           -- 'pending', 'shipped', 'delivered', 'refunded'
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
""")

# 2. Generate fake customers
fake = Faker()
Faker.seed(42)       
random.seed(42)

# Apple customer types: whether they own devices, subscribe to services, or hold a warranty plan
customer_segments = ["Standard", "AppleCare+ Holder", "Apple One Subscriber"]
countries = ["USA", "Canada", "UK", "India", "Germany", "Australia"]

customers = []
for _ in range(200):
    name = fake.name()
    email = fake.unique.email()
    plan = random.choice(customer_segments)
    signup = fake.date_between(start_date="-3y", end_date="today").isoformat()
    country = random.choice(countries)
    customers.append((name, email, plan, signup, country))

cur.executemany(
    "INSERT INTO customers (name, email, plan, signup_date, country) "
    "VALUES (?, ?, ?, ?, ?)",
    customers
)

# 3. Generate fake tickets
ticket_subjects = [
    ("Return request for iPhone 15",              "I want to return my iPhone 15 purchased 10 days ago. Is it eligible under the 14-day return window?"),
    ("Refund for cancelled order",                "I cancelled my MacBook Air order before shipment but haven't received my refund yet."),
    ("AppleCare+ cancellation and refund",        "I'd like to cancel my AppleCare+ plan for my iPad and want to know if I'm entitled to a pro-rated refund."),
    ("Screen replacement under AppleCare+",       "My iPhone screen cracked from a drop. What's the service fee under my AppleCare+ coverage?"),
    ("Defective AirPods Pro",                     "My AirPods Pro purchased 3 weeks ago have stopped charging in the right ear. Requesting warranty service."),
    ("Exchange MacBook for different configuration","I received my custom-configured MacBook but want to exchange it for a different model. Is exchange possible?"),
    ("Gift return — iPad",                        "I received an iPad as a gift and want to return it. How do I do this without the original buyer's account?"),
    ("Apple Gift Card refund issue",              "Part of my refund was sent as an Apple Gift Card instead of back to my credit card. Why?"),
    ("Battery replacement inquiry",               "My iPhone battery capacity has dropped below 80%. Is this covered under AppleCare+?"),
    ("Price protection request",                  "Apple lowered the price of the iPad I bought 7 days ago. Can I request a refund for the price difference?"),
    ("In-store pickup not ready",                 "I selected in-store pickup but haven't been notified that my order is ready. It's been 3 days."),
    ("Wireless service not cancelled after return","I returned my iPhone but my carrier is still charging me. Who is responsible for cancelling the plan?"),
]

statuses   = ["open", "in_progress", "resolved", "closed"]
priorities = ["low", "medium", "high", "urgent"]
resolutions = {
    "resolved": "Issue resolved after troubleshooting steps provided by agent.",
    "closed":   "Customer confirmed issue no longer occurring. Ticket closed.",
    "open":     None,
    "in_progress": None,
}

tickets = []
for _ in range(800):
    cust_id = random.randint(1, 200)
    subj, desc = random.choice(ticket_subjects)
    status = random.choice(statuses)
    tickets.append((
        cust_id, subj, desc, status,
        random.choice(priorities),
        fake.date_between(start_date="-1y", end_date="today").isoformat(),
        resolutions[status],
    ))

cur.executemany(
    "INSERT INTO tickets (customer_id, subject, description, status, priority, created_at, resolution) "
    "VALUES (?, ?, ?, ?, ?, ?, ?)",
    tickets
)

# 4. Generating fake orders
products = [
    ("iPhone 15",                         799.00),
    ("iPhone 15 Pro",                    1099.00),
    ("iPad Air",                          599.00),
    ("iPad Pro 11\"",                     999.00),
    ("MacBook Air 13\"",                 1199.00),
    ("MacBook Pro 14\"",                 1999.00),
    ("AirPods Pro",                       249.00),
    ("AirPods Max",                       549.00),
    ("Apple Watch Series 10",             399.00),
    ("Apple Watch Ultra 2",               799.00),
    ("AppleCare+ for iPhone (2 years)",   199.00),
    ("AppleCare+ for Mac (3 years)",      379.00),
    ("Apple Gift Card - $100",            100.00),
]
order_statuses = ["pending", "shipped", "delivered", "refunded"]

orders = []
for _ in range(500):
    cust_id = random.randint(1, 200)
    product, amount = random.choice(products)
    orders.append((
        cust_id, product, amount,
        fake.date_between(start_date="-1y", end_date="today").isoformat(),
        random.choice(order_statuses),
    ))

cur.executemany(
    "INSERT INTO orders (customer_id, product, amount_usd, order_date, status) "
    "VALUES (?, ?, ?, ?, ?)",
    orders
)

# Save everything and close
conn.commit()
conn.close()

print(f"Database created at {DB_PATH}")
print(f"   - 200 customers")
print(f"   - 800 tickets")
print(f"   - 500 orders")