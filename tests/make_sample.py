"""Generate a deliberately messy dataset that exercises every detector.

Run:  python tests/make_sample.py   -> sample_data/messy_employees.csv
"""
import csv
import os
import random

random.seed(7)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "sample_data", "messy_employees.csv")

first = ["Aarav", "Priya", "Rahul", "Sneha", "Vikram", "Ananya", "Karan", "Meera", "Rohan", "Isha",
         "John", "Maria", "José", "Zoë", "Chloé", "Liam", "Noah", "Emma", "Olivia", "Ava"]
last = ["Sharma", "Verma", "Patel", "Singh", "Gupta", "Mehta", "Reddy", "Iyer", "Khan", "Das",
        "Smith", "García", "Müller", "Brown", "Wilson"]
departments = ["Sales", "Marketing", "Engineering", "HR", "Finance"]
dept_variants = {
    "Sales": ["Sales", "sales", "SALES ", " Sales"],
    "Marketing": ["Marketing", "marketing", "Marketting", "Marketing "],
    "Engineering": ["Engineering", "engineering", "Engineeering", "ENGINEERING"],
    "HR": ["HR", "hr", "H.R.", "Hr "],
    "Finance": ["Finance", "finance", "Fnance", "FINANCE"],
}
cities = {
    "New York": ["New York", "new york", "New Yrok", "NEW YORK", "New  York"],
    "Mumbai": ["Mumbai", "mumbai", "Mumbai ", "Mumbay"],
    "London": ["London", "london", "Londn", "LONDON"],
    "Berlin": ["Berlin", "berlin", "Berlin "],
    "Bangalore": ["Bangalore", "bangalore", "Bengaluru", "Banglore"],
}
nulls = ["", "N/A", "n/a", "NA", "null", "-", "?", "None", "unknown", "NULL", "--"]


def mojibake(s: str) -> str:
    try:
        return s.encode("utf-8").decode("latin-1")
    except UnicodeError:
        return s


def date_variants(y, m, d):
    forms = [
        f"{y:04d}-{m:02d}-{d:02d}",
        f"{d:02d}/{m:02d}/{y:04d}",
        f"{m}/{d}/{y}",
        f"{['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][m-1]} {d}, {y}",
        f"{d:02d}-{m:02d}-{y:04d}",
        f"{d} {['January','February','March','April','May','June','July','August','September','October','November','December'][m-1]} {y}",
    ]
    return random.choice(forms)


rows = []
for i in range(1, 131):
    emp_id = f"EMP{i:04d}"
    name = f"{random.choice(first)} {random.choice(last)}"
    if random.random() < 0.15:
        name = mojibake(name) if any(ord(c) > 127 for c in name) else name
    if random.random() < 0.12:
        name = "  " + name + " "
    if random.random() < 0.05:
        name = name.replace(" ", "  ")
    dept_key = random.choice(departments)
    dept = random.choice(dept_variants[dept_key])
    city_key = random.choice(list(cities))
    city = random.choice(cities[city_key])
    age = random.randint(21, 60)
    age_s = str(age)
    r = random.random()
    if r < 0.06:
        age_s = random.choice(nulls)
    elif r < 0.09:
        age_s = "999"
    elif r < 0.11:
        age_s = "-5"
    elif r < 0.13:
        age_s = "150"
    elif r < 0.16:
        age_s = f" {age} "
    salary = random.randint(30, 150) * 1000
    r = random.random()
    if r < 0.3:
        salary_s = f"${salary:,}"
    elif r < 0.45:
        salary_s = f"Rs {salary:,}"
    elif r < 0.55:
        salary_s = f"{salary:,}"
    elif r < 0.62:
        salary_s = random.choice(nulls)
    elif r < 0.65:
        salary_s = str(salary * 40)      # outlier
    else:
        salary_s = str(salary)
    y, m, d = random.randint(2012, 2024), random.randint(1, 12), random.randint(1, 28)
    r = random.random()
    if r < 0.06:
        join = random.choice(nulls)
    elif r < 0.08:
        join = "2031-06-15"           # future
    elif r < 0.10:
        join = "01/01/1900"           # placeholder
    elif r < 0.12:
        join = "not available"
    else:
        join = date_variants(y, m, d)
    email = f"{name.strip().split()[0]}.{name.strip().split()[-1]}@example.com"
    email = mojibake(email) if False else email
    r = random.random()
    if r < 0.2:
        email = email.upper()
    elif r < 0.3:
        email = " " + email + " "
    elif r < 0.35:
        email = email.split("@")[0] + "@"   # invalid
    elif r < 0.4:
        email = random.choice(nulls)
    phone = random.choice([
        f"(555) {random.randint(100,999)}-{random.randint(1000,9999)}",
        f"555.{random.randint(100,999)}.{random.randint(1000,9999)}",
        f"+1 555 {random.randint(100,999)} {random.randint(1000,9999)}",
        f"555{random.randint(100,999)}{random.randint(1000,9999)}",
    ])
    active = random.choice(["Yes", "No", "yes", "no", "Y", "N", "TRUE", "FALSE", "1", "0", "true", "false"])
    rating = random.randint(40, 100)
    r = random.random()
    if r < 0.3:
        rating_s = f"{rating}%"
    elif r < 0.35:
        rating_s = "105"
    elif r < 0.4:
        rating_s = random.choice(nulls)
    else:
        rating_s = str(rating)
    notes = random.choice(["", "", "", "Great team player", "Needs improvement", "Promoted last year",
                           mojibake("Doesn’t meet deadlines"), mojibake("Café shift lead"), "   "])
    rows.append([emp_id, name, dept, city, age_s, salary_s, join, email, phone, active, rating_s, notes, "India", dept, ""])

# duplicates: exact and case/space variants
for _ in range(6):
    rows.append(list(random.choice(rows)))
for _ in range(4):
    r = list(random.choice(rows))
    r[1] = r[1].upper() + " "
    rows.append(r)
# duplicated ids on different people
rows.append(["EMP0007", "Duplicate Person", "Sales", "Berlin", "33", "40000", "2020-05-05", "dup@example.com", "5551234567", "Yes", "70", "", "India", "Sales", ""])
# missing ids
rows.append(["", "No Id Person", "HR", "London", "29", "45000", "2019-02-02", "noid@example.com", "5559876543", "No", "60", "", "India", "HR", ""])
# empty & mostly-empty rows
rows.append([""] * 15)
rows.append([""] * 15)
rows.append(["EMP9999", "", "", "", "", "", "", "", "", "", "", "", "India", "", ""])
random.shuffle(rows)

header = ["Employee ID", " Full Name ", "Department", "City", "Age", "Salary (USD)", "Join Date", "Email",
          "Phone#", "Is Active?", "Rating %", "Notes", "Country", "Department", "Unnamed: 14"]
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(header)
    w.writerows(rows)
print(f"wrote {OUT} with {len(rows)} rows")
