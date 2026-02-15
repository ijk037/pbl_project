from pymongo import MongoClient
from faker import Faker
import bcrypt
import random

fake = Faker("en_IN")

MONGO_URI = "mongodb://localhost:27017/pbl"

client = MongoClient(MONGO_URI)
db = client.get_database()

students_collection = db["students"]
faculty_collection = db["faculties"]

students_collection.delete_many({})
faculty_collection.delete_many({})

subjects = [
    "Mathematics",
    "Physics",
    "Data Structures",
    "DBMS",
    "Operating Systems",
    "Computer Networks"
]

departments = [
    "Computer Science",
    "Electronics",
    "Mechanical",
    "Civil",
    "AI & ML"
]

def hash_password(password):
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

# ---------- Generate Students ----------

students = []

for i in range(50):

    performances = []

    for subject in subjects:
        performances.append({
            "subject": subject,
            "midSem": random.randint(50, 95),
            "endSem": random.randint(55, 98)
        })

    name = fake.name()

    student = {
        "name": name,
        "email": f"{name.lower().replace(' ', '')}{i}@college.com",
        "regNo": f"22CS{100 + i}",
        "password": hash_password("1234"),
        "department": random.choice(departments),
        "year": random.choice([1, 2, 3, 4]),
        "performances": performances
    }

    students.append(student)

students_collection.insert_many(students)

# ---------- Generate Faculties ----------

faculties = []

for i in range(5):

    name = fake.name()

    faculty = {
        "name": name,
        "email": f"{name.lower().replace(' ', '')}@college.com",
        "password": hash_password("1234"),
        "department": random.choice(departments),
        "experienceYears": random.randint(3, 25)
    }

    faculties.append(faculty)

faculty_collection.insert_many(faculties)

print("✅ FULL DATABASE GENERATED SUCCESSFULLY")
