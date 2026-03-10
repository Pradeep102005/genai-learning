from typing import TypedDict

class Person(TypedDict):
    name: str
    age: int
    city: str

person: Person = {
    "name": "John",
    "age": 30,
    "city": "New York"
}

print(person)
