from graph import app

question = input("Ask a question: ")

result = app.invoke(
    {
        "question": question
    }
)

print("\n" + "=" * 50)
print("FINAL STATE")
print("=" * 50)

print(result)