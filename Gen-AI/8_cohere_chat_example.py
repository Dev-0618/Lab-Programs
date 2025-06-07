import cohere

co = cohere.Client("xSfqQtfn5Yw4lSguFJnzVNDhdexYMTzQwTLTVtsT")
print("You:")
response = co.chat(
    message=input()
)

print("Generated Response:\n", response.text)
