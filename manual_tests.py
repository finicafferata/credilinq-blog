import requests

# Upload a document to the knowledge base
"""file_path = "knowledge_base/Document1.txt"
with open(file_path, "rb") as f:
    files = {'file': f}
    data = {'document_title': 'Document 1'}
    resp = requests.post("http://localhost:8000/documents/upload", files=files, data=data)
    print("Status code:", resp.status_code)
    print("Raw response:", resp.text)
    try:
        print("JSON:", resp.json())
    except Exception as e:
        print("Error decoding JSON:", e)

"""
# Create a blog
data = {
    "title": "What is the breakeven point in cash flow terms?",
    "company_context": "Credilinq.ai is a fintech leader in embedded lending and B2B credit solutions across Southeast Asia, some countries in Europe and the US."
}
resp = requests.post("http://localhost:8000/blogs", json=data)
print(resp.json())
"""
# Get blogs
resp = requests.get("http://localhost:8000/blogs")
print(resp.json())  

# Details of a blog
resp = requests.get("http://localhost:8000/blogs/1")
print(resp.json())  

# Edit a blog
data = {
    "content_markdown": "New content"
}
resp = requests.put("http://localhost:8000/blogs/1", json=data)
print(resp.json())  

# Revise a blog
data = {
    "instruction": "Haz este párrafo más conciso.",
    "text_to_revise": "Este es un párrafo muy largo y repetitivo que podría ser más corto."
}
resp = requests.post("http://localhost:8000/blogs/1/revise", json=data)
print(resp.json())  """