import urllib.request, sys
query = sys.argv[1]
url = f'http://export.arxiv.org/api/query?search_query={query}&start=0&max_results=1'

with urllib.request.urlopen(url) as reader:
    print(reader.read())
