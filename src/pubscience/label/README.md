# Label

This functionality is meant as a poor-man's automatic labeler using LLMs, off or on-premise.

```python
from pubscience import label

TextLabeler = label.extract(how='llm', model='gpt4o')
label = TextLabeler.transform(oldText)
```
