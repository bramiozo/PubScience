# Identify

Identify relevant documents:
* rule-based:
  * keyword presence
  * phrase presence
* classification-based


```python
from pubscience import Identify

identifier = Identify(model_name="UMCU/SomeModel", inclusion_terms=['oompa'], exlusion_terms=['loompa'])
relevant_ids = identifier.GetIds(input_location="/location/of/file/to/parse.jsonl")
```
