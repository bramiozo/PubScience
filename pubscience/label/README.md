# Transform

This functionality is meant as a generic, unconstrained text-transformer using LLM's at first, and later also
encoder models (e.g. for random token replacement) and encoder/decoder models (e.g. for specific paraphrasing)

```python 
from pubscience import textTransform 

TextTransformer = textTransform(how='llm', model='gpt4o', system_prompt='paraphrase stuff for me')
newText = TextTransformer.transform(oldText)
```

or use a template, such that the incoming text is embedding in a template before it sent to the model.

```python 
from pubscience import transform 

template = """Rewrite this text and structure it as a discharge summary without translation:
[INPUT]
and create a version in Dutch. Only return the Dutch version."""

TextTransformer = transform.textTransform(how='llm', model='gpt4o', system_prompt='paraphrase, then translate', template=template)
newText = TextTransformer.transform(oldText)
```

and if you need to add an extraction 

```python 
from pubscience import textTransform 


template = """Rewrite this text and structure it as a discharge summary without translation:
[INPUT]
and create a version in Dutch preceded by <TRANSLATION>"""

TextTransformer = transform.textTransform(how='llm', model='gpt4o', system_prompt='paraphrase, then translate', template=template)
newTextRAW = TextTransformer.transform(oldText)

newText = TextTransform.extract(by="<TRANSLATION>", how="after")
```

and you can chain prompts.


