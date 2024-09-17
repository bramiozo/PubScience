
# Sources

* [Google LLM API](https://aistudio.google.com/app/apikey)
* [Anthropic LLM](https://console.anthropic.com/dashboard)
* [OpenAI LLM](https://platform.openai.com)
* [Google Translate](https://cloud.google.com/translate?hl=en)
* [DeepL Translate](https://www.deepl.com/en/pro-api)

# API functionality

```python
from pubscience.translate import api as pubApi
from pubscience.share import parqueteur

kwargs = {'source_lang': 'en',
          'target_lang': 'nl'}

transAPI = pubApi(provider='google', glossary=None, n_jobs=2, cost_limit=100, **kwargs)
translator = transAPI.translate(list_of_texts) # iterator that gives a tuple(source, target)
parqueteur.from_translations(translator, )
```

# LLM functionality


# NTM functionality
