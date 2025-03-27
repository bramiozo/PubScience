"""
Batch processing is available for OpenAI, Anthropic and Google models and allows for large amounts of documents to be processed in a single request, also for these
vendors it is 50% cheaper than the regular processing.

OpenAI: expect the upload of a JSONL, see https://platform.openai.com/docs/guides/batch
Anthropic: expects the upload of request via a POST, this returns a result URL that can be used to download the results, see https://docs.anthropic.com/en/docs/build-with-claude/message-batches
Google: requires the use of vertexai and the upload of a JSONL to a GCS bucket, see https://cloud.google.com/vertex-ai/generative-ai/docs/model-reference/batch-prediction-api#generative-ai-batch-text-

The batch processing feature is available for the following models:
- OpenAI: 50.000 messages or 200MB of data, gpt-4o-mini
- Anthropic: 100.000 messages or 256MB of data, claude-3-5-haiku-20241022
- Google: gemini-1.5-flash-002
"""



class OpenAI:
    '''
        Sources:
        - https://platform.openai.com/docs/guides/batch
    '''
    def __init__(self, api_key: str):
        self.api_key = api_key

    def batch(self, file_path: str, model: str, output_file: str):
        pass


class Anthropic:
    '''
        Sources:
        - https://docs.anthropic.com/en/docs/build-with-claude/batch-processing
    '''
    pass

class Google:
    '''
        Sources:
        - https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/batch-prediction-gemini#generative-ai-batch-text-python_genai_sdk
    '''
    pass
