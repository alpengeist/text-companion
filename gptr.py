import os
from openai import OpenAI


# Wrapper around Responses API
# There are some annoying peculiarities depending on the model type
# that we like to shield the caller from
def call_model(**kwargs):
    model = kwargs.get('model', '')
    args = {'model': model,
            'input': kwargs.get('input'),
            'stream': kwargs.get('stream', False),
            'instructions': kwargs.get('instructions'),
            }

    if model.startswith('o') or model.startswith('gpt-5'):
        # reasoning models
        args['max_output_tokens'] = kwargs.get('max_output_tokens', 50000)
        args['reasoning'] = kwargs.get('reasoning', {'effort': 'low', 'summary': 'auto'})
    else:
        # only non-reasoning models have temperature
        args['temperature'] = kwargs.get('temperature', 0)
        args['max_output_tokens'] = kwargs.get('max_output_tokens', 1500)

    # some params are only allowed if tools are present
    if tools := kwargs.get('tools'):
        args['tools'] = tools
        args['tool_choice'] = kwargs.get('tool_choice')
        args['parallel_tool_calls'] = kwargs.get('parallel_tool_calls', False)

    # this has not been solved elegantly in the API. Three different entrypoints.
    if kwargs.get('text_format'):
        args['text_format'] = kwargs.get('text_format')
        return client.responses.parse(**args)
    else:
        # if stream:
        #     return client.responses.stream(**args)
        # else:
        return client.responses.create(**args)

# -- initialization --
if 'OPENAI_API_KEY' in os.environ:
    client = OpenAI()  # will pick up API key from env
else:
    raise EnvironmentError(
        'Missing OPENAI_API_KEY. Make sure your .env is set correctly.'
    )
