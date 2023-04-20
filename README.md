# ChatbotPyOpenAi
A conversational agent using the OpenAI API Chat Completion endpoint.  It maintains a conversational memory for generating responses.

As the conversation progresses through user input prompts and API completion responses, the conversation context is stored in a historical chat context. As the historical context approaches the API endpoint's total token limit (the largest number of words we can send it and receive in one call), it prunes context list to manage memory and summarizes the conversation so far to retain some of what's pruned.

Place an OpenAI API key either directly in the chatbotpyopenai.py code, or create a config.ini in the same folder, and put the API key (without quotes) under an [API_KEYS] heading, then OPENAI_API_KEY = api_key_here
