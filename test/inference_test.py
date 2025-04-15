import os
import json
import sys

os.environ["TOKENIZERS_PARALLELISM"] = "false"


from mode_rag import (
    EmbeddingGenerator,
    ModeInference,
    ModelPrompt,
)


main_processor = ModeInference(
    persist_directory="attention",
)

print("====start======")
# Create a PromptManager instance

query = "What are the key mathematical operations involved in computing self-attention?"

embed_gen = EmbeddingGenerator()
embedding = embed_gen.generate_embedding(query)

prompts = ModelPrompt(
    ref_sys_prompt="Use the following pieces of context to answer the user's question. \nIf you don't know the answer, just return you don't know.",
    ref_usr_prompt="context: ",
    syn_sys_prompt="You have been provided with a set of responses from various models to the latest user query. Your task is to synthesize these responses into a single, high-quality response. It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect. Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction. Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\nResponses from models:",
    syn_usr_prompt="responses:",
)

response = main_processor.invoke(
    query,
    embedding,
    prompts,
    model_input={"temperature": 0.3, "model": "openai/gpt-4o-mini"},
    top_n_model=2,
)
print(response)
