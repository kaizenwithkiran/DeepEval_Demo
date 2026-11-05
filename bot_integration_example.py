import time
from chat_logger import log_chat

def ask_and_log(question, your_rag_chain):
    """Adapt result keys below to match your own RAG chain."""
    start_time = time.time()

    result = your_rag_chain.invoke({"question": question})

    answer = result["answer"]       # Change if your chain uses another key.
    context = result["context"]     # Change if your chain uses another key.
    latency = time.time() - start_time

    # Replace 0.0 with actual calculated cost if your app tracks token usage.
    token_cost = 0.0

    log_chat(
        user_input=question,
        actual_output=answer,
        retrieved_context=context,
        token_cost=token_cost,
        latency=latency
    )
    return answer
