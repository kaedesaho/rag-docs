from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()
client = OpenAI()

def generate_answer(query, chunks):
    context = "\n\n".join([c.page_content for c in chunks])
    
    prompt = f"""
    You are a helpful assistant that answers questions about LangChain and LlamaIndex documentation.
    Answer the question using only the provided context. If the answer is not in the context, say "I don't know."

    Context:
    {context}

    Question: {query}

    Answer:
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content