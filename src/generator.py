from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=False)
client = OpenAI()

def generate_answer(query, chunks):
    context = "\n\n".join([c.page_content for c in chunks])
    
    prompt = f"""You are a helpful assistant that answers questions about LangChain and LlamaIndex documentation.

    Answer the question using only the provided context. Be concise and directly useful. Include a code example if one exists in the context and is relevant.
    
    Do not refer the user to external documentation or links. Do not say "refer to the documentation." If the context doesn't contain enough information to answer, say "The documentation doesn't cover this specifically" and summarize what related information is available.

    Context:
    {context}

    Question: {query}

    Answer:"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    return response.choices[0].message.content