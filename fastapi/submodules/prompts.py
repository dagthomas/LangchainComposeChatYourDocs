def documentSearch(prompt, docs):
    template = f"""You are given the following extracted parts of a long document and a question, create a final answer with references.
	If you don't know the answer, just say that you don't know. Don't try to make up an answer.
	If you know the answer ALWAYS return the sources in the answer.
    Identify the language and reply in the indetified language. Do not output the identified language.
	=========
	QUESTION: {prompt}
	=========
	CONTENT: {docs}
	=========
	FINAL ANSWER:"""
    return template
