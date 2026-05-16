def summarize_documents(chunks):

    text = ""

    for chunk in chunks[:3]:

        text += chunk.page_content

    return text[:2000]