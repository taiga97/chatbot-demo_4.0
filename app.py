import chainlit as cl
import langchain_community
from langchain_community.document_loaders.text import TextLoader
# プロンプトを定義する. This will be sent to openai.
from langchain_core.prompts.prompt import PromptTemplate

prompt = PromptTemplate(
    template="""
    文章を前提にして質問に答えてください。

    文章 :
    {document}

    質問 : {query}
    """,
    input_variables=["document", "query"],
)


@cl.on_chat_start
async def on_chat_start():
# Define a hook that is called when a new chat session is created.

    # PDFを読み込む処理
    files = None

        # By placing await before cl.AskFileMessage().send(),
        # the code tells Python to pause the execution of the current function (on_chat_start) at that point.
    while files is None:
        # chainlitの機能に、ファイルをアップロードさせるメソッドがある。
        files = await cl.AskFileMessage(
            # Maximum file size in MB. Defaults to 2.
            max_size_mb=20,
            # Text displayed above the upload button.
            content="Upload a file, accepting .pdf/.xlsx/.doxs",
            # PDFファイルを指定する
            accept=["application/pdf","application/vnd.openxmlformats-officedocument.spreadsheetml.sheet","application/vnd.openxmlformats-officedocument.wordprocessingml.document"],
            # タイムアウトなし
            raise_on_timeout=False,
        # The send() method of the AskFileMessage object is called to initiate the file upload process.
        # The returned value from send() is assigned to the files variable.
        ).send()


    if files is not None:
        for file in files:
            if file.name.endswith('.pdf'):
                from langchain_community.document_loaders import PyMuPDFLoader
                documents = PyMuPDFLoader(file.path).load()

            # Load word files
            if file.name.endswith('.docx'):
                from langchain_community.document_loaders import Docx2txtLoader
                documents = Docx2txtLoader(file.path).load()

            # Load xlsx files
            if file.name.endswith('.xlsx'):
                from langchain_community.document_loaders.excel import \
                    UnstructuredExcelLoader
                documents = UnstructuredExcelLoader(file.path).load()

    # Split the text into chunks
    from langchain.text_splitter import SpacyTextSplitter
    text_splitter = SpacyTextSplitter(chunk_size=400, pipeline="ja_core_news_sm")
    splitted_documents = text_splitter.split_documents(documents)

    # Create a Chroma vector store
    from langchain.embeddings.openai import OpenAIEmbeddings
    from langchain_community.vectorstores import Chroma
    embeddings = OpenAIEmbeddings()
    docsearch = await cl.make_async(Chroma.from_documents)(splitted_documents, embeddings)

    from langchain.memory import ChatMessageHistory
    message_history = ChatMessageHistory()

    from langchain.memory import ConversationBufferMemory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        output_key="answer",
        chat_memory=message_history,
        return_messages=True,
    )

    # Create a chain that uses the Chroma vector store
    from langchain.chains.conversational_retrieval.base import \
        ConversationalRetrievalChain
    from langchain_openai import ChatOpenAI
    chain = ConversationalRetrievalChain.from_llm(
        ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        memory=memory,
        return_source_documents=True,
    )

    # Send a response back to the user
    await cl.Message(content=f"You can now ask questions!").send()
    cl.user_session.set("chain", chain)


# define a hook that is called when a new message is received from the user.
@cl.on_message
async def main(message: cl.Message):
    chain = cl.user_session.get("chain")
    cb = cl.AsyncLangchainCallbackHandler()

    res = await chain.acall(message.content, callbacks=[cb])
    answer = res["answer"]
    source_documents = res["source_documents"]  # type: List[Document]

    text_elements = []  # type: List[cl.Text]

    if source_documents:
        for source_idx, source_doc in enumerate(source_documents):
            source_name = f"source_{source_idx}"
            # Create the text element referenced in the message
            text_elements.append(
                cl.Text(content=source_doc.page_content, name=source_name)
            )
        source_names = [text_el.name for text_el in text_elements]

        if source_names:
            answer += f"\nSources: {', '.join(source_names)}"
        else:
            answer += "\nNo sources found"

    await cl.Message(content=answer, elements=text_elements).send()