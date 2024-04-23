import openai
import os
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from qdrant_client import QdrantClient
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import  RunnablePassthrough, RunnableParallel
import logging
from langchain_community.document_loaders import  Docx2txtLoader,PyMuPDFLoader
from langchain_community.vectorstores import Qdrant
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.docstore.document import Document
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
openai.api_key  = os.getenv("OPENAI_API_KEY")
qdrant_url  = os.getenv('qdrant_url')
qdrant_api_key  = os.getenv('qdrant_api')
from langchain_openai import OpenAIEmbeddings




logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter(
    '%(asctime)s:%(name)s:%(levelname)s:%(message)s:%(funcName)s')
file_handler = logging.FileHandler('arabic_utils.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)





class IncomingFileProcessor():
    def __init__(self, chunk_size=750) -> None:
        self.chunk_size = chunk_size


    
    def get_pdf_splits(self, pdf_file: str, filename: str):
        try:
            loader = PyMuPDFLoader(pdf_file)
            pages = loader.load()
            logger.info("Succesfully loaded the pdf file")
            # textsplit = RecursiveCharacterTextSplitter(
            #     chunk_size=self.chunk_size, chunk_overlap=15, length_function=len)
            textsplit = RecursiveCharacterTextSplitter(
                separators=["\n\n",".","\n"],
                chunk_size=self.chunk_size, chunk_overlap=15, 
                length_function=len)
            doc_list = []
            for pg in pages:
                pg_splits = textsplit.split_text(pg.page_content)
                for page_sub_split in pg_splits:
                    metadata = {"source": filename}
                    doc_string = Document(page_content=page_sub_split, metadata=metadata)
                    doc_list.append(doc_string)
            logger.info("Succesfully split the pdf file")
            return doc_list
            # return pages
        except Exception as e:
            logger.critical(f"Error in Loading pdf file: {str(e)}")
            raise Exception(str(e))


def create_vector_db_for_files(files_info):
    res=[]
    try:
        for f_info in files_info:
            path=f_info["file_path"] 
            filename=f_info["filename"]
            file_extension=filename.split(".")[-1] 
            all_texts = load_data(path, file_extension, filename)
            assert len(all_texts)!=0
            embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
            collection_name="testing_arabic"
            create_new_vectorstore_qdrant(all_texts,embeddings,collection_name,qdrant_url,qdrant_api_key)
            logger.info('VectorDB stored successfully on filesystem')
            res.append({
                "filename" : filename,
                "collection_name":collection_name,
            })
            # print(res)
    except AssertionError as error:
        logger.critical(f'File doesnot contain any text or maybe corrupted {error}')
        # return JSONResponse(content = {"error": "File doesnot contain any text or maybe corrupted"},status_code=status.HTTP_404_NOT_FOUND)
    return res


file_processor = IncomingFileProcessor(chunk_size=1000)
def load_data(file_path, file_extension, file_name):
    if file_extension.lower() == "pdf":
        logger.info("enter in pdf file loader")
        texts = file_processor.get_pdf_splits(str(Path(__file__).parent.joinpath( file_path)), file_name)
        os.remove(file_path)
        logger.info("Successfully remove the pdf file")
        return texts
    

def create_new_vectorstore_qdrant(doc_list, embed_fn, COLLECTION_NAME,qdrant_url,qdrant_api_key):
    try:
        qdrant = Qdrant.from_documents(
            documents = doc_list,
            embedding = embed_fn,
            url=qdrant_url,
            prefer_grpc=False,
            api_key=qdrant_api_key,
            collection_name=COLLECTION_NAME,
        )
        logger.info("Successfully created the vectordb")
        return qdrant
    except Exception as ex:
        logger.critical("Vectordb Failed:"+str(ex))
        raise Exception({"Error": str(ex)})
    
def load_local_vectordb_using_qdrant(vectordb_folder_path, embed_fn):
    qdrant_client = QdrantClient(
        url=qdrant_url, 
        # prefer_grpc=True,
        api_key=qdrant_api_key,

    )
    qdrant_store= Qdrant(qdrant_client, vectordb_folder_path, embed_fn)
    return qdrant_store  
def arabic_qa(query, vectorstore):
    try:
        num_chunks= 6
        retriever= vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": num_chunks})
        template = """Answer the question in your own words as truthfully as possible from the following pieces of context:
        {context}

        Question: {question}
        """
        prompt = ChatPromptTemplate.from_template(template)
        setup_and_retrieval = RunnableParallel(
            {"context": retriever, "question": RunnablePassthrough()}
        )
        model = ChatOpenAI(model = "gpt-3.5-turbo-16k", openai_api_key = os.getenv("OPENAI_API_KEY"), temperature=0.3)
        output_parser= StrOutputParser()
        # chain = setup_and_retrieval | prompt | model | output_parser
        context= setup_and_retrieval.invoke(query)
        prompt_answer= prompt.invoke(context)
        model_answer= model.invoke(prompt_answer)
        response= output_parser.invoke(model_answer)
        return response

    except Exception as e:
        raise Exception('OpenAI key Error') 