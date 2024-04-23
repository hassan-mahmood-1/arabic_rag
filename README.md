# arabic_rag

**Overview**:

This repository contains a system for processing user queries, and generating responses based on contextual analysis of the PDF content. The system utilizes advanced natural language processing (NLP) and vector embedding techniques provided by OpenAI to enhance search and retrieval functionalities.

**Functionality:**

**1-Data Ingestion:** PDF data provided by clients is ingested into the Qdrant Vector Database for efficient storage and retrieval.

**2-Embedding Data:** Upon startup, the application loads pre-existing vector embeddings of PDF data using the OpenAI embedding model, facilitating quick access during search operations.

**3-User Query:** Users can submit queries through the frontend interface, initiating the search process.

**4-Embedding Query:**The OpenAI embedding model processes the user's query text, generating an embedding representation to capture semantic similarities.

**5-Search in Vector Database:** Embedded queries are compared with chunks of text stored in the vector database using cosine similarity, identifying relevant matches.

**6-Retrieve Chunks:** Text chunks with high cosine similarity scores to the query are retrieved from the vector database, forming the basis for further analysis.

**7-Prompt Generation:** Retrieved text chunks and the original query are passed to the OpenAI language model (GPT-3.5) to generate a contextual prompt, providing additional context for answer generation.

**8-Answer Generation:** The contextual prompt and retrieved text chunks serve as input to the GPT-3.5 model, which generates a response based on the provided context and query.

**9-Return Response:** The generated answer is sent back to the frontend for display to the user, completing the query-response cycle.

**Usage**

1-Clone this repository to your local machine.

2-Install necessary dependencies listed in requirements.txt.

3-Start the application and access the frontend interface to submit queries and retrieve responses.




