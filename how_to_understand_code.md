1) read  app/core/config: use for load .env, this takes .env as input and return a settings object containing all the details(variables) of env file

2) read app/core/llm_provider: use to select llm model and embedding mode. takes settings object as input and return selected model as output

3) ingestion.py: 
(i) chunk_document function convert text into chunks.
(ii) extract_triples_for_chunk. convert chunks into triples(subject,predicate,object)

4) read app/core/vector_store: use to help generate embedding of chunks, in this three functions are there 
(i)  add_chunks: add chunks in chroma db. take chunks as input and store it into db
(ii) similarity_search: make similarity search on db
(iii)

5) read app/schemas/models.py : for this file there is not serial no we have to read this files to understand each and every object's structure like in chunk how many keys are there, for triples how many keys are there and so on...

6) read app/core/graph_store: 