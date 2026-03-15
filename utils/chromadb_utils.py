import chromadb
import uuid

persist_directory = "./chromadb"

client = chromadb.PersistentClient(path=persist_directory)
collection = client.get_or_create_collection(name="emb_faces", metadata={"hnsw:space": "cosine"})

class ChromaDB():
    
    def insert_emb(self, emb, path_file, metadado=[{'fonte': 'cadastro_by_streamlit'}]):
        try:
            id_file = str(uuid.uuid4())
            collection.add(
                embeddings=[emb],
                documents=[path_file],
                metadatas=metadado,
                ids=[id_file]
            )
        except Exception as ex:
            raise InsertEmbeddingException('Erro ao tentar inserir embedding ao banco vetorial.')

    def find_emb(self, emb):
        resultado = None
        try:
            resultado = collection.query(
                query_embeddings=[emb],
                n_results=1
            )
        except Exception as ex:
            raise FindEmbeddingException('Erro ao tentar encontrar similaridade de embedding.')
        return resultado
    
    def get_total_emb(self):
        count = collection.count()
        return count
    
class InsertEmbeddingException(Exception):
    def __init__(self, mensagem):
        super().__init__(mensagem)

class FindEmbeddingException(Exception):
    def __init__(self, mensagem):
        super().__init__(mensagem)