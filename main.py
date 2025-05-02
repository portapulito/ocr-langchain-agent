import os
import re
import json
from dotenv import load_dotenv
from typing import List
from pydantic import BaseModel, Field

from langchain_core.documents import Document
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.tools import tool
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser


load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY non trovato nel file .env")




llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=openai_api_key,
    temperature=0.7,
    max_tokens=2048,
    request_timeout=30,
)




def pulisci_testo_ocr(testo: str) -> str:
    testo = testo.replace('\x0c', '')
    testo = re.sub(r'[^\x00-\x7F]+', ' ', testo)
    testo = re.sub(r'\s+', ' ', testo).strip()
    return testo




@tool
def estrai_testo_ocr_pulito(percorso_immagine: str) -> str:
    """
    Estrae il testo OCR da un'immagine e lo restituisce come stringa pulita.
    """
    if not os.path.exists(percorso_immagine):
        raise FileNotFoundError(f"File non trovato: {percorso_immagine}")

    loader = AmazonTextractPDFLoader(percorso_immagine)
    documents = loader.load()
    if not documents:
        return ""

    testo_raw = documents[0].page_content
    return pulisci_testo_ocr(testo_raw)




class MessaggioSingolo(BaseModel):
    mittente: str = Field(description="Chi ha inviato il messaggio")
    destinatario: str = Field(description="Chi ha ricevuto il messaggio")
    messaggio: str = Field(description="Contenuto del messaggio")


class ConversazioneOCR(BaseModel):
    messaggi: List[MessaggioSingolo]

parser = PydanticOutputParser(pydantic_object=ConversazioneOCR)

prompt_parse = PromptTemplate(
    template="""
Sei un assistente che analizza conversazioni WhatsApp OCRizzate.
Estrai TUTTI i messaggi (mittente, destinatario, contenuto) da questo testo:

{format_instructions}

Testo OCR:
{input}
""",
    input_variables=["input"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)


chain_parser = prompt_parse | llm | parser



if __name__ == "__main__":
    percorso_file = r"immagini/foto_prova2.jpg"

    print(f"📥 Analizzando immagine: {percorso_file}")
    testo_ocr = estrai_testo_ocr_pulito.invoke(percorso_file)

    if not testo_ocr.strip():
        raise ValueError("nessun testo OCR trovato. Controlla l'immagine o la qualità del testo.")

    print("\nTesto OCR estratto:")
    print(testo_ocr)

    risultato = chain_parser.invoke({"input": testo_ocr})

    print("\nRisultato strutturato:")
    for messaggio in risultato.messaggi:
        print("\n---")
        print(f"Mittente: {messaggio.mittente}")
        print(f"Destinatario: {messaggio.destinatario}")
        print(f"Messaggio: {messaggio.messaggio}")

  


    output_path = "output/messaggi_estratti.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump([m.dict() for m in risultato.messaggi], f, ensure_ascii=False, indent=2)

    print(f"\nMessaggi salvati in {output_path}")



