from langchain.document_loaders import PDFPlumberLoader

# Path to your PDF file
pdf_path = '/Users/kunaldere/Desktop/genAIOneshot/medibot/data/The_GALE_ENCYCLOPEDIA_of_MEDICINE_SECOND.pdf'

# Initialize the loader
loader = PDFPlumberLoader(pdf_path)
# Load and split the document into pages (as LangChain documents)
documents = loader.load()

# Displaying content of the first page
print(documents[0].page_content)
