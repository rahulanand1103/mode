from langchain.text_splitter import RecursiveCharacterTextSplitter


class Chunker:
    def __init__(self, chunk_size=800):
        self.chunk_size = chunk_size

    def read_file(self):
        with open("sample.txt") as f:
            sample = f.read()

        return sample

    def combine_chunk(self, chunks):
        combine_chunks = []

        for index, chunk in enumerate(chunks):
            if index > 0 and len(chunk.split(" ")) < 10:
                # Combine with the last chunk
                combine_chunks[-1] += " " + chunk
            elif index == 0 and len(chunk.split(" ")) < 10:
                # Combine with the next chunk
                combine_chunks.append(chunk + " " + chunks[index + 1])
            else:
                # Append to combine_chunks
                combine_chunks.append(chunk)

        return combine_chunks

    def chunks(self, text: str):
        text = text.replace("\\n\\n", "\n\n").replace("\\n", "\n")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=40,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )
        texts = text_splitter.split_text(text)
        return self.combine_chunk(texts)
