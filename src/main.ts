import 'dotenv/config'
import { getEnv } from './utils/env.js'
import { OpenAI } from 'langchain/llms/openai'
import { Chroma } from 'langchain/vectorstores/chroma'
import { TextLoader } from 'langchain/document_loaders/fs/text'
import { OpenAIEmbeddings } from 'langchain/embeddings/openai'
import path from 'path'
import { ChromaClient } from 'chromadb'
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter'
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory'
import { RetrievalQAChain } from 'langchain/chains'

const { OPENAI_API_KEY } = getEnv()

const loader = new DirectoryLoader(path.resolve('assets'), {
  '.md': filepath => new TextLoader(filepath),
})

const documents = await loader.load()

const splitter = new RecursiveCharacterTextSplitter({ chunkSize: 1000, chunkOverlap: 100 })
const texts = await splitter.splitDocuments(documents)

const collectionName = 'specifications'

// Delete collection if it exists
const client = new ChromaClient()
await client.deleteCollection({ name: collectionName })

const embeddings = new OpenAIEmbeddings()
const chroma = await Chroma.fromDocuments(texts, embeddings, {
  collectionName,
})

const llm = new OpenAI({
  openAIApiKey: OPENAI_API_KEY,
  // modelName: 'gpt-3.5-turbo',
  modelName: 'gpt-4',
  temperature: 0.75,
  maxTokens: 1024,
})

const chain = RetrievalQAChain.fromLLM(llm, chroma.asRetriever(), {
  returnSourceDocuments: false,
})

const result = await chain.call({
  query: 'Compare iPhone 15 Pro and Pixel 8 Pro.',
})

console.dir(result, { depth: null })
