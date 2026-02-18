import { SearchClient, AzureKeyCredential } from "@azure/search-documents";
import { DefaultAzureCredential } from "@azure/identity";
import { embed } from "ai";
import { azure } from "@ai-sdk/azure";
import { createHash } from "crypto";

const embeddingModel = azure.textEmbeddingModel(process.env.AZURE_EMBEDDING_DEPLOYMENT_NAME!);
const endpoint = process.env.AZURE_SEARCH_ENDPOINT!;
const indexName = process.env.AZURE_SEARCH_INDEX_NAME!;
const credential = process.env.AZURE_SEARCH_KEY
  ? new AzureKeyCredential(process.env.AZURE_SEARCH_KEY)
  : new DefaultAzureCredential();
const USER_AGENT_PREFIX = "vercel-nextjs-azs";

const searchClient = new SearchClient(
  endpoint,
  indexName,
  credential,
  {
    userAgentOptions: {
      userAgentPrefix: USER_AGENT_PREFIX,
    },
  }
);

export interface SearchDocument {
  id: string;
  text: string;
  similarity?: number;
}

export interface RetrievedChunk {
  id: string;
  text: string;
  score: number;
}

const normalizeText = (value: string): string =>
  value.replace(/\s+/g, " ").trim();

const tokenize = (value: string): string[] =>
  normalizeText(value)
    .toLowerCase()
    .split(/[^a-z0-9]+/g)
    .filter((token) => token.length > 2);

const splitIntoChunks = (
  value: string,
  maxChunkLength = 700,
  overlap = 120
): string[] => {
  const normalizedValue = normalizeText(value);
  if (!normalizedValue) {
    return [];
  }

  if (normalizedValue.length <= maxChunkLength) {
    return [normalizedValue];
  }

  const chunks: string[] = [];
  let start = 0;

  while (start < normalizedValue.length) {
    let end = Math.min(start + maxChunkLength, normalizedValue.length);

    if (end < normalizedValue.length) {
      const lastWhitespace = normalizedValue.lastIndexOf(" ", end);
      if (lastWhitespace > start + Math.floor(maxChunkLength * 0.6)) {
        end = lastWhitespace;
      }
    }

    const chunk = normalizedValue.slice(start, end).trim();
    if (chunk) {
      chunks.push(chunk);
    }

    if (end >= normalizedValue.length) {
      break;
    }

    start = Math.max(end - overlap, start + 1);
  }

  return chunks;
};

const rankChunk = (
  chunk: string,
  queryTokens: Set<string>,
  similarityScore: number
): number => {
  if (queryTokens.size === 0) {
    return similarityScore;
  }

  const chunkTokens = new Set(tokenize(chunk));
  let overlapCount = 0;
  queryTokens.forEach((token) => {
    if (chunkTokens.has(token)) {
      overlapCount += 1;
    }
  });

  const lexicalScore = overlapCount / queryTokens.size;
  return similarityScore + lexicalScore;
};

export const generateEmbedding = async (value: string): Promise<number[]> => {
  const input = value.replaceAll("\n", " ");
  const { embedding } = await embed({
    model: embeddingModel,
    value: input,
  });
  return embedding;
};

export const findRelevantContent = async (
  userQuery: string
): Promise<SearchDocument[]> => {
  const searchParameters: any = {
    top: 5,
    queryType: "simple",
  };

  // Conditionally add semanticSearchOptions
  if (process.env.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME) {
    searchParameters.queryType = "semantic";
    searchParameters.semanticSearchOptions = {
      configurationName: process.env.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME,
    };
  }

  // Conditionally add vectorSearchOptions
  if (process.env.AZURE_SEARCH_VECTOR_FIELD) {
    const userQueryEmbedded = await generateEmbedding(userQuery);
    searchParameters.vectorSearchOptions = {
      queries: [
        {
          kind: "vector",
          fields: [process.env.AZURE_SEARCH_VECTOR_FIELD], // Use the vector field from env vars
          kNearestNeighborsCount: process.env.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME ? 50 : 5,
          vector: userQueryEmbedded,
        },
      ],
    };
  }

  const searchResults = await searchClient.search(userQuery, searchParameters);

  const similarDocs: SearchDocument[] = [];
  const contentColumn = process.env.AZURE_SEARCH_CONTENT_FIELD!;
  for await (const result of searchResults.results) {
    const document = result.document as Record<string, unknown>;
    const rawTextField = Object.prototype.hasOwnProperty.call(document, contentColumn)
      ? document[contentColumn]
      : document;

    const textField =
      typeof rawTextField === "string"
        ? rawTextField
        : JSON.stringify(rawTextField);

    if (!textField) {
      continue;
    }

    const hash = createHash("sha256")
      .update(textField)
      .digest("base64")
      .substring(0, 8);

    similarDocs.push({
      text: textField,
      id: hash,
      similarity: result.score,
    });
  }

  return similarDocs;
};

export const extractRelevantChunks = (
  userQuery: string,
  documents: SearchDocument[],
  maxChunks = 8
): RetrievedChunk[] => {
  const queryTokens = new Set(tokenize(userQuery));
  const uniqueChunks = new Map<string, RetrievedChunk>();

  documents.forEach((document) => {
    const chunks = splitIntoChunks(document.text);
    chunks.forEach((chunkText, index) => {
      const normalizedChunk = normalizeText(chunkText).toLowerCase();
      if (!normalizedChunk) {
        return;
      }

      const score = rankChunk(
        chunkText,
        queryTokens,
        typeof document.similarity === "number" ? document.similarity : 0
      );

      const existingChunk = uniqueChunks.get(normalizedChunk);
      if (!existingChunk || score > existingChunk.score) {
        uniqueChunks.set(normalizedChunk, {
          id: `${document.id}-${index + 1}`,
          text: chunkText,
          score,
        });
      }
    });
  });

  return Array.from(uniqueChunks.values())
    .sort((a, b) => b.score - a.score)
    .slice(0, maxChunks);
};

export const findRelevantChunks = async (
  userQuery: string,
  maxChunks = 8
): Promise<RetrievedChunk[]> => {
  const documents = await findRelevantContent(userQuery);
  return extractRelevantChunks(userQuery, documents, maxChunks);
};
