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
  title?: string;
  sourceName?: string;
  sourceNumber?: string;
  similarity?: number;
}

export interface RetrievedChunk {
  id: string;
  text: string;
  title?: string;
  sourceName?: string;
  sourceNumber?: string;
  score: number;
}

const normalizeText = (value: string): string =>
  value.replace(/\s+/g, " ").trim();

const parseCsvEnv = (value?: string): string[] =>
  (value ?? "")
    .split(",")
    .map((item) => item.trim())
    .filter(Boolean);

const unique = (values: string[]): string[] => Array.from(new Set(values));

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
  const normalizedQuery = normalizeText(userQuery);
  const semanticConfigurationName =
    process.env.AZURE_SEARCH_SEMANTIC_CONFIGURATION_NAME;
  const titleField = process.env.AZURE_SEARCH_TITLE_FIELD || "heading";
  const idField = process.env.AZURE_SEARCH_ID_FIELD || "chunk_id";
  const contentFields = unique([
    ...parseCsvEnv(process.env.AZURE_SEARCH_CONTENT_FIELDS),
    process.env.AZURE_SEARCH_CONTENT_FIELD || "content",
    "objective",
    "scope",
  ]);
  const keywordFields = unique([
    ...parseCsvEnv(process.env.AZURE_SEARCH_KEYWORD_FIELDS),
    "document_name",
    "document_number",
  ]);
  const vectorFields = unique([
    ...parseCsvEnv(process.env.AZURE_SEARCH_VECTOR_FIELDS),
    ...(process.env.AZURE_SEARCH_VECTOR_FIELD ? [process.env.AZURE_SEARCH_VECTOR_FIELD] : []),
  ]);

  const hasTextQuery = normalizedQuery.length > 0;
  const hasVectorQuery = vectorFields.length > 0;

  const searchParameters: any = {
    top: 8,
    searchMode: "all",
    searchFields: unique([titleField, ...contentFields, ...keywordFields]),
    select: unique([idField, titleField, ...contentFields, ...keywordFields]),
  };

  // Hybrid text side: semantic when configured, otherwise lexical simple query.
  // This combines with vectorSearchOptions below when vector fields are configured.
  if (hasTextQuery && semanticConfigurationName) {
    searchParameters.queryType = "semantic";
    searchParameters.semanticSearchOptions = {
      configurationName: semanticConfigurationName,
    };
  } else if (hasTextQuery) {
    searchParameters.queryType = "simple";
  }

  // Conditionally add vectorSearchOptions
  if (hasVectorQuery) {
    const userQueryEmbedded = await generateEmbedding(userQuery);
    searchParameters.vectorSearchOptions = {
      queries: vectorFields.map((fieldName) => ({
          kind: "vector",
          fields: [fieldName],
          kNearestNeighborsCount: semanticConfigurationName ? 50 : 8,
          vector: userQueryEmbedded,
        })),
    };
  }

  const searchText = hasTextQuery ? normalizedQuery : "*";
  const searchResults = await searchClient.search(searchText, searchParameters);

  const similarDocs: SearchDocument[] = [];
  for await (const result of searchResults.results) {
    const document = result.document as Record<string, unknown>;
    const resolvedId = document[idField];
    const title = typeof document[titleField] === "string" ? document[titleField] : undefined;
    const sourceName =
      typeof document.document_name === "string" ? document.document_name : undefined;
    const sourceNumber =
      typeof document.document_number === "string" ? document.document_number : undefined;

    const contentParts = contentFields
      .map((fieldName) => {
        const value = document[fieldName];
        if (typeof value === "string" && value.trim().length > 0) {
          return `${fieldName}: ${value}`;
        }
        return "";
      })
      .filter(Boolean);

    const textField = contentParts.join("\n");

    if (!textField) {
      continue;
    }

    const generatedHash = createHash("sha256")
      .update(`${sourceName ?? ""}|${sourceNumber ?? ""}|${textField}`)
      .digest("base64")
      .substring(0, 8);
    const resultId =
      typeof resolvedId === "string" && resolvedId.trim().length > 0
        ? resolvedId
        : generatedHash;
    const rankingScore =
      typeof result.rerankerScore === "number" ? result.rerankerScore : result.score;

    similarDocs.push({
      text: textField,
      id: resultId,
      title,
      sourceName,
      sourceNumber,
      similarity: rankingScore,
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
          title: document.title,
          sourceName: document.sourceName,
          sourceNumber: document.sourceNumber,
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
