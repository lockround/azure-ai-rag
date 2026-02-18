import { findRelevantChunks } from "@/lib/ai/search";
import { azure } from "@ai-sdk/azure";
import { Message, convertToCoreMessages, streamText } from "ai";

// Allow streaming responses up to 30 seconds
export const maxDuration = 30;

const getTextFromMessageContent = (content: unknown): string => {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .map((part) => {
        if (typeof part === "string") {
          return part;
        }

        if (
          typeof part === "object" &&
          part !== null &&
          "text" in part &&
          typeof (part as { text?: unknown }).text === "string"
        ) {
          return (part as { text: string }).text;
        }

        return "";
      })
      .join(" ")
      .trim();
  }

  return "";
};

const getLatestUserQuery = (messages: Message[]): string => {
  for (let index = messages.length - 1; index >= 0; index -= 1) {
    const message = messages[index];
    if (message.role === "user") {
      const content = getTextFromMessageContent(message.content);
      if (content) {
        return content;
      }
    }
  }

  return "";
};

const formatRetrievedContext = (
  chunks: Array<{ id: string; text: string }>,
  maxContextCharacters = 7000
): string => {
  if (chunks.length === 0) {
    return "No relevant context was retrieved from the knowledge base.";
  }

  const contextSections: string[] = [];
  let currentLength = 0;

  for (const chunk of chunks) {
    const section = `Source ID: ${chunk.id}\n${chunk.text}`;
    if (currentLength + section.length > maxContextCharacters) {
      break;
    }

    contextSections.push(section);
    currentLength += section.length;
  }

  return contextSections.join("\n\n---\n\n");
};

export async function POST(req: Request) {
  try {
    const body = (await req.json()) as { messages?: Message[] };
    const messages = Array.isArray(body.messages) ? body.messages : [];
    const latestUserQuery = getLatestUserQuery(messages);
    const relevantChunks = latestUserQuery
      ? await findRelevantChunks(latestUserQuery)
      : [];
    const retrievedContext = formatRetrievedContext(relevantChunks);

    const result = await streamText({
      model: azure(process.env.AZURE_DEPLOYMENT_NAME!),
      messages: convertToCoreMessages(messages),
      system: `You are a helpful assistant acting as the users' second brain.
      Use only the retrieved context and chat history to answer.
      If no relevant information is found in the retrieved context, respond exactly with "Sorry, I don't know."
      Keep responses short and concise. Answer in a single sentence where possible.
      Cite the sources using source ids at the end of the answer text, like 【234d987】, using the id of the source.
      If you cannot support an answer with the retrieved context, respond exactly with "Sorry, I don't know."

      Current user query:
      """${latestUserQuery || "No user query provided."}"""

      Retrieved context:
      """${retrievedContext}"""
    `,
    });

    return result.toDataStreamResponse();
  } catch (error: unknown) {
    console.error(error);
    return new Response(JSON.stringify({ error: "An unexpected error occurred. Please try again later." }), { status: 500 });
  }
}
