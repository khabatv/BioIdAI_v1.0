import express from "express";
import cors from "cors";
import { createServer as createViteServer } from "vite";
import OpenAI from "openai";
import Anthropic from "@anthropic-ai/sdk";
import { Groq } from "groq-sdk";

async function startServer() {
  const app = express();
  const PORT = process.env.PORT || 3000;

  app.use(cors());
  app.use(express.json());

  // --- AI Provider Endpoints ---

  app.post("/api/ai/proxy", async (req, res) => {
    const { provider, apiKey, prompt, responseSchema } = req.body;

    try {
      if (provider === "OpenAI") {
        const openai = new OpenAI({ apiKey });
        const response = await openai.chat.completions.create({
          model: "gpt-4o",
          messages: [{ role: "user", content: prompt }],
          response_format: { type: "json_object" },
        });
        return res.json(JSON.parse(response.choices[0].message.content || "{}"));
      }

      if (provider === "Anthropic") {
        const anthropic = new Anthropic({ apiKey });
        const response = await anthropic.messages.create({
          model: "claude-3-5-sonnet-20240620",
          max_tokens: 4096,
          messages: [{ role: "user", content: prompt + "\n\nRespond ONLY with a valid JSON object." }],
        });
        const content = response.content[0].type === 'text' ? response.content[0].text : "";
        return res.json(JSON.parse(content));
      }

      if (provider === "Groq") {
        const groq = new Groq({ apiKey });
        const response = await groq.chat.completions.create({
          model: "llama-3.1-70b-versatile",
          messages: [{ role: "user", content: prompt }],
          response_format: { type: "json_object" },
        });
        return res.json(JSON.parse(response.choices[0].message.content || "{}"));
      }

      // Add other providers as needed...
      
      res.status(400).json({ error: `Provider ${provider} not supported via proxy.` });
    } catch (error: any) {
      console.error(`Error with ${provider}:`, error);
      res.status(500).json({ error: error.message || "Internal Server Error" });
    }
  });

  // --- Vite Middleware for Development ---
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    app.use(express.static("dist"));
    app.get("*", (req, res) => {
      res.sendFile("dist/index.html", { root: "." });
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`Server running on http://localhost:${PORT}`);
  });
}

startServer();
