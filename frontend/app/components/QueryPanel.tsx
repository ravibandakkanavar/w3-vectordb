"use client";
import { useState } from "react";

const SAMPLE_QUERIES = [
  "What is the sensitivity and specificity of the IDx-DR system?",
  "What score did Med-PaLM 2 achieve on the USMLE?",
  "What does SHAP stand for and what is it used for?",
  "Explain the relationship between data quality and model performance.",
  "What evidence supports the Epic Sepsis Model underperforming in deployment?",
  "What are the steps involved in federated learning?",
  "What are all the limitations mentioned throughout the document?",
];

interface QueryPanelProps {
  onResults: (results: StrategyResult[]) => void;
  isLoading: boolean;
  setIsLoading: (v: boolean) => void;
}

export interface ChunkResult {
  chunk_id: string;
  content: string;
  source_page: number;
  score: number;
}

export interface StrategyResult {
  strategy: string;
  results: ChunkResult[];
}

export default function QueryPanel({ onResults, isLoading, setIsLoading }: QueryPanelProps) {
  const [query, setQuery] = useState("");
  const [error, setError] = useState("");

  async function handleQuery(q?: string) {
    const text = (q || query).trim();
    if (!text) return;
    setQuery(text);
    setIsLoading(true);
    setError("");

    try {
      const res = await fetch("http://localhost:8000/api/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: text }),
      });
      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Query failed");
      }
      const data: StrategyResult[] = await res.json();
      onResults(data);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : "Query failed");
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-4">
      <h2 className="text-lg font-semibold text-white">🔍 Query</h2>

      <div className="flex gap-2">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleQuery()}
          placeholder="Ask anything about your document..."
          className="flex-1 bg-gray-800 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-500 focus:outline-none focus:border-indigo-500"
        />
        <button
          onClick={() => handleQuery()}
          disabled={!query.trim() || isLoading}
          className="px-5 py-2 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium text-white transition-colors"
        >
          {isLoading ? "..." : "Search"}
        </button>
      </div>

      <div className="space-y-1">
        <p className="text-xs text-gray-500 uppercase tracking-wide">Sample queries</p>
        <div className="flex flex-wrap gap-2">
          {SAMPLE_QUERIES.map((q) => (
            <button
              key={q}
              onClick={() => handleQuery(q)}
              disabled={isLoading}
              className="text-xs px-3 py-1 bg-gray-800 hover:bg-gray-700 border border-gray-600 rounded-full text-gray-300 transition-colors disabled:opacity-50"
            >
              {q.length > 50 ? q.slice(0, 50) + "…" : q}
            </button>
          ))}
        </div>
      </div>

      {error && <p className="text-sm text-red-400">❌ {error}</p>}
    </div>
  );
}
