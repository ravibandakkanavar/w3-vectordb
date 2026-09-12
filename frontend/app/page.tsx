"use client";
import { useState, useEffect, useCallback } from "react";
import StatusBar from "./components/StatusBar";
import UploadPanel from "./components/UploadPanel";
import QueryPanel, { type StrategyResult } from "./components/QueryPanel";
import ResultsTable from "./components/ResultsTable";

interface DBStatus {
  status: string;
  total_chunks: number;
  by_strategy: Record<string, number>;
}

export default function Home() {
  const [dbStatus, setDbStatus] = useState<DBStatus | null>(null);
  const [statusLoading, setStatusLoading] = useState(true);
  const [queryResults, setQueryResults] = useState<StrategyResult[]>([]);
  const [isQuerying, setIsQuerying] = useState(false);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await fetch("http://localhost:8000/api/status");
      const data = await res.json();
      setDbStatus(data);
    } catch {
      setDbStatus(null);
    } finally {
      setStatusLoading(false);
    }
  }, []);

  useEffect(() => { fetchStatus(); }, [fetchStatus]);

  return (
    <main className="max-w-7xl mx-auto px-4 py-8 space-y-6">
      {/* Header */}
      <div className="space-y-1">
        <h1 className="text-3xl font-bold text-white">RAG Pipeline Demo</h1>
        <p className="text-gray-400">Compare <strong className="text-indigo-400">Fixed</strong>, <strong className="text-amber-400">Structural</strong>, and <strong className="text-emerald-400">Semantic</strong> chunking strategies side by side.</p>
      </div>

      {/* Status */}
      <StatusBar status={dbStatus} loading={statusLoading} />

      {/* Upload + Query side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <UploadPanel onIngestComplete={fetchStatus} />
        <QueryPanel
          onResults={setQueryResults}
          isLoading={isQuerying}
          setIsLoading={setIsQuerying}
        />
      </div>

      {/* Results */}
      <div>
        <h2 className="text-lg font-semibold text-white mb-3">📊 Retrieval Results</h2>
        <ResultsTable results={queryResults} isLoading={isQuerying} />
      </div>
    </main>
  );
}
