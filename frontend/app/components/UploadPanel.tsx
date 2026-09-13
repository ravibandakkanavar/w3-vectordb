"use client";
import { useState, useRef } from "react";

interface UploadPanelProps {
  onIngestComplete: () => void;
}

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export default function UploadPanel({ onIngestComplete }: UploadPanelProps) {
  const [file, setFile] = useState<File | null>(null);
  const [phase, setPhase] = useState<"idle" | "uploading" | "polling" | "done" | "error">("idle");
  const [message, setMessage] = useState("");
  const [detail, setDetail] = useState<Record<string, number> | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  async function handleUpload() {
    if (!file) return;
    setPhase("uploading");
    setMessage("Uploading PDF...");
    setDetail(null);

    const form = new FormData();
    form.append("file", file);

    try {
      const res = await fetch(`${API_BASE}/api/ingest`, { method: "POST", body: form });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || "Upload failed");
      }
      setPhase("polling");
      setMessage("Processing...");
      pollProgress();
    } catch (e: unknown) {
      setPhase("error");
      const errorText = e instanceof Error ? e.message : "Upload failed";
      setMessage(`${errorText}. Check that the backend is running and NEXT_PUBLIC_API_URL is correct.`);
    }
  }

  function pollProgress() {
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API_BASE}/api/ingest/progress`);
        if (!res.ok) {
          throw new Error("Backend unavailable");
        }
        const data = await res.json();
        setMessage(data.message || "Processing...");
        if (data.status === "done") {
          clearInterval(interval);
          setPhase("done");
          setDetail(data.detail);
          onIngestComplete();
        } else if (data.status === "error") {
          clearInterval(interval);
          setPhase("error");
          setMessage(data.message || "Ingestion failed.");
        }
      } catch {
        clearInterval(interval);
        setPhase("error");
        setMessage("Lost connection to backend. Check that the Python API is running and the API URL is configured correctly.");
      }
    }, 2000);
  }

  return (
    <div className="bg-gray-900 border border-gray-700 rounded-xl p-6 space-y-4">
      <h2 className="text-lg font-semibold text-white">📄 Upload PDF</h2>
      <p className="text-sm text-gray-400">Upload a PDF to extract, chunk (3 strategies), embed with Gemini, and store in Supabase.</p>

      <div
        className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center cursor-pointer hover:border-indigo-500 transition-colors"
        onClick={() => inputRef.current?.click()}
      >
        {file ? (
          <p className="text-indigo-300 font-medium">{file.name}</p>
        ) : (
          <p className="text-gray-500">Click to select a PDF file</p>
        )}
        <input
          ref={inputRef}
          type="file"
          accept=".pdf"
          className="hidden"
          onChange={(e) => { setFile(e.target.files?.[0] || null); setPhase("idle"); setMessage(""); }}
        />
      </div>

      <button
        onClick={handleUpload}
        disabled={!file || phase === "uploading" || phase === "polling"}
        className="w-full py-2 px-4 bg-indigo-600 hover:bg-indigo-500 disabled:bg-gray-700 disabled:cursor-not-allowed rounded-lg font-medium text-white transition-colors"
      >
        {phase === "uploading" || phase === "polling" ? "Processing..." : "Ingest PDF"}
      </button>

      {message && (
        <div className={`text-sm px-3 py-2 rounded-lg ${phase === "error" ? "bg-red-950 text-red-300" : phase === "done" ? "bg-emerald-950 text-emerald-300" : "bg-gray-800 text-gray-300"}`}>
          {phase === "polling" && <span className="animate-pulse">⏳ </span>}
          {phase === "done" && "✅ "}
          {phase === "error" && "❌ "}
          {message}
          {detail && (
            <span className="ml-2 text-gray-400">
              (fixed: {detail.fixed}, structural: {detail.structural}, semantic: {detail.semantic})
            </span>
          )}
        </div>
      )}
    </div>
  );
}
