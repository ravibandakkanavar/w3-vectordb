"use client";
import type { StrategyResult } from "./QueryPanel";

const STRATEGY_COLORS: Record<string, string> = {
  fixed:      "border-indigo-500 bg-indigo-950",
  structural: "border-amber-500 bg-amber-950",
  semantic:   "border-emerald-500 bg-emerald-950",
};

const STRATEGY_BADGE: Record<string, string> = {
  fixed:      "bg-indigo-800 text-indigo-200",
  structural: "bg-amber-800 text-amber-200",
  semantic:   "bg-emerald-800 text-emerald-200",
};

interface ResultsTableProps {
  results: StrategyResult[];
  isLoading: boolean;
}

function ScoreBar({ score }: { score: number }) {
  const pct = Math.round(score * 100);
  const color = score >= 0.75 ? "bg-emerald-500" : score >= 0.60 ? "bg-amber-500" : "bg-red-500";
  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 bg-gray-800 rounded-full h-1.5">
        <div className={`${color} h-1.5 rounded-full`} style={{ width: `${pct}%` }} />
      </div>
      <span className="text-xs font-mono text-gray-300 w-10 text-right">{score.toFixed(3)}</span>
    </div>
  );
}

export default function ResultsTable({ results, isLoading }: ResultsTableProps) {
  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {["fixed", "structural", "semantic"].map((s) => (
          <div key={s} className={`border rounded-xl p-4 animate-pulse ${STRATEGY_COLORS[s]}`}>
            <div className="h-5 w-24 bg-gray-700 rounded mb-4" />
            {[1, 2, 3].map((i) => <div key={i} className="h-20 bg-gray-800 rounded mb-3" />)}
          </div>
        ))}
      </div>
    );
  }

  if (!results.length) {
    return (
      <div className="text-center py-16 text-gray-600">
        <p className="text-4xl mb-3">🔍</p>
        <p>Run a query to see results across all 3 strategies</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {results.map(({ strategy, results: chunks }) => (
          <div key={strategy} className={`border rounded-xl p-4 space-y-3 ${STRATEGY_COLORS[strategy]}`}>
            <div className="flex items-center justify-between">
              <span className={`text-xs font-semibold px-2 py-1 rounded-full uppercase tracking-wide ${STRATEGY_BADGE[strategy]}`}>
                {strategy}
              </span>
              <span className="text-xs text-gray-400">{chunks.length} results</span>
            </div>

            {chunks.map((chunk, i) => (
              <div key={chunk.chunk_id} className="bg-gray-900 rounded-lg p-3 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono text-gray-500">#{i + 1} {chunk.chunk_id}</span>
                  <span className="text-xs text-gray-500">p.{chunk.source_page}</span>
                </div>
                <ScoreBar score={chunk.score} />
                <p className="text-xs text-gray-300 leading-relaxed line-clamp-5">
                  {chunk.content}
                </p>
              </div>
            ))}
          </div>
        ))}
      </div>

      {/* Mini comparison bar */}
      <div className="bg-gray-900 border border-gray-700 rounded-xl p-4">
        <p className="text-xs text-gray-500 uppercase tracking-wide mb-3">Top-1 Score Comparison</p>
        <div className="space-y-2">
          {results.map(({ strategy, results: chunks }) => {
            const top = chunks[0];
            if (!top) return null;
            return (
              <div key={strategy} className="flex items-center gap-3">
                <span className={`text-xs font-semibold px-2 py-0.5 rounded-full w-24 text-center ${STRATEGY_BADGE[strategy]}`}>
                  {strategy}
                </span>
                <div className="flex-1">
                  <ScoreBar score={top.score} />
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
