"use client";

interface StatusBarProps {
  status: { status: string; total_chunks: number; by_strategy: Record<string, number> } | null;
  loading: boolean;
}

export default function StatusBar({ status, loading }: StatusBarProps) {
  if (loading) return <div className="text-sm text-gray-400 animate-pulse">Checking DB status...</div>;
  if (!status) return null;

  const isOk = status.status === "ok";
  return (
    <div className={`flex items-center gap-4 text-sm px-4 py-2 rounded-lg border ${isOk ? "border-emerald-700 bg-emerald-950 text-emerald-300" : "border-red-700 bg-red-950 text-red-300"}`}>
      <span className={`w-2 h-2 rounded-full ${isOk ? "bg-emerald-400" : "bg-red-400"}`} />
      <span>Supabase {isOk ? "connected" : "error"}</span>
      {isOk && (
        <>
          <span className="text-gray-500">|</span>
          <span>{status.total_chunks} chunks total</span>
          {Object.entries(status.by_strategy).map(([s, c]) => (
            <span key={s} className="text-gray-400">{s}: <strong className="text-white">{c}</strong></span>
          ))}
        </>
      )}
    </div>
  );
}
