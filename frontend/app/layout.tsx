import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "RAG Pipeline Demo",
  description: "Compare 3 chunking strategies for retrieval quality",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-gray-950 text-gray-100 min-h-screen">{children}</body>
    </html>
  );
}
