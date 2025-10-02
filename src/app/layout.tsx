import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";
import "katex/dist/katex.min.css";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "BeamTutor | Beam solver",
  description: "Interactive statics demo for reactions, shear and moment diagrams with teaching mode.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.className} bg-slate-950 text-slate-100 antialiased`}>
        {children}
      </body>
    </html>
  );
}
