"use client";

import { useId, useState } from "react";

export function HelpHint({ label, children }: { label: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return <span className="help-hint">
    <button type="button" aria-label={`${label} için ipucu`} aria-expanded={open} aria-controls={id}
      onClick={(e) => { e.preventDefault(); setOpen(!open); }} onKeyDown={(e) => { if (e.key === "Escape") setOpen(false); }}
      className="hint-button">?</button>
    {open && <span id={id} role="note" className="hint-content">{children}</span>}
  </span>;
}
