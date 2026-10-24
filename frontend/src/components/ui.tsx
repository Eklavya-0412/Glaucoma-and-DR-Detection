import { createContext, useContext, useEffect, useState, type ReactNode } from 'react'
import { X, Maximize2 } from 'lucide-react'
import { fmt } from '../api'

export function Card({ title, subtitle, actions, children, className = '' }: {
  title?: ReactNode; subtitle?: ReactNode; actions?: ReactNode; children: ReactNode; className?: string
}) {
  return (
    <section className={`rounded-xl border border-line bg-surface ${className}`}>
      {(title || actions) && (
        <header className="flex items-start justify-between gap-4 border-b border-line px-5 py-3.5">
          <div>
            {title && <h3 className="text-[15px] font-semibold tracking-tight">{title}</h3>}
            {subtitle && <p className="mt-0.5 text-[13px] text-ink-soft">{subtitle}</p>}
          </div>
          {actions}
        </header>
      )}
      <div className="p-5">{children}</div>
    </section>
  )
}

const tones = {
  neutral: 'bg-canvas text-ink-soft border-line',
  accent: 'bg-accent-soft text-accent border-accent/20',
  good: 'bg-good-soft text-good border-good/20',
  warn: 'bg-warn-soft text-warn border-warn/20',
  bad: 'bg-bad-soft text-bad border-bad/20',
}
export type Tone = keyof typeof tones

export function Badge({ tone = 'neutral', children }: { tone?: Tone; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center rounded-md border px-1.5 py-0.5 text-[11px] font-medium ${tones[tone]}`}>
      {children}
    </span>
  )
}

export function statusTone(s: string | undefined | null): Tone {
  if (!s) return 'neutral'
  if (s === 'ok' || s === 'available' || s === 'segmented' || s === 'completed') return 'good'
  if (s.startsWith('error') || s === 'failed') return 'bad'
  if (s.startsWith('partial') || s.startsWith('warning') || s === 'unavailable') return 'warn'
  return 'neutral'
}

export function KV({ data, exclude = [] }: { data: Record<string, unknown>; exclude?: string[] }) {
  const entries = Object.entries(data).filter(([k]) => !exclude.includes(k))
  if (!entries.length) return null
  return (
    <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-[12px]">
      {entries.map(([k, v]) => (
        <div key={k} className="contents">
          <dt className="text-ink-faint">{k.replace(/_/g, ' ')}</dt>
          <dd className="num break-words font-mono text-[11.5px] text-ink">{fmt(v)}</dd>
        </div>
      ))}
    </dl>
  )
}

export function Table({ columns, rows, dense = false }: {
  columns: { key: string; label: ReactNode; align?: 'left' | 'right'; // eslint-disable-next-line @typescript-eslint/no-explicit-any
    render?: (row: any) => ReactNode }[]
  rows: Record<string, unknown>[]
  dense?: boolean
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-[11.5px] uppercase tracking-wide text-ink-faint">
            {columns.map((c) => (
              <th key={c.key} className={`px-2.5 py-2 font-medium ${c.align === 'right' ? 'text-right' : ''}`}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className="border-b border-line/70 last:border-0 hover:bg-canvas/60">
              {columns.map((c) => (
                <td key={c.key} className={`px-2.5 ${dense ? 'py-1' : 'py-1.5'} align-top ${c.align === 'right' ? 'num text-right font-mono text-[12.5px]' : ''}`}>
                  {c.render ? c.render(r) : fmt(r[c.key])}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-lg border border-dashed border-line px-4 py-6 text-center text-[13px] text-ink-soft">{children}</div>
}

// ------------------------------------------------------------------ lightbox
type LightboxItem = { src: string; title: string; caption?: string }
const LightboxCtx = createContext<(item: LightboxItem) => void>(() => {})
export const useLightbox = () => useContext(LightboxCtx)

export function LightboxProvider({ children }: { children: ReactNode }) {
  const [item, setItem] = useState<LightboxItem | null>(null)
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && setItem(null)
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])
  return (
    <LightboxCtx.Provider value={setItem}>
      {children}
      {item && (
        <div className="fixed inset-0 z-50 flex flex-col bg-[#0b0f17]/95 p-4 sm:p-8" onClick={() => setItem(null)}>
          <div className="mb-3 flex items-start justify-between gap-4 text-white">
            <div>
              <div className="text-[15px] font-semibold">{item.title}</div>
              {item.caption && <div className="mt-0.5 max-w-3xl text-[13px] text-white/70">{item.caption}</div>}
            </div>
            <button className="rounded-md p-1.5 hover:bg-white/10" aria-label="Close"><X size={20} /></button>
          </div>
          <div className="flex min-h-0 flex-1 items-center justify-center">
            <img src={item.src} alt={item.title} className="max-h-full max-w-full object-contain [image-rendering:auto]"
              onClick={(e) => e.stopPropagation()} />
          </div>
        </div>
      )}
    </LightboxCtx.Provider>
  )
}

export function ZoomImage({ src, title, caption, className = '', light = false }: { src: string; title: string; caption?: string; className?: string; light?: boolean }) {
  const open = useLightbox()
  return (
    <button type="button" onClick={() => open({ src, title, caption })}
      className={`group relative block w-full overflow-hidden rounded-lg ${light ? 'border border-line bg-white' : 'checker'} ${className}`} title="Click to enlarge">
      <img src={src} alt={title} loading="lazy" className="mx-auto max-h-[340px] w-full object-contain" />
      <span className="absolute right-2 top-2 rounded-md bg-black/55 p-1 text-white opacity-0 transition group-hover:opacity-100">
        <Maximize2 size={14} />
      </span>
    </button>
  )
}
