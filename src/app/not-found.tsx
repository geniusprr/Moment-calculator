import Link from 'next/link'

export default function NotFound() {
    return (
        <div className="flex min-h-screen flex-col items-center justify-center bg-slate-950 text-slate-100">
            <h2 className="text-2xl font-bold">Sayfa Bulunamadı</h2>
            <p className="mb-4 text-slate-400">Aradığınız kaynak mevcut değil.</p>
            <Link
                href="/"
                className="rounded-lg bg-cyan-600 px-4 py-2 text-sm font-semibold text-white transition hover:bg-cyan-500"
            >
                Ana Sayfaya Dön
            </Link>
        </div>
    )
}
