import { AlertTriangle, Clock, Flag, Globe, Languages, Music, Music2, Search, Sparkles, Zap } from 'lucide-react'
import { useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const languageOptions = [
    { value: 'auto', label: 'Auto detect' },
    { value: 'hi', label: 'Hindi first' },
    { value: 'en', label: 'English only' },
    { value: 'both', label: 'Blend both' },
]

const sampleQueries = [
    { label: 'प्रेम कविता', text: 'प्रेम कविता' },
    { label: 'prem geet', text: 'prem geet' },
    { label: 'heartbreak love song', text: 'heartbreak love song' },
    { label: 'dosti ke geet', text: 'dosti ke geet' },
]

function ResultCard({ result }) {
    const languageLabel = result.language?.startsWith('en') ? 'English' : 'Hindi'

    // Format text: replace \n with actual line breaks and clean up escape sequences
    const formatText = (text) => {
        if (!text) return ''
        return text
            .replace(/\\n/g, '\n')  // Convert literal \n to actual newlines
            .replace(/\\r/g, '')     // Remove carriage returns
            .replace(/\\t/g, '  ')   // Convert tabs to spaces
            .replace(/\\/g, '')      // Remove remaining backslashes
            .trim()
    }

    const formattedText = formatText(result.text)
    const formattedHinglish = result.hinglish ? formatText(result.hinglish) : null

    return (
        <article className="result-card">
            <div className="result-meta">
                <span className={`pill pill-${result.language?.startsWith('en') ? 'en' : 'hi'}`}>{languageLabel}</span>
                {result.score !== null && result.score !== undefined && (
                    <span className="score">score: {result.score.toFixed(3)}</span>
                )}
            </div>
            <h3 className="result-title">
                {result.title ? result.title : result.poet ? `By ${result.poet}` : 'Lyric/Poem'}
            </h3>
            <pre className="result-text">{formattedText}</pre>
            {formattedHinglish && (
                <pre className="hinglish">{formattedHinglish}</pre>
            )}
            <div className="result-footer">
                {result.poet && <span>Poet: {result.poet}</span>}
                {result.period && <span>Period: {result.period}</span>}
            </div>
        </article>
    )
}

function App() {
    const [query, setQuery] = useState('')
    const [lang, setLang] = useState('auto')
    const [includeEnglish, setIncludeEnglish] = useState(true)
    const [topK, setTopK] = useState(5)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState('')
    const [results, setResults] = useState([])
    const [meta, setMeta] = useState(null)

    const apiBase = useMemo(() => API_BASE.replace(/\/$/, ''), [])

    const handleSubmit = async (e) => {
        e?.preventDefault()
        setError('')
        if (!query.trim()) {
            setError('Please enter something to search')
            return
        }

        setLoading(true)
        try {
            const response = await fetch(`${apiBase}/api/search`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query, top_k: Number(topK), lang, include_english: includeEnglish }),
            })

            if (!response.ok) {
                const detail = await response.json().catch(() => ({}))
                throw new Error(detail.detail || 'Search failed')
            }

            const data = await response.json()
            setResults(data.results || [])
            setMeta({ backend: data.backend, counts: data.counts })
        } catch (err) {
            setError(err.message || 'Something went wrong')
            setResults([])
        } finally {
            setLoading(false)
        }
    }

    const handleSampleClick = (text) => {
        setQuery(text)
        setTimeout(() => handleSubmit(), 0)
    }

    return (
        <div className="page">
            <header className="hero">
                <div className="hero-content">
                    <div className="hero-badge">
                        <Sparkles className="badge-icon" size={16} />
                        <span>AI-Powered Discovery</span>
                    </div>
                    <h1 className="hero-title">
                        Discover Songs & Poems
                        <span className="gradient-text"> Across Languages</span>
                    </h1>
                    <p className="hero-subtitle">
                        Search in Hindi, Hinglish, or English and find lyrically similar pieces powered by
                        multilingual embeddings and semantic understanding.
                    </p>
                    <div className="hero-features">
                        <div className="feature-pill">
                            <Flag className="feature-icon" size={16} />
                            <span>हिंदी</span>
                        </div>
                        <div className="feature-pill">
                            <Languages className="feature-icon" size={16} />
                            <span>Hinglish</span>
                        </div>
                        <div className="feature-pill">
                            <Globe className="feature-icon" size={16} />
                            <span>English</span>
                        </div>
                    </div>
                </div>
            </header>

            <div className="features-grid">
                <div className="feature-card">
                    <div className="feature-icon-large"><Music size={48} /></div>
                    <h3>Smart Search</h3>
                    <p>Find similar lyrics using advanced semantic embeddings and FAISS vector search</p>
                </div>
                <div className="feature-card">
                    <div className="feature-icon-large"><Globe size={48} /></div>
                    <h3>Multilingual</h3>
                    <p>Seamlessly search across Hindi, Hinglish, and English with automatic language detection</p>
                </div>
                <div className="feature-card">
                    <div className="feature-icon-large"><Zap size={48} /></div>
                    <h3>Lightning Fast</h3>
                    <p>Optimized vector search delivers instant results from thousands of songs and poems</p>
                </div>
            </div>

            <section className="search-section">
                <div className="search-header">
                    <h2 className="section-title">Start Your Discovery</h2>
                    <p className="section-subtitle">Enter lyrics, themes, or emotions in any language</p>
                </div>
                <div className="panel">
                    <form onSubmit={handleSubmit} className="form">
                        <div className="input-row">
                            <div className="search-input-wrapper">
                                <Search className="search-icon" size={20} />
                                <input
                                    value={query}
                                    onChange={(e) => setQuery(e.target.value)}
                                    placeholder="e.g. दिल टूटने की कविता or heartbreak love song"
                                    className="text-input"
                                />
                            </div>
                            <button type="submit" className="primary" disabled={loading}>
                                {loading ? (
                                    <><Clock className="spinner" size={16} /> Searching…</>
                                ) : (
                                    <><Sparkles size={16} /> Search</>
                                )}
                            </button>
                        </div>

                        <div className="controls">
                            <label className="field">
                                <span>Language focus</span>
                                <select value={lang} onChange={(e) => setLang(e.target.value)}>
                                    {languageOptions.map((opt) => (
                                        <option key={opt.value} value={opt.value}>
                                            {opt.label}
                                        </option>
                                    ))}
                                </select>
                            </label>

                            <label className="field">
                                <span>Top K</span>
                                <input
                                    type="number"
                                    min={1}
                                    max={20}
                                    value={topK}
                                    onChange={(e) => setTopK(Number(e.target.value))}
                                />
                            </label>

                            <label className="checkbox">
                                <input
                                    type="checkbox"
                                    checked={includeEnglish}
                                    onChange={(e) => setIncludeEnglish(e.target.checked)}
                                />
                                <span>Include English lyrics</span>
                            </label>
                        </div>
                    </form>

                    <div className="samples-section">
                        <p className="samples-label">Try these examples:</p>
                        <div className="chips" aria-label="sample queries">
                            {sampleQueries.map((item) => (
                                <button key={item.label} type="button" className="chip" onClick={() => handleSampleClick(item.text)}>
                                    <Sparkles className="chip-icon" size={14} />
                                    {item.label}
                                </button>
                            ))}
                        </div>
                    </div>

                    {meta && (
                        <div className="meta">
                            <span className="pill pill-muted">Backend: {meta.backend}</span>
                            <span className="pill pill-muted">Hindi chunks: {meta.counts?.hi ?? 0}</span>
                            <span className="pill pill-muted">English chunks: {meta.counts?.en ?? 0}</span>
                        </div>
                    )}
                </div>
            </section>

            {error && (
                <div className="alert">
                    <AlertTriangle className="alert-icon" size={20} />
                    {error}
                </div>
            )}

            <section className="results" aria-live="polite">
                {loading && (
                    <div className="loader">
                        <div className="loader-spinner"><Music size={48} /></div>
                        <p>Searching through thousands of lyrics...</p>
                    </div>
                )}
                {!loading && results.length === 0 && !error && (
                    <div className="empty-state">
                        <div className="empty-icon"><Music2 size={64} /></div>
                        <h3>Ready to discover amazing lyrics?</h3>
                        <p>Search above or try one of our sample queries to get started</p>
                    </div>
                )}
                {!loading && results.length > 0 && (
                    <>
                        <div className="results-header">
                            <h2 className="results-title">Your Discoveries</h2>
                            <span className="results-count">{results.length} result{results.length !== 1 ? 's' : ''}</span>
                        </div>
                        <div className="result-grid">
                            {results.map((res) => (
                                <ResultCard key={res.id} result={res} />
                            ))}
                        </div>
                    </>
                )}
            </section>

            <footer className="footer">
                <p>Powered by FAISS vector search & multilingual embeddings</p>
            </footer>
        </div>
    )
}

export default App
