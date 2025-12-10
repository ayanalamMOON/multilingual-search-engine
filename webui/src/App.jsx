import { AlertTriangle, Clock, FileText, Flag, Globe, Languages, Lightbulb, MessageSquare, Music, Music2, Search, Sparkles, Trash2, Zap } from 'lucide-react'
import { useEffect, useMemo, useState } from 'react'
import './App.css'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const languageOptions = [
    { value: 'auto', label: 'Auto detect' },
    { value: 'hi', label: 'Hindi first' },
    { value: 'en', label: 'English only' },
    { value: 'both', label: 'Blend both' },
]

const ragModes = [
    { value: 'summary', label: 'Summary', icon: FileText, description: 'Get a concise summary' },
    { value: 'recommendation', label: 'Recommendations', icon: Lightbulb, description: 'Get personalized suggestions' },
    { value: 'chat', label: 'Chat', icon: MessageSquare, description: 'Ask questions' },
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
    const [useRAG, setUseRAG] = useState(false)
    const [ragMode, setRagMode] = useState('summary')
    const [ragResponse, setRagResponse] = useState(null)
    const [userMessage, setUserMessage] = useState('')
    const [sessionId, setSessionId] = useState(null)
    const [chatHistory, setChatHistory] = useState([])
    const [sidebarOpen, setSidebarOpen] = useState(true)

    const apiBase = useMemo(() => API_BASE.replace(/\/$/, ''), [])

    // Debug: Log chat history changes
    useEffect(() => {
        console.log('Chat history updated:', chatHistory)
        console.log('Session ID:', sessionId)
        console.log('RAG mode:', ragMode)
        console.log('Use RAG:', useRAG)
    }, [chatHistory, sessionId, ragMode, useRAG])

    const handleSubmit = async (e) => {
        e?.preventDefault()
        setError('')
        setRagResponse(null)

        if (!query.trim()) {
            setError('Please enter something to search')
            return
        }

        if (useRAG && ragMode === 'chat' && !userMessage.trim()) {
            setError('Please enter a question for chat mode')
            return
        }

        setLoading(true)
        try {
            if (useRAG) {
                // RAG mode
                const response = await fetch(`${apiBase}/api/rag`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        query,
                        top_k: Number(topK),
                        mode: ragMode,
                        user_message: ragMode === 'chat' ? userMessage : undefined,
                        session_id: ragMode === 'chat' ? sessionId : undefined
                    }),
                })

                if (!response.ok) {
                    const detail = await response.json().catch(() => ({}))
                    throw new Error(detail.detail || 'RAG request failed')
                }

                const data = await response.json()
                setRagResponse(data)

                // Update session ID and chat history for chat mode
                if (ragMode === 'chat') {
                    if (data.session_id) {
                        setSessionId(data.session_id)
                    }
                    if (data.chat_history && Array.isArray(data.chat_history)) {
                        console.log('Chat history received:', data.chat_history)
                        setChatHistory(data.chat_history)
                    }
                    setUserMessage('') // Clear input after sending
                }

                setResults([]) // Clear regular results
                setMeta(null)
            } else {
                // Regular search mode
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
                setRagResponse(null) // Clear RAG response
            }
        } catch (err) {
            setError(err.message || 'Something went wrong')
            setResults([])
            setRagResponse(null)
        } finally {
            setLoading(false)
        }
    }

    const handleSampleClick = (text) => {
        setQuery(text)
        setTimeout(() => handleSubmit(), 0)
    }

    const clearChatHistory = async () => {
        if (!sessionId) return

        try {
            const response = await fetch(`${apiBase}/api/rag/clear/${sessionId}`, {
                method: 'POST',
            })

            if (response.ok) {
                setChatHistory([])
                setRagResponse(null)
            }
        } catch (err) {
            console.error('Failed to clear chat history:', err)
        }
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
                    {/* RAG Mode Toggle */}
                    <div className="rag-toggle">
                        <label className="checkbox">
                            <input
                                type="checkbox"
                                checked={useRAG}
                                onChange={(e) => setUseRAG(e.target.checked)}
                            />
                            <span style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                                <Sparkles size={16} />
                                AI-Enhanced Mode (RAG)
                            </span>
                        </label>
                        {useRAG && (
                            <p style={{ fontSize: '0.875rem', color: '#666', marginTop: '4px', marginLeft: '24px' }}>
                                Get AI-generated insights, summaries, and recommendations
                            </p>
                        )}
                    </div>

                    {/* RAG Mode Selector */}
                    {useRAG && (
                        <div className="rag-modes">
                            {ragModes.map((mode) => {
                                const Icon = mode.icon
                                return (
                                    <button
                                        key={mode.value}
                                        type="button"
                                        className={`rag-mode-btn ${ragMode === mode.value ? 'active' : ''}`}
                                        onClick={() => setRagMode(mode.value)}
                                    >
                                        <Icon size={20} />
                                        <div>
                                            <div style={{ fontWeight: '600' }}>{mode.label}</div>
                                            <div style={{ fontSize: '0.75rem', opacity: 0.7 }}>{mode.description}</div>
                                        </div>
                                    </button>
                                )
                            })}
                        </div>
                    )}

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
                            {!useRAG && (
                                <>
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

                                    <label className="checkbox">
                                        <input
                                            type="checkbox"
                                            checked={includeEnglish}
                                            onChange={(e) => setIncludeEnglish(e.target.checked)}
                                        />
                                        <span>Include English lyrics</span>
                                    </label>
                                </>
                            )}

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
                        </div>

                        {/* Chat mode user message input */}
                        {useRAG && ragMode === 'chat' && (
                            <div style={{ marginTop: '1rem' }}>
                                <label className="field">
                                    <span>Your Question</span>
                                    <input
                                        type="text"
                                        value={userMessage}
                                        onChange={(e) => setUserMessage(e.target.value)}
                                        placeholder="e.g., What makes these poems special?"
                                        className="text-input"
                                    />
                                </label>
                            </div>
                        )}
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

            {/* Chat History Sidebar */}
            {useRAG && ragMode === 'chat' && (
                <>
                    {/* Sidebar Toggle Button */}
                    <button
                        className={`sidebar-toggle ${sidebarOpen ? 'open' : ''}`}
                        onClick={() => setSidebarOpen(!sidebarOpen)}
                        title={sidebarOpen ? 'Close chat history' : 'Open chat history'}
                    >
                        <MessageSquare size={20} />
                        {!sidebarOpen && chatHistory.length > 0 && (
                            <span className="sidebar-badge">{chatHistory.length}</span>
                        )}
                    </button>

                    {/* Sidebar */}
                    <aside className={`chat-sidebar ${sidebarOpen ? 'open' : ''}`}>
                        <div className="chat-sidebar-header">
                            <div className="chat-sidebar-title">
                                <MessageSquare size={20} />
                                <h3>Chat History</h3>
                            </div>
                            <div className="chat-sidebar-actions">
                                {chatHistory.length > 0 && (
                                    <button
                                        type="button"
                                        className="sidebar-clear-btn"
                                        onClick={clearChatHistory}
                                        title="Clear chat history"
                                    >
                                        <Trash2 size={16} />
                                    </button>
                                )}
                                <button
                                    type="button"
                                    className="sidebar-close-btn"
                                    onClick={() => setSidebarOpen(false)}
                                    title="Close sidebar"
                                >
                                    <Zap size={16} style={{ transform: 'rotate(90deg)' }} />
                                </button>
                            </div>
                        </div>
                        <div className="chat-sidebar-content">
                            {sessionId && (
                                <div className="chat-sidebar-session">
                                    <span className="session-label">Session:</span>
                                    <span className="session-id">{sessionId.slice(0, 8)}...</span>
                                </div>
                            )}
                            {chatHistory.length === 0 ? (
                                <div className="chat-sidebar-empty">
                                    <MessageSquare size={48} style={{ opacity: 0.3 }} />
                                    <p>No messages yet</p>
                                    <p style={{ fontSize: '0.875rem', opacity: 0.7 }}>Start chatting to see your conversation history</p>
                                </div>
                            ) : (
                                <div className="chat-sidebar-messages">
                                    {chatHistory.map((msg, idx) => (
                                        <div key={idx} className={`chat-sidebar-message ${msg.role}`}>
                                            <div className="chat-sidebar-message-header">
                                                <span className="chat-sidebar-message-role">
                                                    {msg.role === 'user' ? 'You' : 'AI'}
                                                </span>
                                                <span className="chat-sidebar-message-number">#{idx + 1}</span>
                                            </div>
                                            <div className="chat-sidebar-message-content">
                                                {msg.content}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    </aside>

                    {/* Overlay for mobile */}
                    {sidebarOpen && (
                        <div
                            className="sidebar-overlay"
                            onClick={() => setSidebarOpen(false)}
                        />
                    )}
                </>
            )}

            {/* RAG Response Display */}
            {ragResponse && !loading && (
                <section className="rag-response-section">
                    <div className="rag-response-card">
                        <div className="rag-response-header">
                            <Sparkles size={24} />
                            <div>
                                <h2>AI Response</h2>
                                <p style={{ fontSize: '0.875rem', opacity: 0.7 }}>
                                    Query: "{ragResponse.query}" • Mode: {ragResponse.mode}
                                </p>
                            </div>
                        </div>
                        <div className="rag-response-content">
                            <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit', lineHeight: '1.6' }}>
                                {ragResponse.response}
                            </pre>
                        </div>
                        {ragResponse.sources && ragResponse.sources.length > 0 && (
                            <div className="rag-sources">
                                <h3 style={{ fontSize: '0.875rem', fontWeight: '600', marginBottom: '0.5rem' }}>
                                    Sources ({ragResponse.sources.length})
                                </h3>
                                <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
                                    {ragResponse.sources.map((source, idx) => (
                                        <div key={idx} style={{
                                            padding: '0.5rem',
                                            background: '#f5f5f5',
                                            borderRadius: '4px',
                                            fontSize: '0.875rem'
                                        }}>
                                            <strong>{source.title || 'Untitled'}</strong>
                                            {source.poet && <span> by {source.poet}</span>}
                                            {source.score !== undefined && (
                                                <span style={{ opacity: 0.6 }}> • Score: {source.score.toFixed(3)}</span>
                                            )}
                                        </div>
                                    ))}
                                </div>
                            </div>
                        )}
                    </div>
                </section>
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
