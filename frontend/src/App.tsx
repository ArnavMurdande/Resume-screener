import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Send, FileText, Bot, User, Loader2, PieChart, X, CheckCircle2, AlertTriangle, Sparkles, History, Clock } from 'lucide-react';
import './index.css';
import DocUploadSection from './DocUploadSection';

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:8000';

interface Highlight {
    quote: string;
    relevance: string;
}

interface AnalysisData {
    match_score: number;
    strengths: string[];
    gaps: string[];
    summary: string;
    highlights?: Highlight[];
}

interface Report {
    id: string;
    session_id: string;
    match_score: number;
    analysis_json: AnalysisData;
    created_at: string;
}

interface Message {
    role: 'user' | 'assistant';
    content: string;
}



const MatchReportModal: React.FC<{ data: AnalysisData | null; onClose: () => void }> = ({ data, onClose }) => {
  if (!data) return null;

  const handleCopyQuote = (text: string) => {
    navigator.clipboard.writeText(text);
    alert("Quote copied! Press Ctrl+F (or Cmd+F) to find it in the PDF.");
  };

  const scoreColor = data.match_score >= 70 ? 'text-success' : data.match_score >= 50 ? 'text-warning' : 'text-error';
  const scoreBorder = data.match_score >= 70 ? 'border-success' : data.match_score >= 50 ? 'border-warning' : 'border-error';

  return (
    <div className="modal-overlay">
        <div className="modal-content animate-slide-up">
            
            <div className="modal-header">
                <div className="flex items-center gap-3">
                    <Sparkles className="text-secondary" />
                    <h2>Match Analysis Report</h2>
                </div>
                <button onClick={onClose} className="icon-btn">
                    <X size={24} />
                </button>
            </div>

            <div className="modal-body custom-scrollbar">
                
                <div className="score-summary-container">
                    <div className="score-circle-wrapper">
                         <div className={`score-circle ${scoreBorder}`}>
                            <div className="text-center">
                                <span className={`score-text ${scoreColor}`}>{data.match_score}%</span>
                                <p className="text-muted text-sm mt-1">Match</p>
                            </div>
                         </div>
                    </div>

                    <div className="summary-card">
                        <h3>Executive Summary</h3>
                        <p>{data.summary}</p>
                    </div>
                </div>

                <div className="analysis-grid">
                    <div className="analysis-card strength-card">
                        <h3 className="text-success flex-center gap-2 mb-4">
                            <CheckCircle2 size={20} /> Key Strengths
                        </h3>
                        <ul className="strength-list">
                            {data.strengths.map((str, idx) => (
                                <li key={idx}>
                                    <span className="bullet"></span>
                                    {str}
                                </li>
                            ))}
                        </ul>
                    </div>

                    <div className="analysis-card gap-card">
                        <h3 className="text-error flex-center gap-2 mb-4">
                             <AlertTriangle size={20} /> Potential Gaps
                        </h3>
                        <ul className="gap-list">
                            {data.gaps.map((gap, idx) => (
                                <li key={idx}>
                                    <span className="bullet"></span>
                                    {gap}
                                </li>
                            ))}
                        </ul>
                    </div>
                </div>

                {/* Highlights Render */}
                {data.highlights && data.highlights.length > 0 && (
                    <div className="analysis-card highlight-card mt-4 border border-blue-500/30 bg-blue-900/10">
                        <div className="flex justify-between items-center mb-4">
                            <h3 className="text-blue-400 flex items-center gap-2">
                                <Sparkles size={20} /> Verified Evidence
                            </h3>
                            <span className="text-[10px] uppercase tracking-wider bg-blue-500/20 text-blue-300 px-2 py-1 rounded">
                                Interactive
                            </span>
                        </div>
                        
                        <p className="text-xs text-slate-400 mb-3">
                           *Click quotes to copy & find (Ctrl+F)
                        </p>

                        <div className="highlights-container space-y-3">
                            {data.highlights.map((item, idx) => (
                                <div 
                                    key={idx} 
                                    onClick={() => handleCopyQuote(item.quote)}
                                    className="p-3 bg-slate-800 rounded border border-slate-700 cursor-pointer hover:bg-blue-900/30 transition-all"
                                >
                                    <p className="text-sm italic text-slate-200 mb-2 border-l-2 border-blue-500 pl-2">
                                        "{item.quote}"
                                    </p>
                                    <p className="text-xs text-blue-400 font-medium">
                                        Relevance: {item.relevance}
                                    </p>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

            </div>
        </div>
    </div>
  );
};

const HistorySidebar: React.FC<{ isOpen: boolean; onClose: () => void; reports: Report[]; onSelectReport: (data: AnalysisData) => void }> = ({ isOpen, onClose, reports, onSelectReport }) => {
  const formatDate = (dateStr: string) => {
    const date = new Date(dateStr);
    return date.toLocaleDateString('en-US', { 
      month: 'short', 
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getScoreColor = (score: number) => {
    if (score >= 70) return 'text-success';
    if (score >= 50) return 'text-warning';
    return 'text-error';
  };

  return (
    <>
      {isOpen && <div className="history-overlay" onClick={onClose} />}
      <div className={`history-sidebar ${isOpen ? 'open' : ''}`}>
        <div className="history-sidebar-header">
          <h2 style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <History size={20} /> Report History
          </h2>
          <button onClick={onClose} className="icon-btn">
            <X size={20} />
          </button>
        </div>
        <div className="history-sidebar-body custom-scrollbar">
          {reports.length === 0 ? (
            <div className="empty-history">
              <Clock size={48} style={{ opacity: 0.3, marginBottom: '16px' }} />
              <p>No reports yet</p>
              <p style={{ fontSize: '0.8rem', marginTop: '8px' }}>Generate a match report to see it here</p>
            </div>
          ) : (
            reports.map((report) => (
              <div 
                key={report.id} 
                className="history-item"
                onClick={() => onSelectReport(report.analysis_json)}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
                  <span className={`history-item-score ${getScoreColor(report.match_score)}`}>
                    {report.match_score}%
                  </span>
                  <span className="history-item-date">{formatDate(report.created_at)}</span>
                </div>
                <p className="history-item-summary">
                  {report.analysis_json?.summary || 'No summary available'}
                </p>
              </div>
            ))
          )}
        </div>
      </div>
    </>
  );
};

function App() {
  const [sessionId, setSessionId] = useState<string>(() => {
    const existing = localStorage.getItem('session_id');
    if (existing) return existing;
    const newId = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2) + Date.now().toString(36);
    return newId;
  });

  useEffect(() => {
    localStorage.setItem('session_id', sessionId);
  }, [sessionId]);

  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [resumePreview, setResumePreview] = useState<string | null>(null);
  const [resumeStatus, setResumeStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  const [jdFile, setJdFile] = useState<File | null>(null);
  const [jdPreview, setJdPreview] = useState<string | null>(null);
  const [jdStatus, setJdStatus] = useState<'idle' | 'uploading' | 'success' | 'error'>('idle');

  const [messages, setMessages] = useState<Message[]>([
    { role: 'assistant', content: 'Welcome! Please upload a Resume and Job Description to get started.' }
  ]);
  const [input, setInput] = useState('');
  const [isThinking, setIsThinking] = useState(false);
  const chatEndRef = useRef<HTMLDivElement>(null);

  const [analysisData, setAnalysisData] = useState<AnalysisData | null>(() => {
    const saved = localStorage.getItem(`analysis_${sessionId}`);
    return saved ? JSON.parse(saved) : null;
  });
  const [lastAnalyzedDocs, setLastAnalyzedDocs] = useState<{resume: string | null; jd: string | null}>(() => {
    const saved = localStorage.getItem(`docs_${sessionId}`);
    return saved ? JSON.parse(saved) : { resume: null, jd: null };
  });

  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [showModal, setShowModal] = useState(false);

  // History state
  const [historyReports, setHistoryReports] = useState<Report[]>([]);
  const [showHistory, setShowHistory] = useState(false);
  const [, setIsLoadingHistory] = useState(false);

  // Fetch history from backend
  const fetchHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const res = await axios.get(`${BACKEND_URL}/history/${sessionId}`);
      // FIX: Backend returns the array directly, not wrapped in 'reports'
      setHistoryReports(Array.isArray(res.data) ? res.data : []); 
    } catch (error) {
      console.error('Failed to fetch history:', error);
    } finally {
      setIsLoadingHistory(false);
    }
  };

  // Load history on mount
  useEffect(() => {
    fetchHistory();
  }, [sessionId]);
  
  // RESET SESSION FUNCTION (Fixes "Session Reuse" Bug)
  const handleResetSession = () => {
     if (confirm("Start a new analysis? This will clear current documents and chat.")) {
         const newId = crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).substring(2) + Date.now().toString(36);
         setSessionId(newId);
         setResumeFile(null);
         setResumePreview(null);
         setResumeStatus('idle');
         setJdFile(null);
         setJdPreview(null);
         setJdStatus('idle');
         setMessages([{ role: 'assistant', content: 'New session started! Please upload documents.' }]);
         setAnalysisData(null);
         setLastAnalyzedDocs({ resume: null, jd: null });
     }
  };

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleUpload = async (file: File, type: 'resume' | 'jd') => {
    const setStatus = type === 'resume' ? setResumeStatus : setJdStatus;
    setStatus('uploading');

    const formData = new FormData();
    formData.append('file', file);
    formData.append('doc_type', type);
    formData.append('session_id', sessionId);

    try {
      await axios.post(`${BACKEND_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setStatus('success');
      
      // CHANGED: Reset chat history to prevent "Ghost Context" from previous files
      setMessages([
        { role: 'assistant', content: `✅ New ${type === 'resume' ? 'Resume' : 'JD'} uploaded! Previous chat context cleared.` }
      ]);
      
      // Clear analysis data as it's now stale
      setAnalysisData(null);
      
    } catch (error) {
      console.error(error);
      setStatus('error');
      setMessages(prev => [...prev, { 
          role: 'assistant', 
          content: `❌ Failed to process ${type === 'resume' ? 'Resume' : 'Job Description'}. Please try again.` 
      }]);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Chat Guard Rails
    const isReady = resumeStatus === 'success' && jdStatus === 'success';
    if (!isReady) {
       alert("Please upload both a Resume and Job Description to start the chat.");
       return;
    }

    const userMsg = input;
    setInput('');
    const newHistory: Message[] = [...messages, { role: 'user', content: userMsg }];
    setMessages(newHistory);
    setIsThinking(true);

    try {
      const apiHistory = newHistory.slice(-10).map(m => ({
          role: m.role,
          content: m.content
      }));

      const res = await axios.post(`${BACKEND_URL}/chat`, {
        question: userMsg,
        history: apiHistory,
        session_id: sessionId
      });

      // CHANGED: explicit check and fallback
      const aiResponse = res.data.answer || "⚠️ Received empty response from server.";
      
      setMessages(prev => [...prev, { role: 'assistant', content: aiResponse }]);

    } catch (error: any) {
      console.error("Chat Error Details:", error); // Added logging
      console.error(error);
      if (error.response?.status === 429) {
          setMessages(prev => [...prev, { role: 'assistant', content: "⚠️ AI Traffic High. Please wait 30s before your next message." }]);
      } else {
          setMessages(prev => [...prev, { role: 'assistant', content: "Sorry, I encountered an error. Please try again." }]);
      }
    } finally {
      setIsThinking(false);
    }
  };

  const handleAnalyze = async () => {
      if (!resumeFile && !lastAnalyzedDocs.resume) {
          alert("Please upload a Resume first.");
          return;
      }
      if (!jdFile && !lastAnalyzedDocs.jd) {
          alert("Please upload a Job Description first.");
          return;
      }

      const currentResumeName = resumeFile ? resumeFile.name : lastAnalyzedDocs.resume;
      const currentJdName = jdFile ? jdFile.name : lastAnalyzedDocs.jd;
      
      setIsAnalyzing(true);
      try {
          const res = await axios.post(`${BACKEND_URL}/analyze`, {
              session_id: sessionId
          });
          setAnalysisData(res.data);
          
          // Persist
          localStorage.setItem(`analysis_${sessionId}`, JSON.stringify(res.data));
          const docInfo = { resume: currentResumeName || null, jd: currentJdName || null };
          setLastAnalyzedDocs(docInfo);
          localStorage.setItem(`docs_${sessionId}`, JSON.stringify(docInfo));
          
          setShowModal(true);
          
          // Refresh history after new report
          fetchHistory();
      } catch (error: any) {
          console.error(error);
          if (error.response?.status === 429) {
             alert("⚠️ AI Traffic High. Please wait 30s before generating a report.");
          } else {
             alert("Failed to generate analysis. Ensure both documents are uploaded.");
          }
      } finally {
          setIsAnalyzing(false);
      }
  };

  const handleSelectHistoryReport = (reportData: AnalysisData) => {
    setAnalysisData(reportData);
    setShowHistory(false);
    setShowModal(true);
  };

  return (
    <div className="min-h-screen w-full bg-[#0f172a] text-slate-200 flex flex-col font-sans">
      
      {showModal && <MatchReportModal data={analysisData} onClose={() => setShowModal(false)} />}
      
      <HistorySidebar 
        isOpen={showHistory} 
        onClose={() => setShowHistory(false)} 
        reports={historyReports}
        onSelectReport={handleSelectHistoryReport}
      />
      
      <header className="app-header shadow-md shrink-0">
         <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <div className="logo-box">
                <FileText size={18} className="text-white"/>
            </div>
            <h1 className="app-title">AI Resume <span className="font-normal text-muted">Screener</span></h1>
         </div>
         
         <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button 
               onClick={handleResetSession}
               className="icon-btn"
               title="New Analysis (Clear Data)"
            >
               <Sparkles size={16} /> <span className="text-xs ml-1 font-medium text-blue-300">New Analysis</span>
            </button>
            <div className="divider" style={{height: '24px'}} />

            <button 
                onClick={handleAnalyze}
                disabled={isAnalyzing || !(resumeStatus === 'success' && jdStatus === 'success')}
                className={`analyze-btn ${isAnalyzing || !(resumeStatus === 'success' && jdStatus === 'success') ? 'disabled' : ''}`}
            >
                {isAnalyzing ? (
                   <> <Loader2 size={16} className="animate-spin"/> <span>Analyzing...</span> </>
                ) : (
                   <> <PieChart size={16} /> <span>Generate Match Report</span> </>
                )}
            </button>
            
            <button 
                onClick={() => setShowHistory(true)}
                className="history-btn"
            >
                <History size={16} /> <span>History</span>
            </button>
             
            <div className="mobile-hide" style={{ height: '20px', width: '1px', background: 'rgba(255,255,255,0.1)' }}></div>
            
            <div className="mobile-hide" style={{ display: 'flex', alignItems: 'center', gap: '16px', fontSize: '0.875rem', color: '#94a3b8' }}>
                <span>Gemini 3 Flash</span>
                <div style={{ height: '16px', width: '1px', background: 'rgba(255,255,255,0.1)' }}></div>
                <span>v1.0.0</span>
            </div>
         </div>
      </header>

      <main className="flex-1 p-6">
        <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
            
            {/* Top Row: Split View for Resume & JD */}
            <div className="docs-split-container">
                <div className="doc-card-wrapper">
                    <DocUploadSection 
                        title="Candidate Resume" 
                        type="resume"
                        file={resumeFile}
                        setFile={setResumeFile}
                        previewUrl={resumePreview}
                        setPreviewUrl={setResumePreview}
                        status={resumeStatus}
                        setStatus={setResumeStatus}
                        onUpload={handleUpload}
                    />
                </div>

                <div className="doc-card-wrapper">
                    <DocUploadSection 
                        title="Job Description" 
                        type="jd"
                        file={jdFile}
                        setFile={setJdFile}
                        previewUrl={jdPreview}
                        setPreviewUrl={setJdPreview}
                        status={jdStatus}
                        setStatus={setJdStatus}
                        onUpload={handleUpload}
                    />
                </div>
            </div>

            {/* Bottom Row: Chat Section */}
            <div className="chat-wrapper">
                <div className="chat-section glass-panel h-full flex flex-col">
                    <div className="chat-header shrink-0">
                        <h2 className="flex-center gap-2">
                            <Bot className="text-secondary" size={18} />
                            AI Assistant
                        </h2>
                        <span className="live-badge">Live</span>
                    </div>

                    <div className="chat-body custom-scrollbar flex-1 overflow-y-auto">
                        {messages.map((msg, idx) => (
                        <div key={idx} className={`chat-message ${msg.role === 'user' ? 'user-message' : 'ai-message'}`}>
                            <div className={`avatar ${msg.role === 'user' ? 'user-avatar' : 'ai-avatar'}`}>
                                {msg.role === 'user' ? <User size={14} className="text-white"/> : <Bot size={14} className="text-white"/>}
                            </div>
                            <div className={`message-bubble ${msg.role === 'user' ? 'user-bubble' : 'ai-bubble'}`}>
                                {msg.role === 'user' ? (
                                    <p>{msg.content}</p>
                                ) : (
                                    <div className="markdown-content">
                                        <ReactMarkdown 
                                            children={msg.content} 
                                            remarkPlugins={[remarkGfm]}
                                        />
                                    </div>
                                )}
                            </div>
                        </div>
                        ))}
                        {isThinking && (
                        <div className="chat-message ai-message">
                            <div className="avatar ai-avatar">
                                <Bot size={14} className="text-white"/>
                            </div>
                            <div className="message-bubble ai-bubble flex items-center gap-2">
                                <span className="typing-dot"></span>
                                <span className="typing-dot delay-1"></span>
                                <span className="typing-dot delay-2"></span>
                            </div>
                        </div>
                        )}
                        <div ref={chatEndRef} />
                    </div>

                    <div className="chat-footer shrink-0">
                        <form onSubmit={handleSendMessage} style={{ position: 'relative', width: '100%' }}>
                            <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder={resumeStatus === 'success' && jdStatus === 'success' ? "Ask about the candidate..." : "Process documents to start chat"}
                            className="chat-input"
                            disabled={isThinking || !(resumeStatus === 'success' && jdStatus === 'success')}
                            />
                            <button 
                            type="submit" 
                            disabled={!input.trim() || isThinking || !(resumeStatus === 'success' && jdStatus === 'success')}
                            className="send-btn"
                            >
                            <Send size={18} />
                            </button>
                        </form>
                    </div>
                </div>
            </div>

        </div>
      </main>
    </div>
  );
}

export default App;
