import React, { useState, useEffect, useRef, useCallback } from 'react';
import { fetchFiles, uploadFile, deleteFile, rebuildIndex, chatStream, sendVoiceAudio } from './api';
import { createStreamingTTS } from './tts';
import { Send, Upload, Trash2, Cpu, Database, Mic, MicOff, Volume2, VolumeX, RefreshCw } from 'lucide-react';
import './index.css';

const VOICE_PLACEHOLDER = '🎤 Processing...';

function formatMessageContent(content, isError = false) {
  if (!content) return content;
  let text = content
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1');
  if (isError) {
    text = text.replace(/^Error:\s*/i, '').trim();
  }
  return text;
}

export function App() {
  const [files, setFiles] = useState([]);
  const [activeFile, setActiveFile] = useState('All Files');
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [streamStatus, setStreamStatus] = useState(null);
  const [backendError, setBackendError] = useState(null);
  const [isSpeaking, setIsSpeaking] = useState(false);

  const [isRecording, setIsRecording] = useState(false);
  const [autoSpeak, setAutoSpeak] = useState(false);
  const [isRebuilding, setIsRebuilding] = useState(false);

  const mediaRecorderRef = useRef(null);
  const audioChunksRef = useRef([]);
  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const ttsRef = useRef(
    createStreamingTTS((speaking) => setIsSpeaking(speaking))
  );
  const autoSpeakRef = useRef(autoSpeak);

  useEffect(() => {
    autoSpeakRef.current = autoSpeak;
    ttsRef.current.setEnabled(autoSpeak);
    if (!autoSpeak) {
      ttsRef.current.stop();
      setIsSpeaking(false);
    }
  }, [autoSpeak]);

  useEffect(() => {
    loadFiles();
  }, []);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping, streamStatus]);

  const cancelOngoing = useCallback(() => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    ttsRef.current.stop();
    setIsSpeaking(false);
  }, []);

  const loadFiles = async () => {
    try {
      const data = await fetchFiles();
      setFiles(data.files);
      setBackendError(null);
    } catch (e) {
      console.error(e);
      setBackendError('Backend unreachable. Start the API server on port 8000.');
    }
  };

  const parseStreamMarkers = (buffer) => {
    let text = buffer;
    let sources = null;
    let detectedQuery = null;
    let status = null;

    if (text.includes('__QUERY__')) {
      const parts = text.split('__QUERY__');
      if (parts.length >= 3) {
        detectedQuery = parts[1];
        text = parts.slice(2).join('');
      }
    }

    if (text.includes('__SOURCES__')) {
      const parts = text.split('__SOURCES__');
      if (parts.length >= 3) {
        try {
          sources = JSON.parse(parts[1]);
        } catch (e) {
          console.error('Failed to parse sources', e);
        }
        text = parts.slice(2).join('');
      }
    }

    const statusMatch = text.match(/__STATUS__([a-z_]+)__/);
    if (statusMatch) {
      status = statusMatch[1];
      text = text.replace(statusMatch[0], '');
    }

    return { text, sources, detectedQuery, status };
  };

  const updateUserTranscript = useCallback((detectedQuery) => {
    if (!detectedQuery) return;
    setMessages(prev => {
      const newMsg = [...prev];
      for (let i = newMsg.length - 1; i >= 0; i--) {
        if (newMsg[i].role === 'user' && newMsg[i].content === VOICE_PLACEHOLDER) {
          newMsg[i] = { role: 'user', content: `🎤 ${detectedQuery}` };
          break;
        }
      }
      return newMsg;
    });
  }, []);

  const processStreamResponse = async (response, options = {}) => {
    const { onQueryDetected } = options;
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let hasReceivedBytes = false;
    let finalText = '';
    let finalSources = null;
    let finalQuery = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      hasReceivedBytes = true;
      buffer += decoder.decode(value, { stream: true });

      const parsed = parseStreamMarkers(buffer);
      finalText = parsed.text;
      if (parsed.sources) finalSources = parsed.sources;
      if (parsed.detectedQuery) {
        finalQuery = parsed.detectedQuery;
        onQueryDetected?.(finalQuery);
      }
      if (parsed.status === 'generating') {
        setStreamStatus('generating');
      } else if (parsed.sources && !finalText.trim()) {
        setStreamStatus(parsed.sources.length > 0 ? `retrieved:${parsed.sources.length}` : 'generating');
      }

      if (autoSpeakRef.current && finalText) {
        ttsRef.current.onTextUpdate(finalText);
      }

      setMessages(prev => {
        const newMsg = [...prev];
        const last = newMsg[newMsg.length - 1];
        if (last?.role === 'assistant') {
          newMsg[newMsg.length - 1] = {
            ...last,
            content: finalText,
            sources: finalSources ?? last.sources,
          };
        }
        return newMsg;
      });
    }

    if (autoSpeakRef.current && finalText.trim()) {
      ttsRef.current.flushRemaining(finalText);
    }

    setStreamStatus(null);
    return { assistantText: finalText, sources: finalSources, detectedQuery: finalQuery, hasReceivedBytes };
  };

  const handleApiError = async (response) => {
    let detail = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      if (data.detail) detail = typeof data.detail === 'string' ? data.detail : JSON.stringify(data.detail);
    } catch {
      /* ignore */
    }
    setMessages(prev => {
      const newMsg = [...prev];
      newMsg[newMsg.length - 1] = { role: 'assistant', content: detail, isError: true };
      return newMsg;
    });
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    cancelOngoing();
    const userMessage = { role: 'user', content: input };
    const query = input;
    const historyForApi = [...messages, userMessage];

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsTyping(true);
    setStreamStatus('connecting');

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      setMessages(prev => [...prev, { role: 'assistant', content: '', sources: null }]);

      const response = await chatStream(
        {
          query,
          file_filter: activeFile,
          top_k: 3,
          messages: historyForApi,
        },
        controller.signal
      );

      if (!response.ok) {
        await handleApiError(response);
        return;
      }

      await processStreamResponse(response);
    } catch (e) {
      if (e.name === 'AbortError') return;
      setMessages(prev => {
        const newMsg = [...prev];
        newMsg[newMsg.length - 1] = {
          role: 'assistant',
          content: 'Could not reach the server. Is the backend running?',
          isError: true,
        };
        return newMsg;
      });
    } finally {
      setIsTyping(false);
      setStreamStatus(null);
      abortControllerRef.current = null;
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) audioChunksRef.current.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        stream.getTracks().forEach(track => track.stop());
        const audioBlob = new Blob(audioChunksRef.current, { type: 'audio/webm' });
        await handleVoiceChat(audioBlob);
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (err) {
      alert('Microphone access denied. Please allow microphone access in your browser settings.');
      console.error('Microphone error:', err);
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current?.state === 'recording') {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  };

  const handleVoiceChat = async (audioBlob) => {
    cancelOngoing();
    setIsTyping(true);
    setStreamStatus('connecting');

    setMessages(prev => [
      ...prev,
      { role: 'user', content: VOICE_PLACEHOLDER },
      { role: 'assistant', content: '', sources: null },
    ]);

    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const response = await sendVoiceAudio(audioBlob, activeFile, 3, controller.signal);

      if (!response.ok) {
        await handleApiError(response);
        return;
      }

      await processStreamResponse(response, {
        onQueryDetected: updateUserTranscript,
      });
    } catch (e) {
      if (e.name === 'AbortError') return;
      setMessages(prev => {
        const newMsg = [...prev];
        newMsg[newMsg.length - 1] = {
          role: 'assistant',
          content: 'Error processing voice input.',
          isError: true,
        };
        return newMsg;
      });
    } finally {
      setIsTyping(false);
      setStreamStatus(null);
      abortControllerRef.current = null;
    }
  };

  const handleUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
      await uploadFile(file);
      loadFiles();
    } catch (err) {
      alert('Upload failed: ' + err.message);
    }
    e.target.value = '';
  };

  const handleDelete = async (filename) => {
    if (!window.confirm(`Delete ${filename}?`)) return;
    try {
      await deleteFile(filename);
      if (activeFile === filename) setActiveFile('All Files');
      loadFiles();
    } catch (err) {
      alert('Delete failed: ' + err.message);
    }
  };

  const handleRebuild = async () => {
    setIsRebuilding(true);
    try {
      await rebuildIndex();
      alert('Vectorstore rebuilt successfully!');
      loadFiles();
    } catch (err) {
      alert('Rebuild failed: ' + err.message);
    } finally {
      setIsRebuilding(false);
    }
  };

  const renderStatusLine = () => {
    if (!isTyping || !streamStatus) return null;
    if (streamStatus === 'connecting') {
      return <div className="status-line">Connecting...</div>;
    }
    if (streamStatus.startsWith('retrieved:')) {
      const n = streamStatus.split(':')[1];
      return <div className="status-line">Retrieved {n} chunks — generating answer...</div>;
    }
    if (streamStatus === 'generating') {
      return <div className="status-line">Generating answer...</div>;
    }
    return null;
  };

  return (
    <div className="app-container">
      <div className="sidebar">
        <div className="sidebar-header">
          <h2>NEURAL_LINK</h2>
        </div>

        <div className="sidebar-content">
          {backendError && (
            <div className="backend-banner" role="alert">{backendError}</div>
          )}

          <div className="sidebar-section">
            <h3><Database size={16} /> DATA SOURCES</h3>

            <label className="upload-label">
              <Upload size={16} /> UPLOAD DATA
              <input type="file" className="upload-input" accept=".pdf" onChange={handleUpload} />
            </label>

            <div className="file-list">
              {files.map(f => (
                <div key={f} className="file-item">
                  <span>{f}</span>
                  <button type="button" onClick={() => handleDelete(f)} aria-label={`Delete ${f}`}>
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
              {files.length === 0 && !backendError && (
                <span className="empty-hint">NO DATA FOUND</span>
              )}
            </div>
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section">
            <h3><Cpu size={16} /> FILTER CONTEXT</h3>
            <select value={activeFile} onChange={e => setActiveFile(e.target.value)}>
              <option value="All Files">ALL FILES</option>
              {files.map(f => (
                <option key={f} value={f}>{f}</option>
              ))}
            </select>
          </div>

          <div className="sidebar-section">
            <h3><Cpu size={16} /> LOCAL MODEL</h3>
            <div className="model-status">
              <span className="model-badge">LLAMA 3.1 8B (GGUF)</span>
            </div>
            <p className="section-hint">Running locally via llama-cpp-python</p>
          </div>

          <hr className="sidebar-divider" />

          <div className="sidebar-section">
            <h3><RefreshCw size={16} /> MAINTENANCE</h3>
            <button
              type="button"
              className="rebuild-btn"
              onClick={handleRebuild}
              disabled={isRebuilding}
            >
              <RefreshCw size={14} className={isRebuilding ? 'spinning' : ''} />
              {isRebuilding ? 'REBUILDING...' : 'REBUILD INDEX'}
            </button>
            <p className="section-hint">Required after switching embedding models</p>
          </div>
        </div>
      </div>

      <div className="main-content">
        <div className="chat-header">
          <h1>RAG CHATBOT</h1>
          <p>RAG // NEURAL SEARCH PROTOCOL // LOCAL</p>
          <div className="header-controls">
            {activeFile !== 'All Files' && (
              <div className="target-lock">TARGET: {activeFile}</div>
            )}
            <button
              type="button"
              className={`tts-toggle ${autoSpeak ? 'active' : ''}`}
              onClick={() => setAutoSpeak(!autoSpeak)}
              aria-label={autoSpeak ? 'Turn off auto-speak' : 'Turn on auto-speak'}
            >
              {autoSpeak ? <Volume2 size={16} /> : <VolumeX size={16} />}
              <span>{autoSpeak ? 'TTS ON' : 'TTS OFF'}</span>
            </button>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-block">
              <h2>Welcome</h2>
              <p>Upload PDFs in the sidebar, then ask questions about your documents.</p>
              <p className="welcome-hint">Enable TTS to hear answers spoken in sync with the text.</p>
            </div>
          )}

          {messages.map((m, idx) => (
            <div
              key={`${m.role}-${idx}-${m.content?.slice(0, 20) ?? ''}`}
              className={`message ${m.role} ${m.isError ? 'error' : ''} ${m.role === 'assistant' && isSpeaking && idx === messages.length - 1 ? 'speaking' : ''}`}
            >
              <div>{formatMessageContent(m.content, m.isError)}</div>

              {m.sources && m.sources.length > 0 && (
                <details className="sources-card" open={!m.content?.trim()}>
                  <summary>RETRIEVED VECTORS ({m.sources.length} CHUNKS)</summary>
                  <div className="sources-content">
                    {m.sources.map((src, i) => (
                      <div key={i} className="chunk">
                        <strong>Chunk {i + 1}</strong> · {src.source_file} · Page {src.page}<br />
                        {src.snippet}
                      </div>
                    ))}
                  </div>
                </details>
              )}
            </div>
          ))}

          {renderStatusLine()}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-area">
          <form className="input-container" onSubmit={handleSend}>
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="ENTER QUERY..."
              disabled={isTyping}
              aria-label="Chat message"
            />
            <button
              type="button"
              className={`voice-btn ${isRecording ? 'recording' : ''}`}
              onClick={isRecording ? stopRecording : startRecording}
              disabled={isTyping && !isRecording}
              aria-label={isRecording ? 'Stop recording' : 'Start voice input'}
            >
              {isRecording ? <MicOff size={18} /> : <Mic size={18} />}
              {isRecording && <span className="recording-dot" />}
            </button>
            <button type="submit" disabled={isTyping} aria-label="Send message">
              <Send size={18} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
