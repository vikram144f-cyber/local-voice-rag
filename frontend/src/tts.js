/**
 * Sentence-chunked browser TTS synced with streaming assistant text.
 */

const SENTENCE_END = /([.?!]+)(\s+)/g;

function cleanForSpeech(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, '$1')
    .replace(/\*(.*?)\*/g, '$1')
    .replace(/#{1,6}\s/g, '')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/`(.*?)`/g, '$1')
    .replace(/\n+/g, ' ')
    .trim();
}

export function createStreamingTTS(onSpeakingChange = null) {
  let spokenUpTo = 0;
  let queue = [];
  let speaking = false;
  let enabled = false;

  const notify = () => {
    onSpeakingChange?.(speaking || queue.length > 0);
  };

  const processQueue = () => {
    if (speaking || queue.length === 0 || !enabled) {
      notify();
      return;
    }
    const text = queue.shift();
    if (!text?.trim()) {
      processQueue();
      return;
    }
    speaking = true;
    notify();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.rate = 1.0;
    utterance.pitch = 1.0;
    utterance.volume = 1.0;
    utterance.lang = 'en-US';
    utterance.onend = () => {
      speaking = false;
      processQueue();
    };
    utterance.onerror = () => {
      speaking = false;
      processQueue();
    };
    window.speechSynthesis.speak(utterance);
  };

  const enqueueSentences = (sentences) => {
    for (const s of sentences) {
      const cleaned = cleanForSpeech(s);
      if (cleaned.length > 1) queue.push(cleaned);
    }
    processQueue();
  };

  const extractNewSentences = (fullText) => {
    const slice = fullText.slice(spokenUpTo);
    if (!slice) return [];

    const sentences = [];
    let lastEnd = 0;
    let match;
    SENTENCE_END.lastIndex = 0;
    while ((match = SENTENCE_END.exec(slice)) !== null) {
      const end = match.index + match[0].length;
      sentences.push(slice.slice(lastEnd, end));
      lastEnd = end;
    }

    if (sentences.length > 0) {
      spokenUpTo += lastEnd;
    }
    return sentences;
  };

  return {
    setEnabled(value) {
      enabled = value;
      if (!value) this.stop();
    },
    isSpeaking() {
      return speaking || queue.length > 0;
    },
    stop() {
      window.speechSynthesis.cancel();
      queue = [];
      speaking = false;
      spokenUpTo = 0;
      notify();
    },
    onTextUpdate(fullText) {
      if (!enabled || !fullText) return;
      const newSentences = extractNewSentences(fullText);
      if (newSentences.length > 0) enqueueSentences(newSentences);
    },
    flushRemaining(fullText) {
      if (!enabled || !fullText) return;
      const remainder = fullText.slice(spokenUpTo).trim();
      if (remainder) {
        enqueueSentences([remainder]);
        spokenUpTo = fullText.length;
      }
    },
  };
}
