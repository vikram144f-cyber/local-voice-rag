export const API_BASE = import.meta.env.VITE_API_URL ?? 'http://127.0.0.1:8000';
const API_URL = `${API_BASE}/api`;

export const fetchFiles = async () => {
  const res = await fetch(`${API_URL}/files`);
  if (!res.ok) throw new Error('Failed to fetch files');
  return res.json();
};

export const uploadFile = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const res = await fetch(`${API_URL}/upload`, {
    method: 'POST',
    body: formData,
  });
  if (!res.ok) throw new Error('Upload failed');
  return res.json();
};

export const deleteFile = async (filename) => {
  const res = await fetch(`${API_URL}/files/${filename}`, {
    method: 'DELETE',
  });
  if (!res.ok) throw new Error('Delete failed');
  return res.json();
};

export const rebuildIndex = async () => {
  const res = await fetch(`${API_URL}/rebuild`, {
    method: 'POST',
  });
  if (!res.ok) throw new Error('Rebuild failed');
  return res.json();
};

export const sendVoiceAudio = async (audioBlob, fileFilter = 'All Files', topK = 3, signal = null) => {
  const formData = new FormData();
  formData.append('audio', audioBlob, 'recording.webm');

  const params = new URLSearchParams({
    file_filter: fileFilter,
    top_k: topK.toString(),
  });

  return fetch(`${API_URL}/voice/upload-audio?${params}`, {
    method: 'POST',
    body: formData,
    signal,
  });
};

export const chatStream = async (body, signal = null) => {
  return fetch(`${API_URL}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
};
