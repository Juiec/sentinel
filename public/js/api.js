/* api.js — fetch wrapper, SSE streaming reader, AbortController */
(function () {
  'use strict';

  let abortController = null;

  function startAbort() {
    abortController = new AbortController();
    return abortController.signal;
  }

  function stop() {
    if (abortController) {
      abortController.abort();
    }
  }

  function signal() {
    return abortController ? abortController.signal : null;
  }

  /*
   * Read a Server-Sent-Events stream produced by the server.
   * onToken(text) fires for each token string.
   * onMeta(obj) fires for metadata messages (e.g. type: 'thinking' / 'error').
   * Returns a Promise that resolves when the stream ends.
   */
  function streamChat(messages, onToken, onMeta) {
    return fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: messages }),
      signal: signal()
    }).then(function (res) {
      if (!res.ok) {
        return res.json().then(function (data) {
          return Promise.reject(new Error(data.error || 'Server error.'));
        });
      }
      return readSSE(res, onToken, onMeta);
    });
  }

  /* Parse an SSE response body line by line. */
  async function readSSE(res, onToken, onMeta) {
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const part = await reader.read();
        if (part.done) return;
        buffer += decoder.decode(part.value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop(); // keep the incomplete trailing line for next pass
        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed.startsWith('data:')) continue;
          const payload = trimmed.slice(5).trim();
          if (payload === '[DONE]') continue;
          let data;
          try {
            data = JSON.parse(payload);
          } catch (e) {
            console.warn('api.js: skipping malformed SSE line', payload, e);
            continue;
          }
          if (data.type === 'thinking' || data.type === 'error') {
            if (onMeta) onMeta(data);
          } else if (typeof data.token === 'string' && data.token.length > 0) {
            onToken(data.token);
          }
        }
      }
    } finally {
      reader.cancel().catch(function () {});
    }
  }

  window.Api = {
    startAbort: startAbort,
    stop: stop,
    signal: signal,
    streamChat: streamChat
  };
})();
