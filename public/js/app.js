/* app.js — state management, messages array, localStorage, theme, wiring */
(function () {
  'use strict';

  const THEME_KEY = 'ai-chat-theme';
  const STORAGE_KEY = 'ai-chat-history';
  const messages = [];

  const input = document.getElementById('msg');
  const form = document.getElementById('form');
  const emptyState = document.getElementById('emptyState');
  const chips = document.getElementById('chips');
  const stopBtn = document.getElementById('stopBtn');
  const logPanel = document.getElementById('chatLogPanel');
  const logList = document.getElementById('logList');
  const logBackdrop = document.getElementById('logBackdrop');

  /* Theme: first load from prefers-color-scheme, persist choice */
  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    const btn = document.getElementById('themeToggle');
    btn.textContent = theme === 'dark' ? '☀️ Light' : '🌙 Dark';
  }
  function initTheme() {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored) {
      applyTheme(stored);
    } else {
      const dark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      applyTheme(dark ? 'dark' : 'light');
    }
  }
  initTheme();
  document.getElementById('themeToggle').addEventListener('click', function () {
    const current = document.documentElement.getAttribute('data-theme');
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next);
    localStorage.setItem(THEME_KEY, next);
  });

  function saveMessages() {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(messages));
  }

  function loadMessages() {
    const stored = localStorage.getItem(STORAGE_KEY);
    if (stored) {
      try {
        const parsed = JSON.parse(stored);
        if (Array.isArray(parsed)) {
          parsed.forEach(function (m) {
            messages.push(m);
            Chat.appendMessage(m.role === 'user' ? 'user' : 'ai', m.content, m.thinking || '', m.timestamp || '');
          });
        }
      } catch (e) {
        console.warn('Failed to parse stored chat history:', e);
      }
    }
  }

  function resetChat() {
    messages.length = 0;
    document.getElementById('chat').innerHTML = '';
    localStorage.removeItem(STORAGE_KEY);
    emptyState.hidden = false;
  }

  /* Messages store ISO timestamps; chat.js formats them for display. */
  function newTimestamp() {
    return new Date().toISOString();
  }

  loadMessages();
  emptyState.hidden = messages.length > 0;

  function sendRequest() {
    Chat.setBusy(true);
    Chat.showTypingIndicator();
    Api.startAbort();
    const stream = Chat.startStreamBubble();
    let thinking = '';

    Api.streamChat(messages, stream.onToken, function (meta) {
      if (meta.type === 'thinking') {
        thinking = meta.text || '';
      } else if (meta.type === 'error') {
        stream.div.remove();
        Chat.appendMessage('ai', '❌ ' + (meta.message || 'Error contacting server'), '', '', true);
      }
    }).then(function () {
      Chat.hideTypingIndicator();
      Chat.setBusy(false);
      messages.push({ role: 'assistant', content: stream.text(), thinking: thinking, timestamp: newTimestamp() });
      Chat.finishStreamBubble(stream, thinking);
      saveMessages();
      Chat.addRegenerateButton(stream.div, function () {
        messages.pop();
        saveMessages();
        sendRequest();
      });
    }).catch(function (err) {
      Chat.hideTypingIndicator();
      Chat.setBusy(false);
      stream.div.remove();
      if (err.name === 'AbortError') {
        Chat.appendMessage('ai', 'Stopped.');
      } else {
        Chat.appendMessage('ai', '❌ ' + (err.message || 'Error contacting server'), '', '', true);
        console.error(err);
      }
    });
  }

  /* ---- Chat log: drawer open/close + export --------------------------- */
  function buildLogEntries() {
    return messages.map(function (m) {
      return { role: m.role, content: m.content || '', thinking: m.thinking || '', timestamp: m.timestamp || '' };
    });
  }

  function renderLog() {
    logList.innerHTML = '';
    buildLogEntries().forEach(function (m) {
      const row = document.createElement('div');
      row.className = 'log-row';
      const label = document.createElement('span');
      label.textContent = m.role === 'user' ? 'You' : 'AI';
      row.appendChild(label);
      const body = document.createElement('p');
      body.textContent = m.content || '';
      row.appendChild(body);
      if (m.timestamp) {
        const ts = document.createElement('span');
        ts.className = 'log-ts';
        ts.textContent = new Date(m.timestamp).toLocaleString();
        row.appendChild(ts);
      }
      logList.appendChild(row);
    });
  }

  function openLog() {
    renderLog();
    logPanel.hidden = false;
    logBackdrop.hidden = false;
  }

  function closeLog() {
    logPanel.hidden = true;
    logBackdrop.hidden = true;
  }

  function download(filename, text) {
    const blob = new Blob([text], { type: 'application/octet-stream' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    setTimeout(function () { URL.revokeObjectURL(url); }, 1000);
  }

  function exportLog(format) {
    if (format === 'json') {
      download('chat-log.json', JSON.stringify(buildLogEntries(), null, 2));
    } else {
      const lines = [];
      buildLogEntries().forEach(function (m) {
        lines.push('### ' + (m.role === 'user' ? 'You' : 'AI') + ' — ' + new Date(m.timestamp).toLocaleString());
        lines.push(m.content);
        lines.push('');
      });
      download('chat-log.md', lines.join('\n'));
    }
  }

form.addEventListener('submit', function (e) {
    e.preventDefault();
    const msg = input.value.trim();
    if (!msg) return;

    messages.push({ role: 'user', content: msg, timestamp: newTimestamp() });
    Chat.appendMessage('user', msg, '', newTimestamp());
    input.value = '';
    Chat.autoExpand();
    emptyState.hidden = true;

    sendRequest();
  });

  stopBtn.addEventListener('click', function () {
    Api.stop();
  });

  /* Log drawer: open/close + export menu */
  document.getElementById('logBtn').addEventListener('click', openLog);
  document.getElementById('logClose').addEventListener('click', closeLog);
  logBackdrop.addEventListener('click', closeLog);

  const exportMenu = document.getElementById('exportMenu');
  document.getElementById('exportBtn').addEventListener('click', function () {
    exportMenu.classList.toggle('hidden');
  });
  exportMenu.querySelectorAll('[data-format]').forEach(function (btn) {
    btn.addEventListener('click', function () {
      exportLog(btn.getAttribute('data-format'));
      exportMenu.classList.remove('hidden');
    });
  });

  chips.addEventListener('click', function (e) {
    const chip = e.target.closest('.chip');
    if (!chip) return;
    input.value = chip.textContent;
    input.focus();
  });

  document.getElementById('newChat').addEventListener('click', resetChat);
  document.getElementById('clear').addEventListener('click', resetChat);
  input.addEventListener('input', Chat.autoExpand);
})();
