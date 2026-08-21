require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');
const OpenAI = require('openai');
const rateLimit = require('express-rate-limit');

const app = express();
app.use(cors({ origin: 'http://localhost:3001' }));
app.use(express.json());

// Serve static files from the public folder
app.use(express.static(path.join(__dirname, 'public')));

// Load API client
const openai = new OpenAI({
  apiKey: process.env.API_KEY || 'not-needed',
  baseURL: process.env.API_URL || 'https://api.openai.com/v1',
});

function logApiError(err) {
  console.error('[API Error]', err);
}

// Only send a JSON error if the response has not already started streaming.
// Once SSE headers/tokens have been sent, res.status(...).json(...) throws
// ERR_HTTP_HEADERS_SENT — so guard with res.headersSent and emit an SSE error instead.
function sendChatError(res, err) {
  logApiError(err);
  if (res.headersSent) {
    res.write(`data: ${JSON.stringify({ type: 'error', message: err.message || 'Failed to reach AI service.' })}\n\n`);
    res.write('data: [DONE]\n\n');
    return res.end();
  }
  return res.status(502).json({
    success: false,
    error: err.message || 'Failed to reach AI service. Check API Key, API_URL, or network connection.'
  });
}

// Token check for the /chat endpoint
function authCheck(req, res, next) {
  const token = req.headers['x-secret'] || req.query.secret;
  if (token !== process.env.SECRET) {
    return res.status(401).json({ success: false, error: 'Unauthorized' });
  }
  next();
}

// Rate limiter for /chat
const limiter = rateLimit({
  windowMs: 60 * 1000,
  max: 10,
  message: { success: false, error: 'Too many requests. Try again in a minute.' }
});

// Emit the part of `text` that is safe to send now, holding back a trailing slice
// (the longest suffix that could be the start of THINK_TAG) for the next chunk.
function flushPending(text, thinkTag) {
  for (let len = Math.min(text.length, thinkTag.length - 1); len >= 0; len--) {
    if (text.slice(-len) === thinkTag.slice(0, len)) {
      return text.slice(0, text.length - len);
    }
  }
  return text;
}

// Chat API Endpoint — streams tokens to the client with Server-Sent Events.
// The THINK_TAG marker can be split across stream chunks, so we buffer a trailing
// slice that might become the tag instead of checking only each chunk's start.
app.post('/chat', limiter, authCheck, async (req, res) => {
  try {
    if (!req.body.messages || !Array.isArray(req.body.messages)) {
      return res.status(400).json({ success: false, error: 'Invalid request format. Expected array of messages.' });
    }

    const payload = {
      model: process.env.MODEL || 'unsloth/Qwen3.8-27B-GGUF:UD-IQ2_XXS',
      messages: req.body.messages,
      temperature: parseFloat(req.body.temperature) || 0.7,
      max_tokens: parseInt(req.body.max_tokens) || 2048,
      stream: true
    };

    res.setHeader('Content-Type', 'text/event-stream');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');

    const stream = await openai.chat.completions.create(payload);
    const THINK_TAG = process.env.THINK_TAG || '< think>';
    let thinking = '';
    let inThinking = false;
    let pending = ''; // trailing slice that could be the start of THINK_TAG

    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta || {};
      const piece = delta.content || '';
      if (!piece) continue;

      if (inThinking) {
        thinking += pending + piece;
        pending = '';
        continue;
      }

      pending += piece;
      const markerIdx = pending.indexOf(THINK_TAG);
      if (markerIdx !== -1) {
        inThinking = true;
        const before = pending.slice(0, markerIdx);
        if (before) res.write(`data: ${JSON.stringify({ token: before })}\n\n`);
        thinking += pending.slice(markerIdx + THINK_TAG.length);
        pending = '';
        continue;
      }

      // Emit everything except a trailing slice that could become the tag.
      const emit = flushPending(pending, THINK_TAG);
      if (emit) res.write(`data: ${JSON.stringify({ token: emit })}\n\n`);
      pending = pending.slice(emit.length);
    }

    // Flush any leftover answer text that is not a partial tag.
    if (!inThinking && pending) {
      res.write(`data: ${JSON.stringify({ token: pending })}\n\n`);
      pending = '';
    }

    res.write(`data: ${JSON.stringify({ type: 'thinking', text: thinking })}\n\n`);
    res.write('data: [DONE]\n\n');
    res.end();
  } catch (err) {
    sendChatError(res, err);
  }
});

// SPA index
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 404 handler for unknown routes
app.use((req, res) => {
  res.status(404).json({ success: false, error: 'Route not found' });
});

module.exports = app;

if (require.main === module) {
  const PORT = process.env.PORT || 3001;
  app.listen(PORT, () => console.log(`Proxy server running on http://localhost:${PORT}`));
}
