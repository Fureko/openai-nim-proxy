// server.js - OpenAI to NVIDIA NIM API Proxy (version améliorée)
const express = require('express');
const cors = require('cors');
const axios = require('axios');

const app = express();
const PORT = process.env.PORT || 3000;

// Middleware - l'ordre compte
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// ---- Config NVIDIA NIM ----
const NIM_API_BASE = process.env.NIM_API_BASE || 'https://integrate.api.nvidia.com/v1';
const NIM_API_KEY = process.env.NIM_API_KEY;

const SHOW_REASONING = false;        // true => affiche le raisonnement entre <think>
const ENABLE_THINKING_MODE = false;  // true => chat_template_kwargs.thinking

// ---- Limites internes (protègent ton quota de 40 RPM) ----
const RATE_LIMIT_RPM = parseInt(process.env.RATE_LIMIT_RPM || '32', 10); // marge sous 40
const WINDOW_MS = 60_000;
const MAX_RETRIES = parseInt(process.env.MAX_RETRIES || '3', 10);
const REQUEST_TIMEOUT_MS = parseInt(process.env.REQUEST_TIMEOUT_MS || '120000', 10);
const DEFAULT_MAX_TOKENS = parseInt(process.env.DEFAULT_MAX_TOKENS || '4096', 10);

// Mapping des modèles (alias OpenAI -> modèle NIM réel)
const MODEL_MAPPING = {
  'gpt-3.5-turbo': 'nvidia/llama-3.1-nemotron-ultra-253b-v1',
  'DEEP': 'deepseek-ai/deepseek-v3.1-terminus',
  'DEEP2': 'deepseek-ai/deepseek-v3.2',
  'gpt-4o': 'z-ai/glm4_7',
  'claude-3-opus': 'openai/gpt-oss-120b',
  'claude-3-sonnet': 'openai/gpt-oss-20b',
  'GLM5': 'z-ai/glm-5.2',
  'DEEP4P': 'deepseek-ai/deepseek-v4-pro',
  'DEEP4F': 'deepseek-ai/deepseek-v4-flash',
  'Mistral': 'mistralai/mistral-medium-3.5-128b',
  'MoonShot': 'moonshotai/kimi-k2.6',
  'StepFun': 'stepfun-ai/step-3.7-flash',
  'Gemma': 'google/gemma-4-31b-it',
};

// =====================================================================
//  RATE LIMITER : file d'attente glissante. Aucune requête vers NIM
//  ne part si plus de RATE_LIMIT_RPM ont déjà été émises dans la minute.
// =====================================================================
let requestTimestamps = [];
const queue = [];
let processing = false;

function slotsUsed() {
  const now = Date.now();
  requestTimestamps = requestTimestamps.filter((t) => now - t < WINDOW_MS);
  return requestTimestamps.length;
}

function scheduleNimCall(fn) {
  return new Promise((resolve, reject) => {
    queue.push({ fn, resolve, reject });
    processQueue();
  });
}

async function processQueue() {
  if (processing) return;
  processing = true;
  while (queue.length > 0) {
    if (slotsUsed() >= RATE_LIMIT_RPM) {
      const wait = Math.max(WINDOW_MS - (Date.now() - requestTimestamps[0]) + 50, 100);
      console.log(`[rate-limit] file pleine (${queue.length} en attente), pause ${wait}ms`);
      await new Promise((r) => setTimeout(r, wait));
      continue;
    }
    const { fn, resolve, reject } = queue.shift();
    requestTimestamps.push(Date.now());
    // on ne bloque pas la boucle : plusieurs requêtes peuvent être en vol
    fn().then(resolve).catch(reject);
  }
  processing = false;
}

// Appel NIM avec retry/backoff interne (respecte Retry-After)
async function callNim(nimRequest, isStream) {
  let attempt = 0;
  while (true) {
    const response = await scheduleNimCall(() =>
      axios.post(`${NIM_API_BASE}/chat/completions`, nimRequest, {
        headers: {
          Authorization: `Bearer ${NIM_API_KEY}`,
          'Content-Type': 'application/json',
        },
        responseType: isStream ? 'stream' : 'json',
        timeout: isStream ? 0 : REQUEST_TIMEOUT_MS,
        validateStatus: () => true,
      })
    );

    if (response.status === 429 && attempt < MAX_RETRIES) {
      const retryAfter = parseInt(response.headers['retry-after'] || '0', 10);
      const wait = retryAfter > 0 ? retryAfter * 1000 : Math.min(2 ** attempt * 1000, 8000);
      console.warn(`[429] tentative ${attempt + 1}/${MAX_RETRIES}, nouvelle tentative dans ${wait}ms`);
      // si c'est un stream, on consomme/jette le corps pour libérer la connexion
      if (isStream && response.data && typeof response.data.resume === 'function') {
        response.data.resume();
      }
      await new Promise((r) => setTimeout(r, wait));
      attempt++;
      continue;
    }
    return response;
  }
}

// Résolution du modèle SANS appel de test (évite le double-appel)
function resolveModel(model) {
  if (MODEL_MAPPING[model]) return MODEL_MAPPING[model];
  if (model.includes('/')) return model; // déjà un identifiant NIM complet
  const m = model.toLowerCase();
  if (m.includes('gpt-4') || m.includes('405b')) return 'meta/llama-3.1-405b-instruct';
  if (m.includes('70b')) return 'meta/llama-3.1-70b-instruct';
  return model; // on laisse NIM trancher (et renvoyer son erreur si besoin)
}

// ---- Endpoints ----
app.get('/', (req, res) => {
  res.json({
    service: 'OpenAI to NVIDIA NIM Proxy',
    version: '2.0.0',
    endpoints: { health: '/health', models: '/v1/models', chat: '/v1/chat/completions' },
  });
});

app.get('/health', (req, res) => {
  res.json({
    status: 'ok',
    service: 'OpenAI to NVIDIA NIM Proxy',
    reasoning_display: SHOW_REASONING,
    thinking_mode: ENABLE_THINKING_MODE,
    nim_api_configured: !!NIM_API_KEY,
    rate_limit_rpm: RATE_LIMIT_RPM,
    slots_used: slotsUsed(),
    queue_length: queue.length,
  });
});

app.get('/v1/models', (req, res) => {
  res.json({
    object: 'list',
    data: Object.keys(MODEL_MAPPING).map((model) => ({
      id: model,
      object: 'model',
      created: Math.floor(Date.now() / 1000),
      owned_by: 'nvidia-nim-proxy',
    })),
  });
});

app.post('/v1/chat/completions', async (req, res) => {
  try {
    if (!NIM_API_KEY) {
      return res.status(500).json({
        error: { message: 'NVIDIA API key not configured', type: 'invalid_request_error', code: 500 },
      });
    }

    const { model, messages, temperature, max_tokens, stream, top_p, frequency_penalty, presence_penalty } = req.body;

    if (!model || !messages) {
      return res.status(400).json({
        error: { message: 'Missing required fields: model and messages are required', type: 'invalid_request_error', code: 400 },
      });
    }

    const nimModel = resolveModel(model);
    console.log(`Model: ${model} -> ${nimModel}${stream ? ' (stream)' : ''}`);

    const nimRequest = {
      model: nimModel,
      messages,
      temperature: temperature !== undefined ? temperature : 0.6,
      max_tokens: max_tokens || DEFAULT_MAX_TOKENS,
      stream: stream || false,
    };
    if (top_p !== undefined) nimRequest.top_p = top_p;
    if (frequency_penalty !== undefined) nimRequest.frequency_penalty = frequency_penalty;
    if (presence_penalty !== undefined) nimRequest.presence_penalty = presence_penalty;
    if (ENABLE_THINKING_MODE) nimRequest.extra_body = { chat_template_kwargs: { thinking: true } };

    const response = await callNim(nimRequest, !!stream);

    // Erreur renvoyée par NIM
    if (response.status >= 400) {
      let errData = response.data;
      // en mode stream, le corps d'erreur est un flux -> on le lit
      if (stream && errData && typeof errData.on === 'function') {
        errData = await new Promise((resolve) => {
          let buf = '';
          errData.on('data', (c) => (buf += c.toString()));
          errData.on('end', () => {
            try { resolve(JSON.parse(buf)); } catch { resolve({ error: { message: buf } }); }
          });
          errData.on('error', () => resolve(null));
        });
      }
      console.error('NVIDIA API error:', response.status, errData);
      const headers = {};
      if (response.headers['retry-after']) headers['retry-after'] = response.headers['retry-after'];
      return res.set(headers).status(response.status).json({
        error: {
          message: errData?.error?.message || errData?.detail || 'NVIDIA API request failed',
          type: 'invalid_request_error',
          code: response.status,
        },
      });
    }

    // ---- STREAM ----
    if (stream) {
      res.setHeader('Content-Type', 'text/event-stream');
      res.setHeader('Cache-Control', 'no-cache');
      res.setHeader('Connection', 'keep-alive');

      let buffer = '';
      let reasoningStarted = false;

      response.data.on('data', (chunk) => {
        buffer += chunk.toString();
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          if (line.includes('[DONE]')) {
            res.write(line + '\n\n');
            continue;
          }
          try {
            const data = JSON.parse(line.slice(6));
            const delta = data.choices?.[0]?.delta;
            if (delta) {
              const reasoning = delta.reasoning_content;
              const content = delta.content;

              if (SHOW_REASONING) {
                let combined = '';
                if (reasoning && !reasoningStarted) { combined = '<think>\n' + reasoning; reasoningStarted = true; }
                else if (reasoning) { combined = reasoning; }
                if (content && reasoningStarted) { combined += '</think>\n\n' + content; reasoningStarted = false; }
                else if (content) { combined += content; }
                if (combined) { delta.content = combined; delete delta.reasoning_content; }
              } else {
                delta.content = content || '';
                delete delta.reasoning_content;
              }
            }
            res.write(`data: ${JSON.stringify(data)}\n\n`);
          } catch (e) {
            console.error('Parse stream chunk error:', e.message);
            res.write(line + '\n\n');
          }
        }
      });

      response.data.on('end', () => { res.end(); });
      response.data.on('error', (err) => {
        console.error('Stream error:', err.message);
        // on signale proprement la fin pour éviter un retry côté client
        if (!res.writableEnded) { res.write('data: [DONE]\n\n'); res.end(); }
      });
      // si le client (Janitor) coupe, on coupe le flux NIM pour ne rien gaspiller
      req.on('close', () => {
        if (response.data && typeof response.data.destroy === 'function') response.data.destroy();
      });
      return;
    }

    // ---- NON-STREAM ----
    if (!Array.isArray(response.data?.choices)) {
      console.error('Réponse NIM inattendue:', response.data);
      return res.status(502).json({
        error: { message: 'Réponse inattendue de NVIDIA NIM', type: 'invalid_request_error', code: 502 },
      });
    }

    const openaiResponse = {
      id: `chatcmpl-${Date.now()}`,
      object: 'chat.completion',
      created: Math.floor(Date.now() / 1000),
      model,
      choices: response.data.choices.map((choice) => {
        let fullContent = choice.message?.content || '';
        if (SHOW_REASONING && choice.message?.reasoning_content) {
          fullContent = '<think>\n' + choice.message.reasoning_content + '\n</think>\n\n' + fullContent;
        }
        return {
          index: choice.index,
          message: { role: choice.message?.role || 'assistant', content: fullContent },
          finish_reason: choice.finish_reason,
        };
      }),
      usage: response.data.usage || { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    };

    res.json(openaiResponse);
  } catch (error) {
    console.error('Proxy error:', error.message);
    const status = error.response?.status || (error.code === 'ECONNABORTED' ? 504 : 500);
    res.status(status).json({
      error: {
        message: error.code === 'ECONNABORTED' ? 'Timeout en attendant NVIDIA NIM' : error.message || 'Internal server error',
        type: 'invalid_request_error',
        code: status,
      },
    });
  }
});

app.all('*', (req, res) => {
  res.status(404).json({
    error: { message: `Endpoint ${req.method} ${req.path} not found`, type: 'invalid_request_error', code: 404 },
  });
});

app.listen(PORT, '0.0.0.0', () => {
  console.log('========================================');
  console.log(`OpenAI to NVIDIA NIM Proxy (v2) on port ${PORT}`);
  console.log(`Rate limit interne : ${RATE_LIMIT_RPM} req/min`);
  console.log(`Retry max : ${MAX_RETRIES} | timeout : ${REQUEST_TIMEOUT_MS}ms`);
  console.log(`Reasoning display: ${SHOW_REASONING ? 'ON' : 'OFF'} | Thinking: ${ENABLE_THINKING_MODE ? 'ON' : 'OFF'}`);
  console.log(`NIM API Key: ${NIM_API_KEY ? 'YES' : 'NO'}`);
  console.log('========================================');
});

module.exports = app;
