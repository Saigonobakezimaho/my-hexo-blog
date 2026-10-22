export default {
  async fetch(request, env) {
    // CORS 头
    const corsHeaders = {
      'Access-Control-Allow-Origin': '*',
      'Access-Control-Allow-Methods': 'POST, OPTIONS',
      'Access-Control-Allow-Headers': 'Content-Type, Authorization',
    };

    // 处理预检请求
    if (request.method === 'OPTIONS') {
      return new Response(null, { headers: corsHeaders });
    }

    // 只处理 POST
    if (request.method !== 'POST') {
      return new Response('Only POST allowed', { status: 405, headers: corsHeaders });
    }

    try {
      // 转发到 chatanywhere
      const response = await fetch('https://api.chatanywhere.tech/v1/chat/completions', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${env.AI_API_KEY}`,
          'Content-Type': 'application/json',
        },
        body: await request.text(),
      });

      // 返回响应，保留流式传输
      return new Response(response.body, {
        status: response.status,
        headers: {
          ...corsHeaders,
          'Content-Type': response.headers.get('Content-Type') || 'text/event-stream',
        },
      });
    } catch (e) {
      return new Response(JSON.stringify({ error: e.message }), {
        status: 500,
        headers: { ...corsHeaders, 'Content-Type': 'application/json' },
      });
    }
  },
};
