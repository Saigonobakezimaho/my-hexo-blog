const fs = require('fs');
const path = require('path');
const https = require('https');
const crypto = require('crypto');

const API_HOST = process.env.AI_API_HOST || 'https://api.chatanywhere.tech/v1/chat/completions';
const API_KEY = process.env.AI_API_KEY;
const MODEL = process.env.AI_MODEL || 'gpt-4o-mini';
const COUNT = Number(process.env.POSTS_PER_RUN || 2);
const DRY_RUN = process.argv.includes('--dry-run');
const POSTS_DIR = path.join(__dirname, '..', 'source', '_posts');

const COVERS = [
  'https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=1200&h=630&q=80',
  'https://images.unsplash.com/photo-1555066931-4365d14bab8c?auto=format&fit=crop&w=1200&h=630&q=80',
  'https://images.unsplash.com/photo-1461749280684-dccba630e2f6?auto=format&fit=crop&w=1200&h=630&q=80'
];

const TOPIC_HINTS = [
  'Python 实战技巧',
  '后端架构设计',
  '数据库与存储',
  '前端工程化',
  'DevOps 与自动化',
  'AI 工程实践',
  '代码质量与重构',
  '性能优化',
  '开源工具与生态',
  '软件工程方法论'
];

function requestJson(url, payload) {
  return new Promise((resolve, reject) => {
    const body = JSON.stringify(payload);
    const req = https.request(url, {
      method: 'POST',
      headers: {
        'Authorization': 'Bearer ' + API_KEY,
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body)
      },
      timeout: 120000
    }, (res) => {
      let data = '';
      res.setEncoding('utf8');
      res.on('data', (chunk) => { data += chunk; });
      res.on('end', () => {
        if (res.statusCode < 200 || res.statusCode >= 300) {
          reject(new Error('HTTP ' + res.statusCode + ': ' + data.slice(0, 300)));
          return;
        }
        try {
          resolve(JSON.parse(data));
        } catch (err) {
          reject(new Error('接口返回的不是 JSON: ' + data.slice(0, 200)));
        }
      });
    });
    req.on('timeout', () => { req.destroy(new Error('请求超时')); });
    req.on('error', reject);
    req.write(body);
    req.end();
  });
}

function getExistingTitles() {
  if (!fs.existsSync(POSTS_DIR)) return [];
  return fs.readdirSync(POSTS_DIR)
    .filter((f) => f.endsWith('.md'))
    .map((f) => {
      const raw = fs.readFileSync(path.join(POSTS_DIR, f), 'utf8');
      const m = raw.match(/^title:\s*(.+)$/m);
      return m ? m[1].trim() : '';
    })
    .filter(Boolean);
}

function pick(arr) {
  return arr[Math.floor(Math.random() * arr.length)];
}

function shanghaiNow() {
  return new Date(Date.now() + 8 * 3600 * 1000);
}

function formatDate(d) {
  const p = (n) => String(n).padStart(2, '0');
  return d.getUTCFullYear() + '-' + p(d.getUTCMonth() + 1) + '-' + p(d.getUTCDate()) +
    ' ' + p(d.getUTCHours()) + ':' + p(d.getUTCMinutes()) + ':' + p(d.getUTCSeconds());
}

function buildPrompt(existingTitles, hint) {
  const recent = existingTitles.slice(-40).join('、');
  return [
    '你是一个中文技术博客作者，请写一篇原创技术文章。',
    '风格要求：口语化、有真实场景和具体例子，像资深工程师分享经验，避免空话和 AI 腔。',
    '主题方向参考：' + hint + '。',
    '不要与以下已有文章标题重复：' + recent,
    '正文要求：Markdown 格式，开头用一级标题，包含 3-5 个二级标题，正文 1200-2000 字，可以使用代码块和列表。',
    '严格按照下面的格式输出，不要输出任何解释或代码围栏：',
    'TITLE: 文章标题（不超过20字）',
    'CATEGORIES: 一个分类',
    'TAGS: 标签1, 标签2',
    '---',
    '（这里开始是 Markdown 正文）'
  ].join('\n');
}

function parsePost(text) {
  const lines = text.replace(/```[a-z]*\n?/gi, '').trim().split('\n');
  const result = { title: '', categories: [], tags: [], content: '' };
  let i = 0;
  for (; i < lines.length; i++) {
    const line = lines[i];
    if (/^TITLE:\s*/i.test(line)) result.title = line.replace(/^TITLE:\s*/i, '').trim();
    else if (/^CATEGORIES:\s*/i.test(line)) result.categories = splitList(line.replace(/^CATEGORIES:\s*/i, ''));
    else if (/^TAGS:\s*/i.test(line)) result.tags = splitList(line.replace(/^TAGS:\s*/i, ''));
    else if (/^-{3,}$/.test(line.trim())) { i++; break; }
    else if (result.title && line.trim()) break;
  }
  result.content = lines.slice(i).join('\n').trim();
  if (!result.title) {
    const h1 = result.content.match(/^#\s+(.+)$/m);
    if (h1) result.title = h1[1].trim();
  }
  return result;
}

function splitList(value) {
  return value.split(/[,，、]/).map((s) => s.trim()).filter(Boolean).slice(0, 3);
}

async function generatePost(existingTitles, hint) {
  const res = await requestJson(API_HOST, {
    model: MODEL,
    temperature: 0.9,
    messages: [
      { role: 'system', content: '你是一个中文技术博客作者，严格按照用户要求的格式输出，不要输出额外解释。' },
      { role: 'user', content: buildPrompt(existingTitles, hint) }
    ]
  });
  const text = res.choices && res.choices[0] && res.choices[0].message && res.choices[0].message.content;
  if (!text) throw new Error('接口没有返回内容');
  return parsePost(text);
}

function slugify(title) {
  return title
    .replace(/[\\/:*?"<>|]/g, '')
    .replace(/\s+/g, '-')
    .slice(0, 40);
}

function toMarkdown(post) {
  const date = formatDate(shanghaiNow());
  const categories = Array.isArray(post.categories) && post.categories.length ? post.categories : ['技术'];
  const tags = Array.isArray(post.tags) && post.tags.length ? post.tags : ['技术', '编程'];
  const lines = [
    '---',
    'title: ' + post.title,
    'date: ' + date,
    'categories: [' + categories.join(', ') + ']',
    'tags: [' + tags.map((t) => '"' + t + '"').join(', ') + ']',
    'cover: ' + pick(COVERS),
    '---',
    '',
    post.content.trim(),
    ''
  ];
  return { markdown: lines.join('\n'), date };
}

async function main() {
  if (!API_KEY) {
    console.error('缺少环境变量 AI_API_KEY');
    process.exit(1);
  }

  const existingTitles = getExistingTitles();
  let created = 0;

  for (let i = 0; i < COUNT; i++) {
    const hint = pick(TOPIC_HINTS);
    let post = null;
    for (let attempt = 1; attempt <= 3 && !post; attempt++) {
      try {
        const candidate = await generatePost(existingTitles, hint);
        if (!candidate.title || !candidate.content) throw new Error('缺少 title 或 content');
        if (existingTitles.includes(candidate.title)) throw new Error('标题重复: ' + candidate.title);
        post = candidate;
      } catch (err) {
        console.error('第 ' + attempt + ' 次生成失败: ' + err.message);
      }
    }
    if (!post) {
      console.error('跳过该篇：连续生成失败');
      continue;
    }

    const built = toMarkdown(post);
    const fileName = built.date.slice(0, 10) + '-' + slugify(post.title) + '-' +
      crypto.randomBytes(4).toString('hex') + '.md';

    if (DRY_RUN) {
      console.log('\n=== ' + fileName + ' ===');
      console.log(built.markdown.slice(0, 800));
      console.log('...（正文共 ' + post.content.length + ' 字）');
    } else {
      fs.mkdirSync(POSTS_DIR, { recursive: true });
      fs.writeFileSync(path.join(POSTS_DIR, fileName), built.markdown, 'utf8');
      console.log('已生成: ' + fileName);
    }

    existingTitles.push(post.title);
    created++;
  }

  if (created === 0) {
    console.error('本次没有生成任何文章');
    process.exit(1);
  }
  console.log('完成，共 ' + created + ' 篇');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
