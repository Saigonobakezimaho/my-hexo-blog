const { exec } = require('child_process');

// 执行 hexo server 命令
const server = exec('hexo server', {
  cwd: __dirname, // 当前项目目录
});

// 输出日志
server.stdout.on('data', (data) => {
  console.log(`Hexo 输出: ${data}`);
});

server.stderr.on('data', (data) => {
  console.error(`Hexo 错误: ${data}`);
});

server.on('close', (code) => {
  console.log(`Hexo 进程已退出，退出码 ${code}`);
});