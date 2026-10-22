// hexo-server.js
const { exec } = require('child_process');

// 启动 Hexo 服务（默认端口 4000）
exec('hexo server', (error, stdout, stderr) => {
  if (error) {
    console.error(`执行错误: ${error}`);
    return;
  }
  console.log(`输出: ${stdout}`);
  console.error(`错误: ${stderr}`);
});