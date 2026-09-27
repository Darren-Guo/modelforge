window.addEventListener('error', (e) => {
  const el = document.createElement('pre');
  el.id = 'debug-error';
  el.style.cssText = 'color:red;background:#fff;padding:12px;z-index:99999;position:fixed;inset:0;overflow:auto';
  el.textContent = 'ERR: ' + (e.error?.stack || e.message);
  document.body.appendChild(el);
});

import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { ConfigProvider, App as AntApp } from 'antd';
import zhCN from 'antd/locale/zh_CN';
import App from './App';
import './index.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider locale={zhCN} theme={{ token: { borderRadius: 6 } }}>
      <AntApp>
        <App />
      </AntApp>
    </ConfigProvider>
  </StrictMode>,
);
