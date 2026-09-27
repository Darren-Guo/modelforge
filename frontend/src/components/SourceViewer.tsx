/** 源码查看器：Monaco 编辑器 + 文件树 + zip 下载（双击节点/生成代码共用）。 */
import { useMemo, useState } from 'react';
import Editor from '@monaco-editor/react';
import { Button, Drawer, Empty, List, Space, Tag } from 'antd';
import { DownloadOutlined, FileOutlined } from '@ant-design/icons';
import { useUIStore } from '../stores/uiStore';
import * as api from '../api/client';

function languageOf(file: string): string {
  if (file.endsWith('.py')) return 'python';
  if (file.endsWith('.json')) return 'json';
  if (file.endsWith('.md')) return 'markdown';
  return 'plaintext';
}

export default function SourceViewer() {
  const { sourceViewer, closeSource } = useUIStore();
  const files = sourceViewer.files;
  const fileNames = files ? Object.keys(files) : [];
  const [activeFile, setActiveFile] = useState<string>('');
  const currentFile = activeFile && files?.[activeFile] !== undefined ? activeFile : (fileNames[0] ?? '');
  const code = useMemo(
    () => (files ? (files[currentFile] ?? '') : sourceViewer.code),
    [files, currentFile, sourceViewer.code],
  );

  return (
    <Drawer
      title={
        <Space>
          <span>{sourceViewer.title}</span>
          {files && <Tag color="blue">{fileNames.length} 个文件</Tag>}
        </Space>
      }
      open={sourceViewer.open}
      onClose={closeSource}
      width="72%"
      extra={
        sourceViewer.downloadFolder ? (
          <Button icon={<DownloadOutlined />} onClick={() => window.open(api.downloadUrl(sourceViewer.downloadFolder!))}>
            下载 zip
          </Button>
        ) : null
      }
    >
      {sourceViewer.open && (files ? (
        <div className="mf-source-layout">
          <div className="mf-source-tree">
            <List
              size="small"
              dataSource={fileNames}
              renderItem={(f) => (
                <List.Item
                  className={f === currentFile ? 'active' : ''}
                  onClick={() => setActiveFile(f)}
                  style={{ cursor: 'pointer', padding: '6px 10px' }}
                >
                  <FileOutlined style={{ marginRight: 6 }} /> {f}
                </List.Item>
              )}
            />
          </div>
          <div className="mf-source-code">
            <Editor
              height="100%"
              theme="vs-dark"
              path={currentFile}
              language={languageOf(currentFile)}
              value={code}
              options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
            />
          </div>
        </div>
      ) : code ? (
        <Editor
          height="100%"
          theme="vs-dark"
          language="python"
          value={code}
          options={{ readOnly: true, minimap: { enabled: false }, fontSize: 13, scrollBeyondLastLine: false }}
        />
      ) : (
        <Empty description="暂无源码" />
      ))}
    </Drawer>
  );
}
