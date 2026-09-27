/** 应用布局：工具栏 + 左面板 + 画布 + 右属性面板 + 各弹窗。 */
import { useEffect } from 'react';
import { ReactFlowProvider } from '@xyflow/react';
import { Alert } from 'antd';
import Toolbar from './components/Toolbar';
import Palette from './components/Palette';
import Canvas from './components/Canvas';
import PropPanel from './components/PropPanel';
import SourceViewer from './components/SourceViewer';
import SaveModuleModal from './components/SaveModuleModal';
import TrainModal from './components/TrainModal';
import ModelsDrawer from './components/ModelsDrawer';
import { useGraphStore } from './stores/graphStore';

export default function App() {
  const { init, validation } = useGraphStore();
  const initFailed = validation?.errors.some((e) => e.where === '网络') ?? false;

  useEffect(() => {
    void init();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="mf-app">
      <Toolbar />
      {initFailed && (
        <Alert
          type="warning" showIcon banner
          message="后端未连接：请在 backend 目录运行 uvicorn app.main:app --port 8000 后刷新页面"
        />
      )}
      <div className="mf-main">
        <aside className="mf-left"><Palette /></aside>
        <section className="mf-center">
          <ReactFlowProvider>
            <Canvas />
          </ReactFlowProvider>
        </section>
        <aside className="mf-right"><PropPanel /></aside>
      </div>
      <SourceViewer />
      <SaveModuleModal />
      <TrainModal />
      <ModelsDrawer />
    </div>
  );
}
