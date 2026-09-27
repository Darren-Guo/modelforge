/** UI 状态：各类弹窗/抽屉的开关与数据。 */
import { create } from 'zustand';

export interface SourceViewerState {
  open: boolean;
  title: string;
  code: string;
  files?: Record<string, string>; // 多文件模式（文件树）
  downloadFolder?: string;
}

interface UIState {
  sourceViewer: SourceViewerState;
  openSource: (s: Omit<SourceViewerState, 'open'>) => void;
  closeSource: () => void;

  saveModuleOpen: boolean;
  setSaveModuleOpen: (v: boolean) => void;

  trainOpen: boolean;
  setTrainOpen: (v: boolean) => void;

  modelsOpen: boolean;
  setModelsOpen: (v: boolean) => void;

  message: string | null; // 顶部轻提示（不依赖 antd message 的场景）
}

export const useUIStore = create<UIState>((set) => ({
  sourceViewer: { open: false, title: '', code: '' },
  openSource: (s) => set({ sourceViewer: { ...s, open: true } }),
  closeSource: () => set({ sourceViewer: { open: false, title: '', code: '' } }),

  saveModuleOpen: false,
  setSaveModuleOpen: (v) => set({ saveModuleOpen: v }),

  trainOpen: false,
  setTrainOpen: (v) => set({ trainOpen: v }),

  modelsOpen: false,
  setModelsOpen: (v) => set({ modelsOpen: v }),

  message: null,
}));
