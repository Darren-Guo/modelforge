/** 自定义模块编辑会话：进入前备份模型级端口，退出（保存/取消）后恢复。
 *
 * graphStore.startEditModule 会用模块接口覆盖 graphInputs/graphOutputs（graphStore.ts:633-634），
 * 而其 snapshot 只存 nodes/edges，cancelEditModule 不回滚图级端口——模型的对外接口会被
 * 静默替换成被编辑模块的接口，导出/校验/训练全部基于错误端口。根因应在 store 的 snapshot
 * 里补 graphInputs/graphOutputs；store 层修复前，这里在组件层兜底：
 * 进入编辑时把模型端口备份到会话变量，保存/取消退出时一并写回。
 */
import { useGraphStore } from '../stores/graphStore';
import type { ModuleDef, Port } from '../schema/graph';
import { namespaceGraphIds } from './graphIds';

let modelPorts: { inputs: Port[]; outputs: Port[] } | null = null;

export function beginModuleEdit(mod: ModuleDef): void {
  const s = useGraphStore.getState();
  if (s.editing) return; // 已在编辑态：不要用模块端口覆盖备份
  modelPorts = {
    inputs: s.graphInputs.map((p) => ({ ...p })),
    outputs: s.graphOutputs.map((p) => ({ ...p })),
  };
  // 模块定义里的历史 id 可能与会话内 nextId 撞车，进入编辑前改名
  s.startEditModule({ ...mod, graph: namespaceGraphIds(mod.graph) });
  useGraphStore.setState({ selectedNodeIds: [], selectedEdgeIds: [] });
}

export function endModuleEdit(): void {
  const s = useGraphStore.getState();
  s.cancelEditModule();
  if (modelPorts) {
    useGraphStore.setState({
      graphInputs: modelPorts.inputs,
      graphOutputs: modelPorts.outputs,
      selectedNodeIds: [],
      selectedEdgeIds: [],
    });
    modelPorts = null;
  }
}
