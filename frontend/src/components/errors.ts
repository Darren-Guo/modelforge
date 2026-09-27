/** 组件层 API 错误文案统一格式化。
 *
 * 后端 detail 有三种形态：纯字符串、结构化对象（{message} / {errors} / {report.errors}）、
 * FastAPI 422 校验数组（[{loc,msg}]）。这里统一成一行可读文案，避免把原始 JSON 抛给用户。
 */
import { ApiError } from '../api/client';

export function apiErrorText(e: unknown, fallback: string): string {
  const detail = e instanceof ApiError ? e.detail : null;
  if (typeof detail === 'string' && detail) return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((x: { loc?: (string | number)[]; msg?: string }) =>
        `${(x.loc ?? []).filter((p) => p !== 'body').join('.')}: ${x.msg ?? JSON.stringify(x)}`)
      .join('；');
  }
  if (detail && typeof detail === 'object') {
    const d = detail as {
      message?: string;
      errors?: { where?: string; message: string }[];
      report?: { errors?: { where?: string; message: string }[] };
    };
    const errs = d.report?.errors ?? d.errors;
    if (errs?.length) {
      return errs.map((x) => `${x.where ? `${x.where}: ` : ''}${x.message}`).join('；');
    }
    if (d.message) return d.message;
  }
  return e instanceof Error ? e.message : fallback;
}
