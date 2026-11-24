export type Row = Record<string, any>;
export async function request<T = any>(path: string, options: RequestInit = {}): Promise<T> {
  const response = await fetch(`/api${path}`, {
    ...options, headers: { 'Content-Type': 'application/json', ...options.headers },
  });
  const value = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(value.error || value.message || `请求失败 (${response.status})`);
  return value as T;
}
export const api = {
  bootstrap: () => request<Row>('/bootstrap'),
  createInstance: (body: Row) => request<Row>('/instances', { method: 'POST', body: JSON.stringify(body) }),
  updateInstance: (id: string, body: Row) => request<Row>(`/instances/${encodeURIComponent(id)}`, { method: 'PATCH', body: JSON.stringify(body) }),
  replay: (id: string) => request<Row>(`/replay/${encodeURIComponent(id)}`),
  optimize: (body: Row) => request<Row>('/optimization/tasks', { method: 'POST', body: JSON.stringify(body) }),
  task: (id: string) => request<Row>(`/optimization/tasks/${encodeURIComponent(id)}`),
  cancelTask: (id: string) => request<Row>(`/optimization/tasks/${encodeURIComponent(id)}/cancel`, { method: 'POST', body: '{}' }),
  promote: (id: string, instanceId?: string) => request<Row>(`/genomes/${encodeURIComponent(id)}/promote`, { method: 'POST', body: JSON.stringify({ instance_id: instanceId || undefined }) }),
  backtest: (body: Row) => request<Row>('/backtests', { method: 'POST', body: JSON.stringify(body) }),
  backtestDetail: (id: string) => request<Row>(`/backtests/${encodeURIComponent(id)}`),
};
